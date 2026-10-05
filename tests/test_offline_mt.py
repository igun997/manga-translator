"""Behaviour tests for the offline Argos translation wrapper.

The fakes below mirror Argos semantics (identity translations plus pivot
composites wired by ``get_installed_languages``) so the wrapper's routing,
identity short-circuit and error paths are exercised without downloading
models or touching the network.  A guarded integration test uses real installed
models when the ``argostranslate`` package and a language pair happen to exist.
"""

import sys
import unittest
from pathlib import Path
from unittest import mock

import offline_mt
from offline_mt import ArgosBackend, OfflineTranslator, TranslationModelError, install_models



# --- deterministic fakes ---------------------------------------------------


class _Identity:
    def __init__(self, language):
        self.from_lang = self.to_lang = language

    def translate(self, text):
        return text


class _Direct:
    def __init__(self, from_lang, to_lang, mapping):
        self.from_lang = from_lang
        self.to_lang = to_lang
        self.mapping = mapping

    def translate(self, text):
        return " ".join(self.mapping.get(word, word) for word in text.split())


class _Composite:
    def __init__(self, first, second):
        self.first = first
        self.second = second
        self.from_lang = first.from_lang
        self.to_lang = second.to_lang

    def translate(self, text):
        return self.second.translate(self.first.translate(text))


class _Language:
    def __init__(self, code):
        self.code = code
        self.translations_from = []

    def get_translation(self, to):
        for translation in self.translations_from:
            if translation.to_lang.code == to.code:
                return translation
        return None


def _languages(edges, extra=()):
    """Build languages with identity + pivot composites, as Argos does."""
    languages = {}

    def language(code):
        if code not in languages:
            languages[code] = _Language(code)
        return languages[code]

    for code in extra:
        language(code)
    for (source, target), mapping in edges.items():
        language(source).translations_from.append(
            _Direct(language(source), language(target), mapping)
        )
    for lang in languages.values():
        lang.translations_from.insert(0, _Identity(lang))
    for lang in languages.values():
        adding = True
        while adding:
            adding = False
            for first in list(lang.translations_from):
                for second in list(first.to_lang.translations_from):
                    if lang.get_translation(second.to_lang) is None:
                        adding = True
                        lang.translations_from.append(_Composite(first, second))
    return list(languages.values())


class _FakePackage:
    type = "translate"

    def __init__(self, from_code, to_code):
        self.from_code = from_code
        self.to_code = to_code
        self.downloads = 0

    def download(self):
        self.downloads += 1
        return Path(f"{self.from_code}_{self.to_code}.argosmodel")


class _FakeBackend:
    def __init__(self, available=(), installed_pairs=(), languages=(), forbid_index=False):
        self.available = list(available)
        self.installed_pairs = set(installed_pairs)
        self.languages = list(languages)
        self.installed_paths = []
        self.forbid_index = forbid_index

    def available_packages(self):
        if self.forbid_index:
            raise AssertionError("the package index must not be used during translate")
        return list(self.available)

    def installed_packages(self):
        return [
            pkg for pkg in self.available if (pkg.from_code, pkg.to_code) in self.installed_pairs
        ]

    def install_from_path(self, path):
        path = Path(path)
        self.installed_paths.append(path)
        source, target = path.stem.split("_")
        self.installed_pairs.add((source, target))

    def get_installed_languages(self):
        return list(self.languages)


# --- translation -----------------------------------------------------------


class OfflineTranslatorTests(unittest.TestCase):
    def test_direct_route_translates_without_using_the_index(self):
        edges = {("en", "fr"): {"hello": "bonjour", "world": "monde"}}
        backend = _FakeBackend(languages=_languages(edges), forbid_index=True)
        translator = OfflineTranslator("en", "fr", backend=backend)
        self.assertEqual(translator.translate("hello world"), "bonjour monde")

    def test_pivot_route_translates_through_intermediate_language(self):
        edges = {
            ("en", "fr"): {"hello": "bonjour", "world": "monde"},
            ("fr", "ja"): {"bonjour": "konnichiwa", "monde": "sekai"},
        }
        backend = _FakeBackend(languages=_languages(edges), forbid_index=True)
        out = OfflineTranslator("en", "ja", backend=backend).translate("hello world")
        self.assertEqual(out, "konnichiwa sekai")

    def test_languages_are_normalised(self):
        edges = {("en", "fr"): {"hello": "bonjour"}}
        backend = _FakeBackend(languages=_languages(edges), forbid_index=True)
        self.assertEqual(OfflineTranslator(" EN ", "Fr", backend=backend).translate("hello"), "bonjour")

    def test_identity_returns_text_without_touching_argos(self):
        with mock.patch.object(offline_mt, "ArgosBackend", side_effect=AssertionError("imported")):
            self.assertEqual(OfflineTranslator("ja", "ja").translate("こんにちは"), "こんにちは")

    def test_identity_does_not_use_a_supplied_backend(self):
        backend = _FakeBackend(forbid_index=True)
        self.assertEqual(OfflineTranslator("en", "en", backend=backend).translate("same"), "same")

    def test_missing_route_is_reported_before_processing_pages(self):
        backend = _FakeBackend(languages=_languages({}, extra=("en", "id")), forbid_index=True)
        translator = OfflineTranslator("en", "id", backend=backend)
        with self.assertRaisesRegex(TranslationModelError, "No installed offline route"):
            translator.ensure_ready()

    def test_missing_language_reports_the_installer(self):
        edges = {("en", "fr"): {"hello": "bonjour"}}
        backend = _FakeBackend(languages=_languages(edges), forbid_index=True)
        translator = OfflineTranslator("en", "ja", backend=backend)
        with self.assertRaisesRegex(TranslationModelError, r"install_models\('en', 'ja'\)"):
            translator.translate("hello")

    def test_missing_route_between_known_languages_is_explicit(self):
        backend = _FakeBackend(languages=_languages({}, extra=("en", "ja")), forbid_index=True)
        with self.assertRaisesRegex(TranslationModelError, "No installed offline route"):
            OfflineTranslator("en", "ja", backend=backend).translate("hello")

    def test_absent_argostranslate_is_actionable(self):
        saved = sys.modules.get("argostranslate")
        sys.modules["argostranslate"] = None
        try:
            with self.assertRaisesRegex(TranslationModelError, "pip install argostranslate"):
                ArgosBackend()
        finally:
            if saved is None:
                sys.modules.pop("argostranslate", None)
            else:
                sys.modules["argostranslate"] = saved

    def test_empty_translation_input_is_passed_through(self):
        edges = {("en", "fr"): {"hello": "bonjour"}}
        backend = _FakeBackend(languages=_languages(edges), forbid_index=True)
        self.assertEqual(OfflineTranslator("en", "fr", backend=backend).translate(""), "")

    def test_invalid_language_codes_are_rejected(self):
        with self.assertRaises(ValueError):
            OfflineTranslator("", "en")
        with self.assertRaises(ValueError):
            OfflineTranslator("en", "   ")


# --- installation ----------------------------------------------------------


class _ColdIndex:
    def __init__(self):
        self.updated = False
        self.pair = _FakePackage("en", "id")
        self.installed = []

    def update_package_index(self):
        self.updated = True

    def get_available_packages(self):
        return [self.pair] if self.updated else []

    def get_installed_packages(self):
        return self.installed

    def install_from_path(self, path):
        self.installed.append(self.pair)


class InstallModelsTests(unittest.TestCase):
    def test_direct_route_installs_one_package(self):
        package = _FakePackage("en", "fr")
        backend = _FakeBackend(available=[package])
        self.assertEqual(install_models("en", "fr", backend=backend), ["en->fr"])
        self.assertEqual(package.downloads, 1)
        self.assertEqual(backend.installed_paths, [Path("en_fr.argosmodel")])

    def test_cold_package_index_is_refreshed_before_install(self):
        cold = _ColdIndex()
        backend = ArgosBackend.__new__(ArgosBackend)
        backend._package = cold
        self.assertEqual(install_models("en", "id", backend=backend), ["en->id"])
        self.assertTrue(cold.updated)
        self.assertEqual(cold.installed, [cold.pair])

    def test_existing_pivot_route_needs_no_network(self):
        pairs = [_FakePackage("en", "fr"), _FakePackage("fr", "ja")]
        backend = _FakeBackend(available=pairs, installed_pairs={("en", "fr"), ("fr", "ja")},
                               forbid_index=True)
        self.assertEqual(install_models("en", "ja", backend=backend), ["en->fr", "fr->ja"])
        self.assertEqual(backend.installed_paths, [])

    def test_shortest_route_prefers_direct_over_pivot(self):
        direct = _FakePackage("en", "ja")
        backend = _FakeBackend(
            available=[direct, _FakePackage("en", "fr"), _FakePackage("fr", "ja")]
        )
        self.assertEqual(install_models("en", "ja", backend=backend), ["en->ja"])
        self.assertEqual(direct.downloads, 1)
        self.assertEqual(len(backend.installed_paths), 1)

    def test_shortest_route_prefers_two_hops_over_three(self):
        backend = _FakeBackend(
            available=[
                _FakePackage("en", "fr"),
                _FakePackage("fr", "ja"),
                _FakePackage("en", "de"),
                _FakePackage("de", "nl"),
                _FakePackage("nl", "ja"),
            ]
        )
        self.assertEqual(install_models("en", "ja", backend=backend), ["en->fr", "fr->ja"])
        self.assertEqual(len(backend.installed_paths), 2)

    def test_already_installed_edges_are_not_downloaded_again(self):
        en_fr = _FakePackage("en", "fr")
        fr_ja = _FakePackage("fr", "ja")
        backend = _FakeBackend(
            available=[en_fr, fr_ja], installed_pairs={("en", "fr")}
        )
        self.assertEqual(install_models("en", "ja", backend=backend), ["en->fr", "fr->ja"])
        self.assertEqual(en_fr.downloads, 0)
        self.assertEqual(fr_ja.downloads, 1)
        self.assertEqual(backend.installed_paths, [Path("fr_ja.argosmodel")])

    def test_unavailable_route_is_explicit(self):
        backend = _FakeBackend(available=[_FakePackage("en", "fr")])
        with self.assertRaisesRegex(TranslationModelError, "No Argos package route from 'en' to 'ja'"):
            install_models("en", "ja", backend=backend)

    def test_identity_installs_nothing_and_ignores_the_index(self):
        backend = _FakeBackend(forbid_index=True)
        self.assertEqual(install_models("ja", "JA", backend=backend), [])
        self.assertEqual(backend.installed_paths, [])

    def test_invalid_language_codes_are_rejected(self):
        with self.assertRaises(ValueError):
            install_models("", "en")




if __name__ == "__main__":
    unittest.main()
