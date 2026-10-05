"""Offline translation through locally installed Argos Translate packages.

Ordinary translation never touches the network and never downloads a model:
:class:`OfflineTranslator` only reads packages already installed on disk.
:func:`install_models` is the explicit, network-using entry point that resolves
the shortest route between two languages -- direct or through a pivot -- using
the Argos package index, downloads only the packages that route needs, and
installs them.

Language codes are Argos ISO 639 codes (e.g. ``"en"``, ``"ja"``); callers map
their own OCR language detection to these codes.
"""

from __future__ import annotations

from collections import deque

__all__ = ["TranslationModelError", "ArgosBackend", "OfflineTranslator", "install_models"]


class TranslationModelError(RuntimeError):
    """Raised when a requested offline translation route is not available."""


class ArgosBackend:
    """Lazy adapter over the real ``argostranslate`` package.

    The import happens in the constructor, so importing this module (or merely
    constructing an identity translator) never imports Argos, downloads a
    package index, or fetches a model.
    """

    def __init__(self) -> None:
        try:
            from argostranslate import package, translate
        except ImportError as exc:
            raise TranslationModelError(
                "argostranslate is not installed; install the offline translator "
                "with `pip install argostranslate`"
            ) from exc
        self._package = package
        self._translate = translate

    def available_packages(self) -> list:
        """Refresh the package index only for an explicit model installation."""
        self._package.update_package_index()
        return self._package.get_available_packages()

    def installed_packages(self) -> list:
        """Translation packages already installed on this machine."""
        return [
            pkg
            for pkg in self._package.get_installed_packages()
            if getattr(pkg, "type", "translate") == "translate"
        ]

    def install_from_path(self, path) -> None:
        self._package.install_from_path(path)

    def get_installed_languages(self) -> list:
        """Installed languages; Argos wires pivot routes between them."""
        return self._translate.get_installed_languages()


def _normalise(code: str, name: str) -> str:
    if not isinstance(code, str) or not code.strip():
        raise ValueError(f"{name} language code must be a non-empty string")
    return code.strip().lower()


def _adjacency(packages) -> tuple[dict[str, set[str]], dict[tuple[str, str], object]]:
    edges: dict[str, set[str]] = {}
    by_pair: dict[tuple[str, str], object] = {}
    for pkg in packages:
        if getattr(pkg, "type", "translate") != "translate":
            continue
        from_code = getattr(pkg, "from_code", None)
        to_code = getattr(pkg, "to_code", None)
        if not from_code or not to_code or from_code == to_code:
            continue
        edges.setdefault(from_code, set()).add(to_code)
        by_pair.setdefault((from_code, to_code), pkg)
    return edges, by_pair


def _shortest_route(edges: dict[str, set[str]], source: str, target: str) -> list[str] | None:
    """Breadth-first shortest path, one edge per package, deterministic order."""
    if source == target:
        return [source]
    queue = deque([[source]])
    seen = {source}
    while queue:
        path = queue.popleft()
        for nxt in sorted(edges.get(path[-1], ())):
            if nxt == target:
                return path + [nxt]
            if nxt not in seen:
                seen.add(nxt)
                queue.append(path + [nxt])
    return None


def install_models(source: str, target: str, backend: ArgosBackend | None = None) -> list[str]:
    """Install the shortest Argos route from ``source`` to ``target``.

    Uses the package index to choose the fewest-package route, direct or via
    pivots, and downloads only the packages on that route that are not already
    installed.  ``source == target`` is identity: nothing to install.

    Returns the ordered route as ``"from->to"`` edge strings.  Raises
    :class:`TranslationModelError` when the index offers no such route.
    """
    source = _normalise(source, "source")
    target = _normalise(target, "target")
    if source == target:
        return []

    backend = backend if backend is not None else ArgosBackend()
    installed_packages = backend.installed_packages()
    installed_edges, _ = _adjacency(installed_packages)
    already_installed = _shortest_route(installed_edges, source, target)
    if already_installed is not None:
        return [f"{left}->{right}" for left, right in zip(already_installed, already_installed[1:])]

    edges, by_pair = _adjacency([*installed_packages, *backend.available_packages()])
    route = _shortest_route(edges, source, target)
    if route is None:
        raise TranslationModelError(
            f"No Argos package route from '{source}' to '{target}' exists in the "
            "package index; check whether models are published for both languages"
        )

    installed = {(pkg.from_code, pkg.to_code) for pkg in installed_packages}
    route_edges: list[str] = []
    for from_code, to_code in zip(route, route[1:]):
        route_edges.append(f"{from_code}->{to_code}")
        if (from_code, to_code) in installed:
            continue
        package = by_pair.get((from_code, to_code))
        if package is None:
            raise TranslationModelError(
                f"The package index lists no download for '{from_code}->{to_code}'"
            )
        backend.install_from_path(package.download())
    return route_edges


class OfflineTranslator:
    """Translate text with models already installed, never with the network.

    ``source == target`` is treated as identity and returns the text untouched,
    without importing Argos.  Any other pair resolves a translation among the
    installed Argos languages; Argos wires pivot routes, so a pair that only
    exists through an intermediary also resolves.  A missing model or route
    raises :class:`TranslationModelError` naming the installer to run.
    """

    def __init__(self, source: str, target: str, backend: ArgosBackend | None = None) -> None:
        self.source = _normalise(source, "source")
        self.target = _normalise(target, "target")
        self._backend = backend
        self._translation = None

    def _resolve(self):
        if self._translation is not None:
            return self._translation
        backend = self._backend
        if backend is None:
            backend = self._backend = ArgosBackend()
        languages = {language.code: language for language in backend.get_installed_languages()}
        missing = [code for code in (self.source, self.target) if code not in languages]
        if missing:
            missing_text = ", ".join(repr(code) for code in missing)
            raise TranslationModelError(
                f"No installed offline model for language {missing_text}; run "
                f"install_models({self.source!r}, {self.target!r}) to download the "
                "required Argos package(s)"
            )
        translation = languages[self.source].get_translation(languages[self.target])
        if translation is None:
            raise TranslationModelError(
                f"No installed offline route from '{self.source}' to '{self.target}'; run "
                f"install_models({self.source!r}, {self.target!r}) to download it"
            )
        self._translation = translation
        return translation

    def ensure_ready(self) -> None:
        """Validate the installed language route without translating sample text."""
        if self.source != self.target:
            self._resolve()

    def translate(self, text: str) -> str:
        """Return ``text`` translated, using installed models only."""
        if self.source == self.target or not text:
            return text
        return self._resolve().translate(text)
