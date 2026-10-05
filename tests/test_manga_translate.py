import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from manga_bubbles import detect_text_regions, recognize_bubbles
from manga_translate import collect_images, main, translate_page


class PlainTranslator:
    source = "en"
    target = "id"

    def ensure_ready(self):
        pass


    def translate(self, text):
        return "HALO DUNIA" if "HELLO" in text.upper() else "TEKS BARU"


class BatchTranslationTests(unittest.TestCase):
    def test_folder_and_selected_files_have_stable_distinct_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pages = root / "pages"
            chapter = pages / "chapter"
            chapter.mkdir(parents=True)
            (pages / "02.jpg").write_bytes(b"image")
            (chapter / "01.png").write_bytes(b"image")
            (pages / "notes.txt").write_text("ignore")
            output = pages / "translated"
            output.mkdir()
            (output / "old.png").write_bytes(b"result")
            selected = root / "cover.webp"
            selected.write_bytes(b"image")
            entries = collect_images([pages, selected], output)
            self.assertCountEqual([(p.name, dest.as_posix()) for p, dest in entries], [
                ("01.png", "pages/chapter/01.png"),
                ("02.jpg", "pages/02.png"),
                ("cover.webp", "cover.png"),
            ])

    def test_explicit_input_inside_output_cannot_overwrite_original(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "translated"
            output.mkdir()
            original = output / "page.png"
            Image.new("RGB", (20, 20)).save(original)
            with self.assertRaisesRegex(ValueError, "inside the output"):
                collect_images([original], output)

    def test_blank_page_is_reported_as_no_text(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "blank.png"
            output = Path(temp) / "output"
            Image.new("RGB", (200, 160), "white").save(source)
            class UnusedTranslator:
                def ensure_ready(self):
                    pass

                def translate(self, text):
                    raise AssertionError("Blank pages have nothing to translate")

            with patch("manga_translate.OfflineTranslator", return_value=UnusedTranslator()):
                with redirect_stdout(io.StringIO()):
                    result = main(["translate", str(source), "--source", "en",
                                   "--target", "id", "--output", str(output)])
            report = json.loads((output / "translation-report.json").read_text())
            self.assertEqual(result, 0)
            self.assertEqual(report["pages"][0]["status"], "no_text")
            self.assertTrue((output / "blank.png").is_file())

    def test_missing_llm_config_is_a_cli_error_only_when_opted_in(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "blank.png"
            Image.new("RGB", (200, 160), "white").save(source)
            missing = Path(temp) / "missing.yml"
            with patch("manga_translate.OfflineTranslator", return_value=PlainTranslator()):
                with redirect_stderr(io.StringIO()) as stderr:
                    with self.assertRaises(SystemExit) as caught:
                        main(["translate", str(source), "--source", "en", "--target", "id",
                              "--llm", "--models-file", str(missing)])
            self.assertEqual(caught.exception.code, 2)
            self.assertIn("not found", stderr.getvalue())

    def test_caption_is_translated_and_outside_interior_is_unchanged(self):
        image = Image.new("RGB", (420, 260), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((80, 20, 295, 237), outline="black", width=5)
        draw.text((132, 112), "HELLO", font=ImageFont.truetype("DejaVuSans.ttf", 25), fill="black")
        regions = detect_text_regions(image)
        self.assertEqual(len(regions), 1)
        class RecordingTranslator(PlainTranslator):
            def translate(self, text):
                self.input_text = text
                return super().translate(text)

        translator = RecordingTranslator()
        output, items = translate_page(image, "en", "id", "eng", 6,
                                       "DejaVuSans.ttf", translator)
        self.assertEqual(items[0]["source"], "HELLO")
        self.assertEqual(translator.input_text, "Hello")
        self.assertEqual(items[0]["status"], "translated")
        self.assertEqual(items[0]["translation"], "HALO DUNIA")
        original = np.asarray(image)
        edited = np.asarray(output)
        changed = np.any(original != edited, axis=2)
        self.assertTrue(np.any(changed))
        self.assertFalse(np.any(changed & (regions[0].safe_mask == 0)))
        self.assertIn("HALO", recognize_bubbles(output, regions)[0]["text"].upper())

    def test_unfittable_translation_does_not_erase_original(self):
        image = Image.new("RGB", (240, 185), "white")
        draw = ImageDraw.Draw(image)
        draw.ellipse((35, 20, 190, 165), outline="black", width=5)
        draw.text((76, 79), "HELLO", font=ImageFont.truetype("DejaVuSans.ttf", 17), fill="black")
        class TooLong:
            def translate(self, text):
                return "impossibly long untranslatedword" * 400
        output, items = translate_page(image, "en", "id", "eng", 6,
                                       "DejaVuSans.ttf", TooLong())
        self.assertEqual(items[0]["status"], "unfittable")
        self.assertTrue(np.array_equal(np.asarray(image), np.asarray(output)))


if __name__ == "__main__":
    unittest.main()
