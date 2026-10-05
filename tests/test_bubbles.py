import unittest

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from manga_bubbles import detect_bubbles, detect_text_regions, fit_text, recognize_bubbles, replace_text


def _open_balloon_page() -> Image.Image:
    """A page whose balloon interior joins the white background through a gap.

    Geometry alone cannot separate this interior from the page, so the learned
    segmenter is the only thing that can supply it. ``EXTRA`` sits inside the
    balloon's bounding box but outside its outline: any detection that widened
    into the full box would read it as dialogue.
    """
    image = Image.new("RGB", (900, 1280), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 900, 300), outline="black", width=6)
    for offset in range(0, 280, 9):
        draw.line((16, 18 + offset, 884, 52 + offset), fill=(105, 105, 105), width=1)
    draw.ellipse((300, 700, 640, 1120), outline="black", width=5)
    draw.rectangle((638, 880, 652, 940), fill="white")  # gap in the outline
    draw.text((606, 706), "EXTRA", font=ImageFont.truetype("DejaVuSans.ttf", 26), fill="black")
    draw.multiline_text(
        (342, 780), "HA...HA...\n15 YEARS OF LIFE.\nIT'S A BIT SHORT,\nISN'T IT?",
        font=ImageFont.truetype("DejaVuSans.ttf", 30), fill="black", align="center", spacing=6,
    )
    return image


class BubbleTests(unittest.TestCase):
    def setUp(self):
        image = Image.new("RGB", (400, 240), "white")
        draw = ImageDraw.Draw(image)
        draw.ellipse((35, 20, 220, 204), fill="white", outline="black", width=5)
        draw.text((102, 91), "HELLO", font=ImageFont.truetype("DejaVuSans.ttf", 20), fill="black")
        draw.text((280, 100), "NOISE", font=ImageFont.truetype("DejaVuSans.ttf", 18), fill="black")
        self.image = image

    def test_detects_enclosed_bubble_not_page(self):
        bubbles = detect_bubbles(self.image)
        self.assertEqual(len(bubbles), 1)
        x, y, right, bottom = bubbles[0].bbox
        self.assertLess(x, 50)
        self.assertGreater(right, 200)
        self.assertLess(y, 40)
        self.assertGreater(bottom, 190)
        self.assertEqual(bubbles[0].mask[100, 300], 0)
        self.assertEqual(bubbles[0].mask[100, 110], 255)

    def test_rejects_caption_and_closes_small_outline_gap(self):
        image = Image.new("RGB", (500, 280), "white")
        draw = ImageDraw.Draw(image)
        draw.ellipse((40, 30, 220, 230), outline="black", width=3)
        draw.rectangle((129, 30, 132, 34), fill="white")  # broken outline
        draw.rectangle((300, 40, 460, 230), outline="black", width=4)
        bubbles = detect_bubbles(image)
        self.assertEqual(len(bubbles), 1)
        self.assertLess(bubbles[0].bbox[0], 70)

    def test_detect_text_regions_includes_rectangular_narration(self):
        image = Image.new("RGB", (500, 280), "white")
        draw = ImageDraw.Draw(image)
        draw.ellipse((35, 30, 218, 235), outline="black", width=4)
        draw.rectangle((300, 25, 460, 244), outline="black", width=4)
        draw.text((333, 110), "A MEMORY", font=ImageFont.truetype("DejaVuSans.ttf", 15), fill="black")
        regions = detect_text_regions(image)
        self.assertEqual(len(regions), 2)
        self.assertEqual(len([r for r in regions if r.bbox[0] > 250]), 1)
        self.assertFalse(np.any(regions[1].safe_mask[100, 270]))

    def test_open_balloon_is_detected_without_claiming_page_background(self):
        page = _open_balloon_page()
        regions = detect_text_regions(page)
        open_bubble = next(region for region in regions if region.safe_mask[910, 470])
        self.assertTrue(300 <= open_bubble.bbox[0] <= 320)
        self.assertTrue(700 <= open_bubble.bbox[1] <= 720)
        # The interior must stop at the outline, not spread over the page or
        # over the neighbouring lettering inside the bounding box.
        self.assertEqual(open_bubble.safe_mask[910, 660], 0)
        self.assertEqual(open_bubble.safe_mask[720, 620], 0)

    def test_relettering_path_also_finds_the_open_balloon(self):
        page = _open_balloon_page()
        bubbles = detect_bubbles(page)
        self.assertTrue(any(bubble.safe_mask[910, 470] for bubble in bubbles))

    def test_open_balloon_ocr_excludes_the_rest_of_the_page(self):
        page = _open_balloon_page()
        open_bubble = next(region for region in detect_text_regions(page)
                           if region.safe_mask[910, 470])
        text = recognize_bubbles(page, [open_bubble], lang="eng")[0]["text"]
        self.assertIn("15 YEARS OF LIFE", text.upper())
        self.assertIn("SHORT", text.upper())
        self.assertNotIn("EXTRA", text.upper())

    def test_fitted_glyphs_stay_inside_curved_interior(self):
        bubble = detect_bubbles(self.image)[0]
        glyphs = fit_text("A longer translation that needs several lines", bubble, "DejaVuSans.ttf")
        self.assertTrue(np.any(glyphs))
        self.assertFalse(np.any((glyphs > 0) & (bubble.safe_mask == 0)))

    def test_wraps_unspaced_japanese_without_crossing_bubble_edge(self):
        bubble = detect_bubbles(self.image)[0]
        glyphs = fit_text("今日は新しい冒険について話しましょう" * 5, bubble, "DejaVuSans.ttf")
        self.assertTrue(np.any(glyphs))
        self.assertFalse(np.any((glyphs > 0) & (bubble.safe_mask == 0)))

    def test_unfittable_text_fails_instead_of_clipping(self):
        bubble = detect_bubbles(self.image)[0]
        with self.assertRaisesRegex(ValueError, "does not fit"):
            fit_text("translation " * 200, bubble, "DejaVuSans.ttf", min_font_size=12)

    def test_replacement_preserves_outside_pixels(self):
        bubble = detect_bubbles(self.image)[0]
        words = [{"text": "HELLO", "bbox": [102, 91, 164, 114]}]
        result = replace_text(self.image, bubble, words, "A new translation", "DejaVuSans.ttf")
        before = np.asarray(self.image)
        after = np.asarray(result)
        changed = np.any(before != after, axis=2)
        self.assertTrue(np.any(changed))
        self.assertFalse(np.any(changed & (bubble.safe_mask == 0)))
        self.assertTrue(np.array_equal(before[100, 300], after[100, 300]))

    def test_ocr_reads_text_inside_only(self):
        bubbles = detect_bubbles(self.image)
        detected = recognize_bubbles(self.image, bubbles, lang="eng")
        self.assertEqual(len(detected), 1)
        self.assertIn("HELLO", detected[0]["text"].upper())
        self.assertNotIn("NOISE", detected[0]["text"].upper())
        self.assertTrue(detected[0]["words"])


if __name__ == "__main__":
    unittest.main()
