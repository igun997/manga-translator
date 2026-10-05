"""Detect enclosed light speech bubbles, read their text, and reletter safely."""

import argparse
import csv
import io
import json
import subprocess
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass
class Bubble:
    bbox: tuple[int, int, int, int]  # left, top, right, bottom; exclusive
    mask: np.ndarray  # full-page uint8 filled interior
    safe_mask: np.ndarray  # interior with clearance from the outline


# A rounded balloon interior fills less of its box than a rectangular caption.
_BALLOON_MAX_FILL = 0.87


def detect_bubbles(image: Image.Image, min_area: int | None = None) -> list[Bubble]:
    """Find closed or open rounded balloon interiors, never captions."""
    return _detect(image, min_area, include_captions=False)


def detect_text_regions(image: Image.Image, min_area: int | None = None) -> list[Bubble]:
    """Find caption interiors and closed or open speech balloons."""
    return _detect(image, min_area, include_captions=True)


def _detect(image: Image.Image, min_area: int | None, include_captions: bool) -> list[Bubble]:
    """Combine light-region geometry with the bundled balloon segmenter.

    The geometric pass misses balloons whose interior joins the page
    background; the segmenter contributes those, deduplicated against the
    geometric regions so a balloon is never OCR'd twice.
    """
    from balloon_detector import detect_model_bubbles

    regions = _find_light_regions(image, min_area, include_captions)
    for proposal in detect_model_bubbles(image, min_area):
        if not include_captions and not _is_rounded(proposal):
            continue  # rectangular narration stays out of the balloon-only path
        proposed_area = np.count_nonzero(proposal.safe_mask)
        if not proposed_area:
            continue
        if any(
            np.count_nonzero(cv2.bitwise_and(proposal.safe_mask, existing.safe_mask))
            > 0.5 * min(proposed_area, np.count_nonzero(existing.safe_mask))
            for existing in regions
        ):
            continue
        regions.append(proposal)
    return sorted(regions, key=lambda region: (region.bbox[1], region.bbox[0]))


def _is_rounded(bubble: Bubble) -> bool:
    left, top, right, bottom = bubble.bbox
    interior = bubble.mask[top:bottom, left:right]
    fill_ratio = np.count_nonzero(interior) / interior.size
    width, height = right - left, bottom - top
    return (0.48 <= fill_ratio <= _BALLOON_MAX_FILL
            and 0.35 <= width / height <= 3.0)



def _find_light_regions(image: Image.Image, min_area: int | None,
                        include_captions: bool) -> list[Bubble]:
    gray = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    light = cv2.morphologyEx(
        (gray >= 205).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(light, 8)
    height, width = gray.shape
    area_limit = min_area if min_area is not None else max(400, gray.size // 400)
    bubbles = []
    for label in range(1, count):
        x, y, w, h, area = (int(v) for v in stats[label])
        fill_ratio = area / (w * h)
        if (x == 0 or y == 0 or x + w == width or y + h == height
                or area < area_limit or w < 20 or h < 20
                or not (0.2 if include_captions else 0.35) <= w / h <= (6.0 if include_captions else 3.0)
                or not 0.48 <= fill_ratio <= (0.98 if include_captions else _BALLOON_MAX_FILL)):
            continue
        region = (labels[y:y + h, x:x + w] == label).astype(np.uint8)
        contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) / cv2.contourArea(cv2.convexHull(contour)) < 0.94:
            continue
        filled = np.zeros_like(region)
        cv2.drawContours(filled, [contour], -1, 255, cv2.FILLED)
        mask = np.zeros_like(gray)
        mask[y:y + h, x:x + w] = filled
        # Erosion is the clearance both for removal and for new glyphs.
        safe = cv2.erode(mask, np.ones((9, 9), np.uint8))
        bubbles.append(Bubble((x, y, x + w, y + h), mask, safe))
    return sorted(bubbles, key=lambda b: (b.bbox[1], b.bbox[0]))


def recognize_bubbles(image: Image.Image, bubbles: list[Bubble], lang: str = "eng",
                      psm: int = 6) -> list[dict]:
    """Run Tesseract on each interior; return text and page-relative word boxes."""
    result = []
    rgb = np.asarray(image.convert("RGB"))
    for index, bubble in enumerate(bubbles, 1):
        x, y, right, bottom = bubble.bbox
        crop = rgb[y:bottom, x:right].copy()
        crop[bubble.mask[y:bottom, x:right] == 0] = 255
        encoded = io.BytesIO()
        Image.fromarray(crop).save(encoded, format="PNG")
        try:
            output = subprocess.run(
                ["tesseract", "stdin", "stdout", "-l", lang, "--psm", str(psm), "tsv"],
                input=encoded.getvalue(), capture_output=True, check=True,
            )
        except FileNotFoundError as exc:
            raise RuntimeError("Tesseract executable is required; install tesseract-ocr") from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(exc.stderr.decode(errors="replace").strip()) from exc
        words = []
        for row in csv.DictReader(io.StringIO(output.stdout.decode("utf-8")), delimiter="\t"):
            text = row["text"].strip()
            if not text or float(row["conf"]) < 0:
                continue
            left = x + int(row["left"])
            top = y + int(row["top"])
            word_right = left + int(row["width"])
            word_bottom = top + int(row["height"])
            box_mask = bubble.safe_mask[top:word_bottom, left:word_right]
            if box_mask.size == 0 or np.count_nonzero(box_mask) / box_mask.size < 0.7:
                continue
            words.append({"text": text, "bbox": [left, top, word_right, word_bottom]})
        result.append({"id": index, "bbox": list(bubble.bbox),
                       "text": " ".join(word["text"] for word in words), "words": words})
    return result


def _wrap(text: str, font: ImageFont.FreeTypeFont, width: int) -> str | None:
    lines = []
    for paragraph in text.split("\n"):
        line = ""
        for index, word in enumerate(paragraph.split()):
            # CJK text normally has no spaces; break it at character boundaries.
            units = list(word) if any(unicodedata.east_asian_width(c) in "WF" for c in word) else [word]
            for unit_index, unit in enumerate(units):
                if font.getlength(unit) > width:
                    return None
                separator = " " if index and unit_index == 0 and line else ""
                candidate = line + separator + unit
                if font.getlength(candidate) > width and line:
                    lines.append(line)
                    line = unit
                else:
                    line = candidate
        lines.append(line)
    return "\n".join(lines)


def fit_text(text: str, bubble: Bubble, font_path: str, min_font_size: int = 8) -> np.ndarray:
    """Render only if *all* glyph pixels fit in the eroded speech-bubble mask."""
    if not text.strip():
        raise ValueError("Replacement text cannot be empty")
    x, y, right, bottom = bubble.bbox
    width, height = right - x, bottom - y
    for size in range(min(42, height // 3), min_font_size - 1, -1):
        font = ImageFont.truetype(font_path, size)
        for line_width in (int(width * fraction) for fraction in (0.65, 0.75, 0.85, 0.95)):
            wrapped = _wrap(text, font, line_width)
            if wrapped is None:
                continue
            spacing = max(2, size // 5)
            canvas = Image.new("L", (width, height))
            painter = ImageDraw.Draw(canvas)
            left, top, text_right, text_bottom = painter.multiline_textbbox(
                (0, 0), wrapped, font=font, spacing=spacing, align="center"
            )
            if text_right - left > width or text_bottom - top > height:
                continue
            for vertical_shift in (0, -height // 12, height // 12, -height // 6, height // 6):
                canvas.paste(0, (0, 0, width, height))
                painter.multiline_text(
                    ((width - (text_right - left)) // 2 - left,
                     (height - (text_bottom - top)) // 2 - top + vertical_shift),
                    wrapped, font=font, fill=255, spacing=spacing, align="center",
                )
                glyphs = np.asarray(canvas)
                if np.any(glyphs) and not np.any((glyphs > 0) & (bubble.safe_mask[y:bottom, x:right] == 0)):
                    full = np.zeros_like(bubble.mask)
                    full[y:bottom, x:right] = glyphs
                    return full
    raise ValueError("Replacement text does not fit inside bubble at the minimum font size")


def replace_text(image: Image.Image, bubble: Bubble, words: list[dict], translation: str,
                 font_path: str, min_font_size: int = 8) -> Image.Image:
    """Erase OCR word boxes and draw fitted text without changing pixels outside the interior."""
    if not words:
        raise ValueError("No OCR text found in bubble; refusing to erase unknown content")
    glyphs = fit_text(translation, bubble, font_path, min_font_size)
    erase = np.zeros_like(bubble.mask)
    for word in words:
        x, y, right, bottom = word["bbox"]
        cv2.rectangle(erase, (max(0, x - 3), max(0, y - 3)),
                      (min(erase.shape[1] - 1, right + 2), min(erase.shape[0] - 1, bottom + 2)),
                      255, -1)
    erase = cv2.bitwise_and(erase, bubble.safe_mask)
    base = np.asarray(image.convert("RGB"))
    clean = cv2.inpaint(base, erase, 3, cv2.INPAINT_TELEA)
    # A mask paste handles antialiased edge pixels without touching the outline.
    return Image.composite(Image.new("RGB", image.size, "black"), Image.fromarray(clean), Image.fromarray(glyphs))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("analyze", "replace"):
        cmd = sub.add_parser(command)
        cmd.add_argument("image", type=Path)
        cmd.add_argument("--lang", default="eng", help="installed Tesseract language code")
        cmd.add_argument("--psm", type=int, default=6, help="Tesseract page segmentation mode")
        if command == "replace":
            cmd.add_argument("translations", type=Path, help='JSON object: {"1": "new text"}')
            cmd.add_argument("output", type=Path)
            cmd.add_argument("--font", default="DejaVuSans.ttf", help="font file supporting target script")
    args = parser.parse_args()
    with Image.open(args.image) as source:
        image = source.convert("RGB")
    bubbles = detect_bubbles(image)
    detected = recognize_bubbles(image, bubbles, args.lang, args.psm)
    if args.command == "analyze":
        print(json.dumps(detected, indent=2, ensure_ascii=False))
        return
    translations = json.loads(args.translations.read_text(encoding="utf-8"))
    if not isinstance(translations, dict) or any(
        not isinstance(key, str) or not key.isdigit() or not isinstance(value, str)
        for key, value in translations.items()
    ):
        parser.error("translations must be a JSON object mapping bubble IDs to strings")
    if set(translations) - {str(item["id"]) for item in detected}:
        parser.error("translations contain IDs not found on this page")
    for bubble, item in zip(bubbles, detected):
        key = str(item["id"])
        if key in translations:
            image = replace_text(image, bubble, item["words"], translations[key], args.font)
    image.save(args.output)


if __name__ == "__main__":
    main()
