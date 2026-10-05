"""Batch manga OCR, offline machine translation, and optional LLM post-editing."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from manga_bubbles import Bubble, detect_text_regions, recognize_bubbles, replace_text
from offline_mt import OfflineTranslator, TranslationModelError, install_models

IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"})
OCR_CODES = {
    "en": "eng", "id": "ind", "ja": "jpn", "zh": "chi_sim", "ko": "kor",
    "es": "spa", "fr": "fra", "de": "deu", "pt": "por", "it": "ita",
    "ru": "rus", "ar": "ara", "hi": "hin", "th": "tha", "vi": "vie",
    "tr": "tur", "nl": "nld", "pl": "pol", "uk": "ukr", "ms": "msa",
}
CJK_FONT = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")


def collect_images(inputs: list[Path], output_dir: Path) -> list[tuple[Path, Path]]:
    """List selected images in stable order with collision-free relative PNG outputs."""
    entries = []
    seen_sources = set()
    seen_outputs = set()
    multiple = len(inputs) > 1
    output_root = output_dir.resolve()
    for entry in inputs:
        root = entry.resolve()
        if not root.exists():
            raise ValueError(f"Input does not exist: {entry}")
        if root.is_file():
            if root.is_relative_to(output_root):
                raise ValueError(f"Input is inside the output directory: {entry}")
            candidates = [(root, Path(root.name))]
        elif root.is_dir():
            candidates = [
                (image, Path(root.name) / image.relative_to(root) if multiple
                 else image.relative_to(root))
                for image in sorted(root.rglob("*"))
                if image.is_file() and image.suffix.lower() in IMAGE_SUFFIXES
                and not image.resolve().is_relative_to(output_root)
            ]
        else:
            raise ValueError(f"Unsupported input: {entry}")
        for image, relative in candidates:
            if image.suffix.lower() not in IMAGE_SUFFIXES:
                raise ValueError(f"Not an image: {image}")
            image = image.resolve()
            if image in seen_sources:
                continue
            destination = relative.with_suffix(".png")
            if destination in seen_outputs:
                raise ValueError(f"Input files would share output path: {destination}")
            seen_sources.add(image)
            seen_outputs.add(destination)
            entries.append((image, destination))
    if not entries:
        raise ValueError("No supported images found in the selected inputs")
    return entries


def _manual_regions(image: Image.Image, annotations: list[dict]) -> list[Bubble]:
    height, width = image.height, image.width
    regions = []
    for item in annotations:
        mask = np.zeros((height, width), dtype=np.uint8)
        if "polygon" in item:
            points = np.asarray(item["polygon"], dtype=np.int32)
            if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
                raise ValueError("A region polygon needs at least three [x, y] points")
            if np.any(points[:, 0] < 0) or np.any(points[:, 0] >= width) or np.any(points[:, 1] < 0) or np.any(points[:, 1] >= height):
                raise ValueError("A region polygon extends outside the image")
            cv2.fillPoly(mask, [points], 255)
        elif "bbox" in item:
            x, y, right, bottom = map(int, item["bbox"])
            if not (0 <= x < right <= width and 0 <= y < bottom <= height):
                raise ValueError("A region bbox must be inside the image")
            mask[y:bottom, x:right] = 255
        else:
            raise ValueError("Each manual region needs a polygon or bbox")
        ys, xs = np.nonzero(mask)
        if not len(xs):
            raise ValueError("A manual region must have nonzero area")
        bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
        safe = cv2.erode(mask, np.ones((9, 9), np.uint8))
        regions.append(Bubble(bbox, mask, safe))
    return regions


def translate_page(image: Image.Image, source: str, target: str, ocr_lang: str,
                   psm: int, font_path: str, translator: OfflineTranslator,
                   corrector=None, annotations: list[dict] | None = None) -> tuple[Image.Image, list[dict]]:
    """Translate detected interiors; leave failed or unrecognized regions intact."""
    original = image.convert("RGB")
    regions = detect_text_regions(original)
    for manual in _manual_regions(original, annotations or []):
        # Do not OCR/translate an enclosed region twice.
        if any(np.count_nonzero(cv2.bitwise_and(manual.safe_mask, region.safe_mask))
               > np.count_nonzero(manual.safe_mask) * 0.5 for region in regions):
            continue
        regions.append(manual)
    regions.sort(key=lambda region: (region.bbox[1], region.bbox[0]))
    detected = recognize_bubbles(original, regions, lang=ocr_lang, psm=psm)
    output = original.copy()
    results = []
    for region, item in zip(regions, detected):
        source_text = item["text"]
        if source in {"ja", "zh"}:
            source_text = "".join(word["text"] for word in item["words"])
        record = {"id": item["id"], "bbox": item["bbox"], "source": source_text,
                  "translation": "", "status": "no_text"}
        if not item["words"]:
            results.append(record)
            continue
        try:
            # Argos treats all-caps comic lettering as out-of-vocabulary tokens.
            mt_input = source_text.capitalize() if source_text.isupper() else source_text
            draft = translator.translate(mt_input).strip()
            if not draft or draft == source_text:
                record["status"] = "untranslated"
                results.append(record)
                continue
            candidate = corrector.correct(source_text, draft, source, target) if corrector else draft
            try:
                updated = replace_text(output, region, item["words"], candidate, font_path)
                record["translation"] = candidate
                record["status"] = "translated"
            except ValueError:
                if candidate == draft:
                    raise
                updated = replace_text(output, region, item["words"], draft, font_path)
                record["translation"] = draft
                record["status"] = "offline_fallback"
            output = updated
        except ValueError as exc:
            record["status"] = "unfittable"
            record["error"] = str(exc)
        except (TranslationModelError, RuntimeError) as exc:
            record["status"] = "failed"
            record["error"] = str(exc)
        results.append(record)
    return output, results


def _ocr_code(source: str, requested: str | None) -> str:
    if requested:
        return requested
    if source not in OCR_CODES:
        raise ValueError(f"No default Tesseract language for {source!r}; pass --ocr-lang")
    return OCR_CODES[source]


def _font_path(target: str, requested: str | None) -> str:
    if requested:
        return requested
    if target in {"ja", "zh", "ko"}:
        if not CJK_FONT.is_file():
            raise ValueError("A CJK font is required for this language; pass --font /path/to/font.ttf")
        return str(CJK_FONT)
    return "DejaVuSans.ttf"


def _check_ocr(lang: str) -> None:
    try:
        proc = subprocess.run(["tesseract", "--list-langs"], capture_output=True, text=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise ValueError("Install Tesseract OCR and its language data before translating") from exc
    installed = set(proc.stdout.splitlines()[1:] + proc.stderr.splitlines()[1:])
    missing = set(lang.split("+")) - installed
    if missing:
        raise ValueError(f"Tesseract language data missing: {', '.join(sorted(missing))}; install its language pack or pass --ocr-lang")


def _load_annotations(path: Path | None) -> dict:
    if path is None:
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or any(not isinstance(key, str) or not isinstance(value, list)
                                         for key, value in data.items()):
        raise ValueError("--regions must be JSON mapping image paths to region lists")
    return data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    translate = commands.add_parser("translate", help="Translate selected images or whole directories")
    translate.add_argument("inputs", nargs="+", type=Path)
    translate.add_argument("--source", required=True, help="Argos source language code, e.g. ja")
    translate.add_argument("--target", required=True, help="Argos target language code, e.g. id")
    translate.add_argument("--output", type=Path, help="Output directory (default: translated-TARGET)")
    translate.add_argument("--ocr-lang", help="Tesseract language code; overrides the source-language mapping")
    translate.add_argument("--psm", type=int, default=6, help="Tesseract layout mode (5 for vertical text)")
    translate.add_argument("--font", help="TrueType/OpenType font supporting target script")
    translate.add_argument("--regions", type=Path, help="JSON polygons/bboxes for missed open balloons")
    translate.add_argument("--overwrite", action="store_true", help="Replace existing output images")
    translate.add_argument("--llm", action="store_true", help="Post-edit offline drafts with an OpenAI-compatible LLM")
    translate.add_argument("--provider", default="netra", help="LLM provider in models.yml (only with --llm)")
    translate.add_argument("--model", help="LLM model ID (only with --llm)")
    translate.add_argument("--models-file", type=Path, help="LLM provider config (only with --llm)")
    models = commands.add_parser("models", help="Install Argos models for offline translation")
    models.add_argument("--source", required=True)
    models.add_argument("--target", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "models":
            installed = install_models(args.source, args.target)
            print("Installed offline route: " + (", ".join(installed) or "identity"))
            return 0
        source, target = args.source.strip().lower(), args.target.strip().lower()
        if source == target:
            raise ValueError("Source and target must differ for translation")
        ocr_lang = _ocr_code(source, args.ocr_lang)
        _check_ocr(ocr_lang)
        font = _font_path(target, args.font)
        output_root = args.output or Path(f"translated-{target}")
        entries = collect_images(args.inputs, output_root)
        annotations = _load_annotations(args.regions)
        translator = OfflineTranslator(source, target)
        translator.ensure_ready()
        corrector = None
        if args.llm:
            from llm_corrector import LLMCorrectorError, load_corrector
            try:
                corrector = load_corrector(config_path=args.models_file, provider=args.provider,
                                           model=args.model)
            except LLMCorrectorError as exc:
                parser.error(str(exc))
        elif args.model or args.models_file or args.provider != "netra":
            raise ValueError("--provider, --model and --models-file require --llm")
        if not Path(font).is_file():
            try:
                from PIL import ImageFont
                ImageFont.truetype(font, 12)
            except OSError as exc:
                raise ValueError(f"Font not found: {font}; pass --font") from exc
        output_root.mkdir(parents=True, exist_ok=True)
        report = {"source": source, "target": target, "llm": bool(corrector), "pages": []}
        errors = 0
        for input_path, relative in entries:
            destination = output_root / relative
            page = {"input": str(input_path), "output": str(destination), "regions": []}
            if destination.exists() and not args.overwrite:
                page["status"] = "exists"
                errors += 1
            else:
                try:
                    with Image.open(input_path) as opened:
                        image = opened.convert("RGB")
                    supplemental = annotations.get(relative.as_posix(), annotations.get(input_path.name, []))
                    translated, page["regions"] = translate_page(
                        image, source, target, ocr_lang, args.psm, font, translator,
                        corrector=corrector, annotations=supplemental)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    translated.save(destination, format="PNG")
                    statuses = {region["status"] for region in page["regions"]}
                    if not statuses or statuses == {"no_text"}:
                        page["status"] = "no_text"
                    elif statuses & {"failed", "unfittable", "untranslated"}:
                        page["status"] = "partial"
                    else:
                        page["status"] = "ok"
                    errors += page["status"] == "partial"
                except (OSError, ValueError, RuntimeError) as exc:
                    page["status"] = "failed"
                    page["error"] = str(exc)
                    errors += 1
            report["pages"].append(page)
            print(f"{page['status']}: {input_path} -> {destination}")
        report_file = output_root / "translation-report.json"
        report_file.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Report: {report_file}")
        return 1 if errors else 0
    except (ValueError, TranslationModelError, OSError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    sys.exit(main())
