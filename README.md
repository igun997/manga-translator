# manga-translator

[![CI](https://github.com/igun997/manga-translator/actions/workflows/ci.yml/badge.svg)](https://github.com/igun997/manga-translator/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/manga-translator)](https://pypi.org/project/manga-translator/)
[![Python](https://img.shields.io/pypi/pyversions/manga-translator)](https://pypi.org/project/manga-translator/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Translate a folder of manga pages, or a few selected pages, from any supported
language to any other. The CLI finds speech balloons and light narration boxes,
reads them with Tesseract, translates with locally installed Argos models, and
reletters the page in your target font.

- **Offline by default.** Detection, OCR and translation all run on your
  machine. Nothing is uploaded, and no model is downloaded during translation.
- **LLM post-editing is optional.** An OpenAI-compatible model can polish the
  offline drafts when you ask for it with `--llm`; otherwise no LLM credential
  is read and no network request is made.
- **Nothing is overwritten blind.** Outputs go to a separate directory, and every
  region that fails to translate or fit is reported instead of clipped.

Two commands ship in the package: `manga-translate` (detect → OCR → translate →
reletter) and `manga-bubbles` (analyze and reletter by hand, no translation).

---

## Requirements

| Requirement | Why | Notes |
| --- | --- | --- |
| Python 3.10+ | runtime | |
| Tesseract OCR | reads the original text | plus a language pack **per source language** |
| Argos package for your language pair | offline translation | installed once by `manga-translate models` |
| A font with your **target** script | reletters the page | CJK needs a CJK font; Latin/Cyrillic/Greek work with DejaVu |

OCR language packs, Argos translation packages and target fonts are three
independent things. Changing the target language usually means installing a
font; changing the source language usually means installing an OCR pack.

## Installation

### 1. System packages (Debian/Ubuntu)

```sh
sudo apt-get install tesseract-ocr tesseract-ocr-eng   # English source
# Japanese source also needs:
#   tesseract-ocr-jpn tesseract-ocr-jpn-vert
# A CJK font is only needed when the TARGET is Japanese/Chinese/Korean:
#   fonts-noto-cjk
```

### 2. The CLI

With `pip`:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e .
```

With [uv](https://docs.astral.sh/uv/) (faster, and what CI uses):

```sh
uv venv .venv
uv pip install --python .venv/bin/python -e .
```

Or straight from PyPI, once published:

```sh
pip install manga-translator
# or
uv tool install manga-translator
```

> `argostranslate` depends on PyTorch. On a GPU-less machine, install the CPU
> build first to avoid pulling several gigabytes of CUDA wheels:
> ```sh
> uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu
> uv pip install --python .venv/bin/python -e .
> ```

Verify:

```sh
.venv/bin/manga-translate --help
```

## Usage

### Step 1 — install the translation route (once, needs internet)

Translation needs an Argos package for the pair you want. `models` resolves the
shortest route, direct or through a pivot such as Japanese → English →
Indonesian, and downloads only the missing packages:

```sh
.venv/bin/manga-translate models --source ja --target id
```

```text
Installed offline route: ja->en, en->id
```

This is the **only** command that touches the network. Run it again for each new
pair; it is a no-op when the route is already installed.

### Step 2 — translate

```sh
# A whole folder, subfolders included
.venv/bin/manga-translate translate ./pages --source ja --target id --output ./translated-id

# Just a few pages
.venv/bin/manga-translate translate cover.jpeg page-01.jpeg \
  --source ja --target id --output ./translated-id
```

Outputs are PNGs under `--output` (default `translated-TARGET`), keeping the
input folder's relative paths. Existing outputs are left alone unless you pass
`--overwrite`.

### Language codes

`--source` / `--target` are **Argos** codes; `--ocr-lang` is a **Tesseract**
code. The Tesseract code is inferred from `--source` for the common languages,
and can always be overridden:

| | | |
| --- | --- | --- |
| `ja` → `jpn` | `en` → `eng` | `id` → `ind` |
| `zh` → `chi_sim` | `ko` → `kor` | `es` → `spa` |

```sh
# Vertical Japanese lettering
.venv/bin/manga-translate translate ./pages --source ja --target id \
  --ocr-lang jpn_vert --psm 5
```

Use `--font /path/to/font.ttf` whenever the default target font cannot render
your script; Japanese/Chinese/Korean default to
`/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc` and fail with an
actionable message if it is absent.

### Reading the report

Every run writes `translation-report.json` next to the images:

```json
{
  "source": "en", "target": "id", "llm": false,
  "pages": [{
    "input": "pages/panel-1.jpeg",
    "output": "translated-id/panel-1.png",
    "status": "ok",
    "regions": [{
      "id": 1,
      "bbox": [576, 4, 856, 436],
      "source": "AT THE MOMENT OF MY DEATH, ...",
      "translation": "Pada saat kematianku, ...",
      "status": "translated"
    }]
  }]
}
```

| Status | Meaning |
| --- | --- |
| `translated` | OCR'd, translated and reletted |
| `offline_fallback` | LLM text did not fit; the offline draft was used instead |
| `untranslated` | translator returned nothing new; the balloon was left untouched |
| `unfittable` | translation cannot fit at the minimum font size; nothing erased |
| `no_text` | detector found no text in this region |
| `failed` | OCR or render error; artwork untouched |

Page statuses are `ok`, `partial` (any region failed) or `no_text`. The command
exits `0` when every page is `ok`, `1` when any page is `partial`/`failed`, and
`2` on a configuration error. **Always eyeball the output**: OCR, detection and
inpainting are all approximate.

### Missed a balloon? Annotate it

Open the image, read the coordinates, and add a region the detectors did not
find. Give points *inside* the balloon outline:

```json
{"panel-1.jpeg": [{"polygon": [[903, 1276], [948, 1281], [998, 1298], [999, 1362], [986, 1465]]}]}
```

```sh
.venv/bin/manga-translate translate panel-1.jpeg --source en --target id \
  --regions regions.json --output ./translated-id
```

Keys are the input filename, or the relative path inside a folder input. A
`bbox: [left, top, right, bottom]` works instead of a polygon. Keep the shape
inset: it masks both the erase step and the fitted text, so a polygon that
spans artwork can erase art.

### Optional: LLM post-editing

The LLM receives the recognized source text and its offline draft — **not the
image** — and rewrites the draft into more natural target language. It never
replaces a translation that does not fit; the offline draft is used then.

```sh
.venv/bin/manga-translate translate ./pages --source ja --target id \
  --llm --provider netra --model deepseek/deepseek-v4-flash-0731 \
  --output ./translated-id
```

Credentials, in order of priority:

1. **A provider file.** Defaults to `~/.omp/agent/models.yml`:

   ```yaml
   providers:
     netra:
       baseUrl: https://your-endpoint/v1
       apiKey: sk-...
       models:
         - id: deepseek/deepseek-v4-flash-0731
   ```

   Point elsewhere with `--models-file ./models.yml`; pick an entry with
   `--provider` and `--model`.

2. **Environment variables**, which skip the file entirely and are convenient
   for CI:

   | Variable | Purpose |
   | --- | --- |
   | `MANGA_LLM_BASE_URL` | OpenAI-compatible base URL |
   | `MANGA_LLM_API_KEY` | key for any provider |
   | `MANGA_LLM_API_KEY_<PROVIDER>` | key for one provider, e.g. `MANGA_LLM_API_KEY_NETRA` |
   | `MANGA_LLM_MODEL` | model id |
   | `OPENAI_API_KEY` | last-resort fallback |

Remote endpoints must be `https`; plain `http` is accepted only for loopback.
`--llm` makes network requests and may cost money. Without it, no key is read,
no config file is opened, and no request is made.

### Manual relettering without translation

`manga-bubbles` finds balloons and gives each an ID you can translate by hand.

```sh
.venv/bin/manga-bubbles analyze panel-1.jpeg > detections.json
```

```json
[{"id": 1, "bbox": [576, 4, 856, 436], "text": "I WANT TO DO IT!", "words": [ ... ]}]
```

Write a JSON map of only the IDs you want changed, then apply it:

```json
{"1": "I WILL DO IT TODAY, EVEN IF IT TAKES ALL DAY!"}
```

```sh
.venv/bin/manga-bubbles replace panel-1.jpeg translations.json out.png \
  --font /usr/share/fonts/truetype/dejavu/DejaVuSans.ttf
```

Use the same `--lang` and `--psm` for `analyze` and `replace` so the IDs line
up. Balloons you do not list, and everything outside the balloon interiors,
stay pixel-identical. Text too long to fit raises an error rather than clipping.

## How detection works

1. **Bright-region geometry** finds enclosed balloons and light rectangular
   narration boxes by thresholding, filling, and filtering on shape.
2. **A bundled ONNX balloon segmenter** adds balloons the geometric pass cannot
   see — notably open balloons whose white interior flows into the page
   background. Its mask is inset by a few pixels before OCR, so lettering
   sitting outside the outline is never read as dialogue.
3. Overlapping proposals are deduplicated, so a balloon is never OCR'd twice.

The segmenter is `huyvux3005/manga109-segmentation-bubble` (Apache-2.0),
exported to a fixed 1600×1600 ONNX graph and run with `onnxruntime` only — no
Ultralytics install, no download at run time. See `manga_models/NOTICE` and
`manga_models/LICENSE`.

## Scope and limits

- Targets speech balloons and light narration boxes. Dark balloons, heavy
  screentone, and unusual shapes may be missed — annotate them with `--regions`.
- Erasure covers OCR word boxes, so stylized lettering or punctuation the OCR
  missed can need manual retouching.
- Right-to-left and vertical reading order is not reconstructed for you; the
  report lists regions top-to-bottom, left-to-right.
- Detection is not a general artwork segmenter, and no OCR result is guaranteed.
- Inspect before publishing anything.

## Development

```sh
uv venv .venv
uv pip install --python .venv/bin/python -e .
uv pip install --python .venv/bin/python "numpy>=1.26" "opencv-python-headless>=4.11" "Pillow>=10" "PyYAML>=6" "onnxruntime>=1.17,<2"

.venv/bin/python -m unittest discover -s tests -v
```

Tests need Tesseract with the `eng` pack, DejaVu fonts, and the bundled
`manga_models/bubble.onnx` (it ships in the wheel, so an editable install has
it). The suite is deterministic, runs offline, and draws its own pages with
Pillow — including the open balloon whose interior joins the page background,
which geometry alone cannot separate. No third-party manga page is redistributed
with this project.

`examples/` is gitignored: it is the place for your own scans.

## Releasing

`.github/workflows/publish.yml` publishes to PyPI when a `v*` tag is pushed,
using [PyPI trusted publishing](https://docs.pypi.org/trusted-publishers/) — no
API token is stored in the repository.

One-time setup on PyPI (**Pending publisher** → *Add a new pending publisher*):

| Field | Value |
| --- | --- |
| PyPI project name | `manga-translator` |
| Owner | `igun997` |
| Repository name | `manga-translator` |
| Workflow name | `publish.yml` |
| Environment name | `pypi` |

Then create the matching environment under **Settings → Environments** in the
repo. Release with:

```sh
git tag v0.3.0 && git push origin v0.3.0
```

Run the workflow manually with target `testpypi` to rehearse the upload
(configure a second pending publisher with environment `testpypi`).

## License

MIT — see [LICENSE](LICENSE). The bundled balloon model weights are Apache-2.0;
see [manga_models/NOTICE](manga_models/NOTICE).

### Research basis

- [OpenCV contour extraction](https://docs.opencv.org/4.13.0/df/d0d/tutorial_find_contours.html): contour-derived interior masks.
- [Pillow text measurement and drawing](https://pillow.readthedocs.io/en/stable/reference/ImageDraw.html): measured line layout and mask rendering.
- [huyvux3005/manga109-segmentation-bubble](https://huggingface.co/huyvux3005/manga109-segmentation-bubble) (Apache-2.0): bundled YOLO11n balloon segmenter, exported to ONNX.
- [Dubray and Laubrock, speech balloon segmentation](https://arxiv.org/abs/1902.08137): open balloons need learned segmentation rather than thresholding alone.
- [comic-text-detector](https://github.com/dmMaze/comic-text-detector): learned text masks as an alternative to OCR rectangles; its GPL-3.0 code is not included.
