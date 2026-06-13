# DataFlow-CV

> 🌊 **Where Vibe Coding meets CV data.** Convert, visualize & evaluate datasets — built with the flow of Claude Code.

<p align="center">
  <a href="https://pypi.org/project/dataflow-cv/"><img src="https://img.shields.io/pypi/v/dataflow-cv.svg" alt="PyPI"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.8+-blue.svg" alt="Python 3.8+"></a>
  <a href="https://github.com/zjykzj/DataFlow-CV/actions/workflows/python-publish.yml"><img src="https://github.com/zjykzj/DataFlow-CV/actions/workflows/python-publish.yml/badge.svg" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="License"></a>
  <br>
  <img src="https://img.shields.io/badge/Linux-Supported-fcc624?logo=linux" alt="Linux">
  <img src="https://img.shields.io/badge/Windows-Supported-00a2e8?logo=windows" alt="Windows">
  <img src="https://img.shields.io/badge/macOS-Supported-999999?logo=apple" alt="macOS">
  <img src="https://img.shields.io/badge/YOLO-.txt-00a86b?style=flat-square" alt="YOLO">
  <img src="https://img.shields.io/badge/LabelMe-.json-f39c12?style=flat-square" alt="LabelMe">
  <img src="https://img.shields.io/badge/COCO-.json-e74c3c?style=flat-square" alt="COCO">
</p>

A computer vision dataset processing library — convert, visualize, and evaluate annotations across YOLO, LabelMe, and COCO formats.

| | | |
|:---|:---|:---|
| 🔄 **Convert** | 6 directions: YOLO ↔ LabelMe ↔ COCO, plus model predictions | `dataflow-cv convert yolo2coco ...` |
| 🎨 **Visualize** | OpenCV rendering with color-coded classes, display & save modes | `dataflow-cv visualize yolo ...` |
| 📊 **Evaluate** | COCO mAP via pycocotools, single-threshold P/R/F1 per class | `dataflow-cv evaluate detection ...` |
| 💻 **CLI + API** | Click-based CLI with rich `--help`; Python API for pipelines | `from dataflow.convert import ...` |

---

## 📦 Installation

```bash
pip install dataflow-cv               # from PyPI
pip install pycocotools               # optional: COCO RLE + evaluation
```

Or from source:

```bash
git clone https://github.com/zjykzj/DataFlow-CV.git
cd DataFlow-CV && pip install .
```

---

## 🚀 Quick Start

### Command-line Interface

All required parameters (image directories, label directories, class files, output paths) are positional arguments for better usability. Use `--help` on any subcommand for detailed usage.

#### 🔄 Format Conversion

```bash
# YOLO → COCO
dataflow-cv convert yolo2coco images/ yolo_labels/ classes.txt output.json

# YOLO → COCO (with RLE encoding)
dataflow-cv convert yolo2coco images/ yolo_labels/ classes.txt output.json --do-rle

# YOLO → LabelMe
dataflow-cv convert yolo2labelme images/ yolo_labels/ classes.txt labelme_json/

# LabelMe → YOLO
dataflow-cv convert labelme2yolo labelme_json/ classes.txt yolo_labels/

# LabelMe → COCO
dataflow-cv convert labelme2coco labelme_json/ classes.txt output.json

# COCO → YOLO
dataflow-cv convert coco2yolo input.json yolo_labels/

# COCO → LabelMe
dataflow-cv convert coco2labelme input.json labelme_json/

# YOLO predictions → COCO (output: plain JSON list — prediction format)
dataflow-cv convert yolo2coco --prediction images/ yolo_preds/ classes.txt pred.json

# Options
dataflow-cv convert yolo2coco --verbose images/ labels/ classes.txt output.json
dataflow-cv convert yolo2coco --no-strict images/ labels/ classes.txt output.json
```

#### 🎨 Visualization

```bash
# Visualize YOLO annotations
dataflow-cv visualize yolo images/ yolo_labels/ classes.txt --save visualized/

# Visualize LabelMe annotations
dataflow-cv visualize labelme images/ labelme_json/ --save visualized/

# Visualize COCO annotations
dataflow-cv visualize coco images/ coco_annotations.json --save visualized/

# Verbose logging + headless mode
dataflow-cv visualize yolo --verbose --no-display images/ yolo_labels/ classes.txt --save visualized/
```

#### 📊 Evaluation

Evaluate object detection and instance segmentation model outputs using COCO-standard metrics. Two COCO-format JSON files are required:

| File | Role | Format | Source |
|------|------|--------|--------|
| **`anno.json`** | Ground Truth (GT) — reference annotations | Full COCO dict (`images`, `annotations`, `categories`) | `yolo2coco` (label mode) |
| **`pred.json`** | Detection (DT) — model predictions | Plain JSON list of annotation dicts (with `score`) | `yolo2coco --prediction`, Detectron2, MMDetection |

##### ① Preparing Evaluation Data

If your annotations and predictions are in YOLO format, convert them to COCO JSON first:

```bash
# Step 1: YOLO ground truth labels → COCO GT (anno.json)
#   Label format:   class_id cx cy w h               ← 5 tokens (detection)
#                   class_id x1 y1 ... xn yn          ← odd tokens (segmentation)
dataflow-cv convert yolo2coco images/ yolo_labels/ classes.txt anno.json

# Step 2: YOLO predictions → COCO DT (pred.json)
#   Prediction fmt: class_id cx cy w h confidence     ← 6 tokens (detection)
#                   class_id x1 y1 ... xn yn confidence ← even tokens (segmentation)
dataflow-cv convert yolo2coco --prediction images/ yolo_preds/ classes.txt pred.json
```

> ⚠️ **Important**: YOLO label files (GT) use **odd** token counts, while prediction files (DT) use **even** token counts with a trailing `confidence`. The `--prediction` flag is required for DT — it outputs a **plain JSON list** of annotation dicts (not a full COCO dict with `images`/`categories`). Mixed label/prediction files in the same directory are not supported.
>
> ℹ️ **Note**: The `--prediction` flag is **only available for `yolo2coco`**. `labelme2coco` does not support prediction conversion — LabelMe files (.json) have no label vs prediction format distinction, so there is no equivalent prediction source format to convert from.

##### ② Detection vs Segmentation — Format Requirements

| Field | Detection GT | Detection DT | Segmentation GT | Segmentation DT |
|-------|:-----------:|:-----------:|:---------------:|:---------------:|
| `bbox` | ✅ Required | ✅ Required | ✅ Required (for area) | ✅ Required (for area) |
| `score` | — | ✅ **Required** | — | ✅ **Required** |
| `segmentation` | ❌ Not required | ❌ Not required | ✅ **Required** | ✅ **Required** |
| `area` | ⚪ Recommended | ⚪ Recommended | ✅ **Required** | ✅ **Required** |
| `iscrowd` | ⚪ Optional | — | ⚪ Optional | — |

- **Object Detection** (`iouType='bbox'`): Bounding box overlap evaluation. Only `bbox` + `score` mandatory in DT.
- **Instance Segmentation** (`iouType='segm'`): Mask overlap evaluation. GT and DT must include `segmentation` (polygon or RLE), `area`, and `bbox`.

##### ③ CLI Commands

```bash
# Object detection evaluation (bbox IoU)
dataflow-cv evaluate detection anno.json pred.json

# Verbose per-class breakdown
dataflow-cv evaluate detection --verbose anno.json pred.json

# P/R/F1 only (skips mAP computation) — default: macro averaging
dataflow-cv evaluate detection --prf1 anno.json pred.json

# P/R/F1 with custom IoU threshold and micro averaging
dataflow-cv evaluate detection --prf1 --prf1-iou 0.75 --prf1-method micro anno.json pred.json

# Instance segmentation evaluation (mask IoU)
dataflow-cv evaluate segmentation anno.json pred.json

# Save results as JSON
dataflow-cv evaluate detection --output results.json anno.json pred.json
```

##### ④ End-to-End Workflow

```bash
# Complete pipeline: YOLO → COCO → Evaluation
dataflow-cv convert yolo2coco images/ yolo_labels/ classes.txt anno.json
dataflow-cv convert yolo2coco --prediction images/ yolo_preds/ classes.txt pred.json
dataflow-cv evaluate detection --verbose anno.json pred.json           # mAP
dataflow-cv evaluate detection --prf1 anno.json pred.json             # P/R/F1
```

### 🐍 Python API

```python
from dataflow.util.logging import LogConfig
from dataflow.convert import YoloAndCocoConverter
from dataflow.visualize import YOLOVisualizer
from dataflow.evaluate import DetectionEvaluator, compute_pr_f1

# ── Convert ──────────────────────────────────────────
# YOLO labels → COCO (label mode)
log_cfg = LogConfig(name="convert", verbose=True)
converter = YoloAndCocoConverter(source_to_target=True, log_config=log_cfg, strict_mode=True)
result = converter.convert(
    source_path="yolo_labels/", target_path="anno.json",
    class_file="classes.txt", image_dir="images/",
)

# YOLO predictions → COCO (prediction mode)
converter = YoloAndCocoConverter(source_to_target=True, prediction=True)
result = converter.convert(
    source_path="yolo_preds/", target_path="pred.json",
    class_file="classes.txt", image_dir="images/",
)

# ── Visualize ────────────────────────────────────────
visualizer = YOLOVisualizer(
    label_dir="yolo_labels/", image_dir="images/",
    class_file="classes.txt", is_show=True, is_save=True,
    output_dir="visualized/", log_config=log_cfg,
)
result = visualizer.visualize()

# ── Evaluate ─────────────────────────────────────────
evaluator = DetectionEvaluator(log_config=LogConfig(name="eval", verbose=True))
result = evaluator.evaluate("anno.json", "pred.json")
print(f"AP: {result.metrics.ap:.3f}, AP50: {result.metrics.ap50:.3f}")

# Quick P/R/F1 at IoU=0.5 (default: macro averaging, bbox IoU)
prf1 = compute_pr_f1("anno.json", "pred.json", iou_threshold=0.5)
print(f"Macro F1: {prf1.overall.f1_score:.3f}")

# Micro averaging P/R/F1 (samples weighted equally)
prf1 = compute_pr_f1("anno.json", "pred.json", method="micro")
print(f"Micro F1: {prf1.overall.f1_score:.3f}")

# Segmentation P/R/F1 (mask IoU)
prf1 = compute_pr_f1("anno_segm.json", "pred_segm.json", iou_type="segm")
print(f"Segm F1: {prf1.overall.f1_score:.3f}")
```

> 📂 See the `samples/` directory for complete examples: `samples/visualize/` (YOLO, LabelMe, COCO demos), `samples/convert/` (conversion examples).

---

## 📖 Documentation

| Resource | Description |
|----------|-------------|
| **[CLAUDE.md](CLAUDE.md)** | Architecture overview, development guide, and known gotchas |
| **[CHANGELOG.md](CHANGELOG.md)** | Version history and breaking changes |
| **[specs/evaluate/](specs/evaluate/)** | Evaluation metric contracts — IoU, matching, AP/mAP/AR |
| **[specs/formats/](specs/formats/)** | External format contracts — YOLO, LabelMe, COCO, conversion rules |
| **[specs/modules/](specs/modules/)** | Internal module architecture, interface contracts, dependency constraints |

### 💡 Key Concepts

- **Format-Native Coordinates**: YOLO uses normalized [0,1] center-based coordinates; LabelMe and COCO use absolute pixel top-left. There is no hidden internal normalization — check `DatasetAnnotations.format` to interpret coordinate semantics.
- **Strict Mode** (default): Validation errors raise exceptions immediately. Disable with `--no-strict` (CLI) or `strict_mode=False` (API) to skip invalid annotations and continue.
- **Verbose Logging**: `--verbose` enables per-module file logging via `LogManager` — console shows INFO-level progress, log files capture DEBUG details. All logging is owned by modules; the CLI uses `click.echo()` for terminal output.
- **Headless Support**: Use `--no-display` for servers/Docker — pair with `--save` to render visualization images without a GUI window.
- **Keyboard Shortcuts** (visualization): `q` / `ESC` to exit, `Enter` / `Space` to advance, any other key to continue.
- **Evaluation**: `--prf1` computes P/R/F1 only (single-threshold, per-class TP/FP/FN) — skips the full COCOeval mAP pipeline for speed. Supports macro/micro averaging and bbox/mask IoU. Run without `--prf1` for standard COCO mAP. For both metrics, run twice.
- **Prediction Files**: YOLO predictions use 6 tokens (detection) or even tokens (segmentation) vs 5/odd for labels. Use `--prediction` with `yolo2coco` — outputs a plain JSON list of annotation dicts compatible with pycocotools `loadRes()`.

---

## 🔧 Development

For detailed developer guidance including advanced test commands, debugging, and architecture overview, see [CLAUDE.md](CLAUDE.md).

### 🧪 Testing

**418 tests, 76% code coverage (3986 statements).**

```bash
pytest                                    # All tests
pytest --cov=dataflow --cov-report=term   # With coverage
pytest tests/convert/test_yolo_and_coco.py  # Single module
pytest tests/evaluate/test_evaluator.py     # Single module
```

<details>
<summary><b>📊 Coverage by module</b></summary>

| Module | Coverage | Highlights |
|--------|:--------:|------------|
| `dataflow/label/` | 68% | models (87%), coco_handler (75%), labelme_handler (70%), yolo_handler (58%) |
| `dataflow/convert/` | 87% | yolo_and_coco (90%), labelme_and_yolo (86%), coco_and_labelme (87%), rle (80%), base (83%), utils (92%) |
| `dataflow/visualize/` | 81% | yolo_vis (100%), labelme_vis (100%), coco_vis (97%), base (74%) |
| `dataflow/evaluate/` | 87% | evaluator (100%), metrics (93%), result (99%), base (91%), utils (68%) |
| `dataflow/cli/` | 59% | main (96%), convert cmd (48%), evaluate cmd (24%), visualize cmd (84%), utils (86%) |
| `dataflow/util/` | 93% | logging (98%) |

</details>

### 🎨 Code Quality

```bash
pip install -e .[dev]        # Install dev dependencies
black dataflow tests samples  # Format
isort dataflow tests samples  # Sort imports
mypy dataflow                 # Type check
flake8 dataflow tests samples # Lint
```

### 🔗 Pre-commit Hooks (Optional)

```bash
pip install pre-commit
pre-commit install            # Install git hooks (run once)

# After this, every `git commit` auto-runs:
#   black → isort → flake8 → whitespace checks

pre-commit run --all-files    # Manual run against all files
```

### 📁 Project Structure

```
dataflow/
├── label/           # Annotation handlers + data models
├── convert/         # Format converters, RLE utility, log templates
├── visualize/       # OpenCV-based rendering, log templates
├── evaluate/        # pycocotools-based metrics, log templates
├── util/            # Unified logging (LogManager + format helpers)
└── cli/             # CLI entry point, commands, validation
tests/               # Unit & integration tests (418 tests, conftest fixtures)
samples/             # Python API usage examples
assets/              # Test data (det/seg by format)
specs/               # Canonical specifications (evaluate/ + formats/ + modules/)
```

---

## 🤝 Contributing

Contributions are welcome! Please review [CLAUDE.md](CLAUDE.md) for architecture and development patterns before contributing.

1. 🍴 Fork the repository
2. 🌿 Create a feature branch
3. ✏️ Make your changes
4. 🧪 Add or update tests as needed
5. ✅ Ensure code passes formatting and linting checks
6. 📬 Submit a pull request

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- Thanks to the creators of YOLO, LabelMe, and COCO formats for establishing these annotation standards
- Built with [OpenCV](https://opencv.org/), [NumPy](https://numpy.org/), [Click](https://click.palletsprojects.com/), and [pycocotools](https://github.com/cocodataset/cocoapi)
- Inspired by the need for seamless format conversion in multi-tool CV pipelines
