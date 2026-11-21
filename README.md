# DataFlow-CV

> 🌊 **Everything your model doesn't do.** Analyse, convert, visualize, evaluate — a single CLI for all CV data.

<p align="center">
  <a href="https://pypi.org/project/dataflow-cv/"><img src="https://img.shields.io/pypi/v/dataflow-cv.svg" alt="PyPI"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.8+-blue.svg" alt="Python 3.8+"></a>
  <a href="https://github.com/zjykzj/DataFlow-CV/actions/workflows/python-publish.yml"><img src="https://github.com/zjykzj/DataFlow-CV/actions/workflows/python-publish.yml/badge.svg" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="License"></a>
  <a href="https://deepwiki.com/zjykzj/DataFlow-CV"><img src="https://img.shields.io/badge/Ask-DeepWiki-A78BFA" alt="Ask DeepWiki"></a>
  <br>
  <img src="https://img.shields.io/badge/Linux-Supported-fcc624?logo=linux" alt="Linux">
  <img src="https://img.shields.io/badge/Windows-Supported-00a2e8?logo=windows" alt="Windows">
  <img src="https://img.shields.io/badge/macOS-Supported-999999?logo=apple" alt="macOS">
  <img src="https://img.shields.io/badge/YOLO-.txt-00a86b?style=flat-square" alt="YOLO">
  <img src="https://img.shields.io/badge/LabelMe-.json-f39c12?style=flat-square" alt="LabelMe">
  <img src="https://img.shields.io/badge/COCO-.json-e74c3c?style=flat-square" alt="COCO">
</p>

A computer vision dataset processing library — analyse, convert, visualize, and evaluate detection, segmentation, and pose (keypoint) annotations across YOLO, LabelMe, and COCO formats. Works with ground-truth labels and model predictions alike, from the CLI or Python API.

| | | |
|:---|:---|:---|
| 🔍 **Analyse** | Stats, train/val split, category filter, N-way partition & file sampling — format auto-detection | `dataflow-cv analyse stats ...` |
| 🔄 **Convert** | 6 directions: YOLO ↔ LabelMe ↔ COCO, plus model predictions | `dataflow-cv convert yolo2coco ...` |
| 🎨 **Visualize** | OpenCV rendering with color-coded classes, display & save modes | `dataflow-cv visualize yolo ...` |
| 📊 **Evaluate** | COCO mAP via pycocotools, single-threshold P/R/F1 per class | `dataflow-cv evaluate detection ...` |
| 📋 **Task Types** | Detection, segmentation & pose (keypoints) supported by all four modules across all three formats — det/seg auto-detected, pose explicit (`--pose`) | `dataflow-cv convert yolo2coco --pose ...` |
| 💻 **CLI + API** | Click-based CLI with rich `--help`; Python API for pipelines | `from dataflow.convert import ...` |
| 🤖 **AI Skills** | Claude Code skill (`/dataflow:dataflow-cv`) for AI assistants — CLI/API reference & known gotchas | `claude plugin install dataflow@claude-skills` |

---

## 📦 Installation

```bash
pip install dataflow-cv               # from PyPI
pip install dataflow-cv[coco]         # optional: COCO RLE + evaluation
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

#### 📂 Data Layout & classes.txt

Input data must follow each format's native layout — dataflow-cv detects the format automatically from what you point it at:

| Format | What you pass | Expected layout |
|--------|---------------|-----------------|
| **YOLO** | `labels/` directory (+ `images/` directory) | One `.txt` per image: `labels/000001.txt` next to `images/000001.jpg` — the label file **must share the image's file name**. Detection lines are `class_id cx cy w h`; segmentation lines are `class_id x1 y1 x2 y2 …`; pose lines are `class_id cx cy w h px1 py1 pv1 …` (5+3k tokens, k fixed per dataset — all coordinates normalized [0,1], `pv` ∈ {0,1,2} visibility). |
| **LabelMe** | a directory of `.json` files | One `.json` per image (`shapes` with `label` + `points`). Pose convention: one `point` shape per keypoint (label = keypoint name) + a `rectangle` (class + bbox) sharing a `group_id`. |
| **COCO** | a single `.json` file | One file containing `images`, `annotations`, `categories` arrays. Pose datasets carry `annotations[].keypoints` + `num_keypoints` and `categories[].keypoints`/`skeleton`. |

`classes.txt` is the **class name mapping** used by YOLO-related commands. It is just one class name per line — the line number is the class id (0-indexed):

```
person
bicycle
car
```

- **Required** for: `yolo2coco`, `yolo2labelme`, `labelme2yolo`, `labelme2coco`, `visualize yolo`
- **Auto-generated** for `coco2yolo` / `coco2labelme` (written next to the output — you can pass your own with `-c` to control the order)
- Also used by `analyse stats -c` for display names and strict category validation

> 💡 Converting a COCO dataset first? Run `coco2yolo` once — it writes a ready-to-use `classes.txt`.

> 🦴 **Pose datasets** use a **keypoint-definition YAML** for names/skeleton metadata (`kpts.yaml`):
>
> ```yaml
> keypoints:           # or per-class "classes:" (X-AnyLabeling style)
>   - nose
>   - left_eye
>   # ...
> skeleton:            # optional: edge pairs as indices into the name list
>   - [0, 1]
> sigmas:              # optional: per-keypoint OKS sigmas (evaluate keypoints)
>   - 0.026
> ```
>
> Pass it with `--kpts-config` (convert / visualize / analyse) or `--save-kpts-config` to export one from a COCO dataset.

#### 📋 Task Types: Detection / Segmentation / Pose

All four modules work on three annotation tasks, and every format can carry all three — what differs is how each task is written on disk and how it is enabled:

| Task | YOLO (per-image `.txt`) | LabelMe (per-image `.json`) | COCO (single `.json`) | Enablement |
|------|-------------------------|-----------------------------|-----------------------|------------|
| 🔲 **Detection** | `class_id cx cy w h` | `rectangle` shape | `bbox` field | Automatic — no flag |
| ✂️ **Segmentation** | `class_id x1 y1 … xn yn` | `polygon` shape | `segmentation` field (polygon or RLE) | Automatic — no flag |
| 🦴 **Pose (keypoints)** | `class_id cx cy w h px1 py1 pv1 …` (5+3k tokens) | `point` shapes + one `rectangle` sharing a `group_id` | `keypoints` field + `categories[].keypoints`/`skeleton` | Explicit — see below |

**Detection and segmentation are auto-detected** — from row token count (YOLO), shape type (LabelMe), or field presence (COCO). There is no task flag for them.

**Pose is explicit.** A YOLO pose row cannot be distinguished from a segmentation row by token count alone (an 11-token row is both a 2-keypoint pose label and a segmentation label), so pose processing is opt-in:

| Module | How pose is enabled | Notes |
|--------|---------------------|-------|
| 🔍 **Analyse** | `--kpts-config kpts.yaml` | Adds per-keypoint labeling-rate statistics; for YOLO sources this flag also enables pose-row parsing |
| 🔄 **Convert** | `--pose` (all six directions; `yolo2coco --prediction --pose` for model output) | Without it, keypoints are dropped with one warning per image (YOLO pose rows effectively require it); in pose mode, objects without keypoints are skipped |
| 🎨 **Visualize** | `--pose` (YOLO) · automatic (LabelMe / COCO) | `--kpts-config` adds skeleton lines (YOLO / LabelMe; COCO reads them from its categories). v=2 filled green, v=1 hollow orange |
| 📊 **Evaluate** | `evaluate keypoints` | OKS-based; `--kpts-config` optionally supplies per-keypoint sigmas |

The keypoint-definition YAML (`kpts.yaml`, see above) carries keypoint names, optional `skeleton` edges, and optional `sigmas`; `coco2yolo` / `coco2labelme --save-kpts-config` exports one from a COCO dataset.

#### 🔍 Dataset Analysis

```bash
# Dataset statistics (auto-detects YOLO / LabelMe / COCO)
dataflow-cv analyse stats yolo_labels/ --image-dir images/ --class-file classes.txt
dataflow-cv analyse stats labelme_json/
dataflow-cv analyse stats coco_annotations.json

# Multiple paths merge into one report (same format); -R recurses subdirectories
dataflow-cv analyse stats yolo_train/ yolo_val/ --recursive

# Train / test split (YOLO / LabelMe only — labels / images / both modes)
#   → outputs/train/ + outputs/val/  (each with labels/ + images/ in both mode)
dataflow-cv analyse split -l yolo_labels/ outputs/ --ratio 0.8 --seed 42 -c classes.txt
dataflow-cv analyse split -i images/ outputs/ --ratio 0.8
dataflow-cv analyse split -l yolo_labels/ -i images/ outputs/ --ratio 0.8

# Category filter (keep a subset of categories, remap IDs per new classes.txt)
#   → filtered/ receives the filtered labels + the new classes.txt
dataflow-cv analyse filter yolo_labels/ classes.txt classes_new.txt filtered/
dataflow-cv analyse filter coco_annotations.json classes.txt classes_new.txt filtered/

# N-way partition — YOLO / LabelMe only (labels drive, images follow by stem)
#   → parts/part_1/ … parts/part_N/  (each with labels/ + images/ in both mode)
dataflow-cv analyse partition -n 4 --label-dir yolo_labels/ --image-dir images/ parts/
dataflow-cv analyse partition -n 4 --image-dir images/ --shuffle parts/

# File sampling — collect N files (random or sequential, labels / images / both modes)
#   → flat files in output/  (or output/labels/ + output/images/ in both mode)
dataflow-cv analyse sample -l yolo_labels/ output/ -n 10
dataflow-cv analyse sample -i images/ output/ -n 10 --no-shuffle
dataflow-cv analyse sample -l yolo_labels/ -i images/ output/ -n 5 --seed 42

# Sort by count descending (default: class ID ascending)
dataflow-cv analyse stats --sort-by count --descending yolo_labels/

# Pose: keypoint statistics (per-keypoint labeling rates; required for
# YOLO pose sources — it also enables pose-row parsing)
dataflow-cv analyse stats --kpts-config kpts.yaml yolo_pose_labels/

# Verbose logging
dataflow-cv analyse stats --verbose yolo_labels/ --class-file classes.txt
```

#### 🔄 Format Conversion

```bash
# YOLO → COCO                      → output.json: single COCO file
dataflow-cv convert yolo2coco images/ yolo_labels/ classes.txt output.json

# YOLO → COCO (with RLE encoding)
dataflow-cv convert yolo2coco images/ yolo_labels/ classes.txt output.json --do-rle

# YOLO → LabelMe                   → labelme_json/: one .json per image
dataflow-cv convert yolo2labelme images/ yolo_labels/ classes.txt labelme_json/

# LabelMe → YOLO                   → yolo_labels/: one .txt per image + classes.txt
dataflow-cv convert labelme2yolo labelme_json/ classes.txt yolo_labels/

# LabelMe → COCO                   → output.json: single COCO file
dataflow-cv convert labelme2coco labelme_json/ classes.txt output.json

# COCO → YOLO                      → yolo_labels/labels/*.txt + classes.txt
dataflow-cv convert coco2yolo input.json yolo_labels/

# COCO → LabelMe                   → labelme_json/: one .json per image
dataflow-cv convert coco2labelme input.json labelme_json/

# iscrowd=1 (crowd) annotations are skipped by default on both directions;
# opt in to writing them as ordinary rows/shapes:
dataflow-cv convert coco2yolo --save-crowd input.json yolo_labels/

# YOLO predictions → COCO (output: plain JSON list — prediction format)
dataflow-cv convert yolo2coco --prediction images/ yolo_preds/ classes.txt pred.json

# Pose: add --pose to any of the six directions
dataflow-cv convert coco2yolo --pose coco_pose.json yolo_pose_labels/
dataflow-cv convert yolo2coco --pose --kpts-config kpts.yaml images/ yolo_pose_labels/ classes.txt pose.json
dataflow-cv convert coco2yolo --pose --save-kpts-config kpts.yaml coco_pose.json yolo_pose_labels/   # export the definition YAML

# Pose predictions (yolo2coco --prediction --pose → keypoints + score)
dataflow-cv convert yolo2coco --prediction --pose images/ yolo_pose_preds/ classes.txt pred.json

# Options
dataflow-cv convert yolo2coco --verbose images/ labels/ classes.txt output.json
dataflow-cv convert yolo2coco --no-strict images/ labels/ classes.txt output.json
```

> 📂 **COCO→YOLO / COCO→LabelMe output naming**: each `.txt` / `.json` is named by the
> image file stem — leading zeros preserved (`000001.jpg` → `000001.txt`). The
> converter generates `labels/` (or the `.json` files) + `classes.txt`; the
> `images/` directory is created but left **empty** — images are never copied,
> place your image files there yourself.

#### 🎨 Visualization

```bash
# Visualize YOLO annotations
dataflow-cv visualize yolo images/ yolo_labels/ classes.txt --save visualized/

# Visualize LabelMe annotations
dataflow-cv visualize labelme images/ labelme_json/ --save visualized/

# Visualize COCO annotations
dataflow-cv visualize coco images/ coco_annotations.json --save visualized/

# Pose: keypoints + skeleton rendering (v=2 filled green, v=1 hollow orange)
dataflow-cv visualize yolo --pose --kpts-config kpts.yaml images/ yolo_pose_labels/ classes.txt --save visualized/
dataflow-cv visualize coco images/ coco_pose.json --save visualized/   # skeleton from COCO categories

# Verbose logging + headless mode
dataflow-cv visualize yolo --verbose --no-display images/ yolo_labels/ classes.txt --save visualized/
```

<p align="center">
  <img src="assets/showcase/seg_demo_1.jpg" width="45%" alt="Segmentation visualization demo 1">
  <img src="assets/showcase/seg_demo_2.jpg" width="45%" alt="Segmentation visualization demo 2">
</p>

#### 📊 Evaluation

Evaluate object detection, instance segmentation, and pose estimation models with COCO-standard metrics. Two COCO-format JSON files are required:

| File | Role | Format | How to create |
|------|------|--------|---------------|
| **`anno.json`** | Ground Truth (GT) | Full COCO dict (`images`, `annotations`, `categories`) | `yolo2coco` (label mode) |
| **`pred.json`** | Detection (DT) | Plain JSON list (with `score`) | `yolo2coco --prediction` |

##### ① Prepare Data

```bash
# GT: YOLO labels → COCO
dataflow-cv convert yolo2coco images/ yolo_labels/ classes.txt anno.json

# DT: YOLO predictions → COCO (add --prediction for model output)
dataflow-cv convert yolo2coco --prediction images/ yolo_preds/ classes.txt pred.json
```

> ⚠️ `--prediction` is required for YOLO prediction files — they have an extra `confidence` token per line. The flag outputs a **plain JSON list** (not a full COCO dict), which is the standard DT format for `loadRes()`. Only `yolo2coco` supports `--prediction`; `labelme2coco` does not need it (LabelMe has no label vs prediction distinction).

##### ② Run Evaluation

```bash
# Object detection (bbox IoU)
dataflow-cv evaluate detection anno.json pred.json
dataflow-cv evaluate detection --verbose anno.json pred.json           # per-class breakdown
dataflow-cv evaluate detection --prf1 anno.json pred.json              # P/R/F1 only (skip mAP)
dataflow-cv evaluate detection --prf1 --prf1-iou 0.75 --prf1-method micro anno.json pred.json

# Instance segmentation (mask IoU)
dataflow-cv evaluate segmentation anno.json pred.json
dataflow-cv evaluate segmentation --verbose anno.json pred.json

# Pose estimation (OKS) — sigmas from the definition YAML, else COCO
# 17-keypoint defaults (uniform 0.05 + warning for other k)
dataflow-cv evaluate keypoints --kpts-config kpts.yaml anno_pose.json pred_pose.json
dataflow-cv evaluate keypoints --prf1 anno_pose.json pred_pose.json   # OKS-based P/R/F1

# Save results as JSON
dataflow-cv evaluate detection --output results.json anno.json pred.json

# Save P/R/F1 results as JSON (thresholds, method, overall + per-class values)
dataflow-cv evaluate detection --prf1 --output prf1_results.json anno.json pred.json

# Custom log directory
dataflow-cv evaluate detection --verbose --log-dir logs/eval/ anno.json pred.json
```

##### ③ Detection vs Segmentation vs Pose

Three evaluation modes, distinguished by how overlap is measured:

- **Object Detection** — bounding box IoU. GT and DT require `bbox`; DT additionally requires `score`.
- **Instance Segmentation** — mask IoU. GT and DT require `bbox`, `segmentation` (polygon or RLE), and `area`; DT additionally requires `score`.
- **Pose Estimation** — OKS (Object Keypoint Similarity). GT and DT require `keypoints` (flat `[x, y, v, ...]` triplets) and `area`; DT additionally requires `score`. Per-keypoint sigmas resolve via `--kpts-config`, the COCO 17-keypoint official values, or a uniform fallback.

`yolo2coco` (label mode) and `yolo2coco --prediction` (prediction mode) automatically populate all required fields for all three modes — no manual editing needed.

#### 🔄 End-to-End Workflow

The full pipeline from raw YOLO data to evaluation — the output of each step feeds the next:

```bash
# 0) Your data:  data/images/*.jpg  +  data/labels/*.txt (same file names)
#                +  classes.txt (one class name per line)

# 1) Inspect the dataset
dataflow-cv analyse stats data/labels/ --image-dir data/images/ --class-file classes.txt

# 2) Split into train / val          → data/split/train/ + data/split/val/
dataflow-cv analyse split -l data/labels/ -i data/images/ -c classes.txt \
    --ratio 0.8 --seed 42 data/split/

# 3) Val labels → COCO ground truth  (GT and DT must cover the SAME images —
#    evaluate your model on the val set)
dataflow-cv convert yolo2coco data/split/val/images/ data/split/val/labels/ \
    classes.txt gt.json

# 4) Model predictions on the val set → COCO detections
#    (prediction files carry confidence — 6 tokens for detection)
dataflow-cv convert yolo2coco --prediction data/split/val/images/ preds/ \
    classes.txt pred.json

# 5) Evaluate (mAP, or quick P/R/F1)
dataflow-cv evaluate detection gt.json pred.json
dataflow-cv evaluate detection --prf1 --output prf1.json gt.json pred.json

# 6) Spot-check the val set visually
dataflow-cv visualize yolo --no-display data/split/val/images/ data/split/val/labels/ \
    classes.txt --save viz/
```

### 🐍 Python API

```python
from dataflow.util.logging import LogConfig
from dataflow.analyse import (
    StatsAnalyser,
    SplitAnalyser,
    FilterAnalyser,
    PartitionAnalyser,
    SampleAnalyser,
)
from dataflow.convert import YoloAndCocoConverter
from dataflow.visualize import YOLOVisualizer
from dataflow.evaluate import DetectionEvaluator, compute_pr_f1

# ── Analyse ─────────────────────────────────────────
log_cfg = LogConfig(name="analyse", verbose=True)

# Dataset statistics
analyser = StatsAnalyser(log_config=log_cfg)
result = analyser.analyse("yolo_labels/", class_file="classes.txt")
print(f"{result.data.total_files} images, {result.data.total_annotations} objects")

# Train/test split (YOLO / LabelMe)
splitter = SplitAnalyser(log_config=log_cfg)
result = splitter.analyse(
    output_dir="output/",
    ratio=0.8,
    seed=42,
    label_dir="yolo_labels/",
    class_file="classes.txt",
)
print(f"Train: {result.data.train_count}, Val: {result.data.val_count}")

# Split with images (both mode — labels drive, images follow by stem)
result = splitter.analyse(
    output_dir="output/",
    ratio=0.8,
    seed=42,
    label_dir="yolo_labels/",
    image_dir="images/",
    class_file="classes.txt",
)

# Category filter (keep / remap categories per new classes.txt)
filterer = FilterAnalyser(log_config=log_cfg)
result = filterer.analyse(
    "yolo_labels/",
    original_class_file="classes.txt",
    new_class_file="classes_new.txt",
    output_dir="filtered/",
)

# N-way partition (YOLO / LabelMe labels; images follow by stem)
partitioner = PartitionAnalyser(log_config=log_cfg)
result = partitioner.analyse(
    output_dir="parts/",
    num=4,
    label_dir="yolo_labels/",
    image_dir="images/",
)

# File sampling (labels, images, or both — random or sequential)
sampler = SampleAnalyser(log_config=log_cfg)
result = sampler.analyse(
    output_dir="sampled/",
    count=10,
    label_dir="yolo_labels/",
    shuffle=True,
    seed=42,
)

# ── Convert ──────────────────────────────────────────
# YOLO labels → COCO (label mode)
log_cfg = LogConfig(name="convert", verbose=True)
converter = YoloAndCocoConverter(source_to_target=True, log_config=log_cfg, strict_mode=True)
result = converter.convert(
    source_path="yolo_labels/",
    target_path="anno.json",
    class_file="classes.txt",
    image_dir="images/",
)

# YOLO predictions → COCO (prediction mode)
converter = YoloAndCocoConverter(source_to_target=True, prediction=True)
result = converter.convert(
    source_path="yolo_preds/",
    target_path="pred.json",
    class_file="classes.txt",
    image_dir="images/",
)

# ── Visualize ────────────────────────────────────────
visualizer = YOLOVisualizer(
    label_dir="yolo_labels/",
    image_dir="images/",
    class_file="classes.txt",
    is_show=True,
    is_save=True,
    output_dir="visualized/",
    log_config=log_cfg,
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

# Pose: OKS evaluation + OKS-based P/R/F1 (threshold = OKS threshold)
from dataflow.evaluate import KeypointEvaluator
kpt_eval = KeypointEvaluator(log_config=LogConfig(name="eval", verbose=True))
result = kpt_eval.evaluate("anno_pose.json", "pred_pose.json")
print(f"AP: {result.metrics.ap:.3f}")
prf1 = compute_pr_f1("anno_pose.json", "pred_pose.json", iou_type="keypoints")
print(f"OKS F1: {prf1.overall.f1_score:.3f}")
```

> 📂 See the `samples/` directory for complete examples: `samples/analyse/` (statistics & split), `samples/convert/` (6 conversion directions), `samples/visualize/` (YOLO, LabelMe, COCO), `samples/evaluate/` (detection & segmentation), `samples/cli/` (CLI workflows).

### 🤖 AI Assistant Skills

We maintain a dedicated Claude Code skill for dataflow-cv — `/dataflow:dataflow-cv` — so LLMs and AI agents can operate its CLI and Python API correctly in any project. The skill covers:

- **CLI command tree and task mapping**: 17 subcommands across analyse / convert / visualize / evaluate
- **Python API reference**: handlers (incl. pose `Keypoints`), analysers, converters, visualizers, evaluators
- **Canonical example commands** and **known gotchas** (coordinate semantics, prediction formats, RLE encoding, …)

Install once per machine:

```bash
claude plugin install dataflow@claude-skills
```

Once installed, AI assistants load the skill on demand when working on dataset conversion, evaluation, or similar tasks — it can also be invoked explicitly via `/dataflow:dataflow-cv`. The library works normally without it.

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
- **Keyboard Shortcuts** (visualization): `←` / `↑` for previous, `→` / `↓` / `Enter` / `Space` for next, `s` to save snapshot, `h` to show hints, `q` / `ESC` to exit. First/last-image boundaries are no-ops. Title bar shows `[N/T]` position.
- **Evaluation**: `--prf1` computes P/R/F1 only (single-threshold, per-class TP/FP/FN) — skips the full COCOeval mAP pipeline for speed. Supports macro/micro averaging and bbox/mask IoU. Run without `--prf1` for standard COCO mAP. For both metrics, run twice.
- **Prediction Files**: YOLO predictions use 6 tokens (detection) or even tokens (segmentation) vs 5/odd for labels. Use `--prediction` with `yolo2coco` — outputs a plain JSON list of annotation dicts compatible with pycocotools `loadRes()`.
- **Pose (Keypoints)**: Pose rows are explicit (`--pose`) — token counts can collide with segmentation rows, so auto-detection is unsafe. Keypoint visibility is `v ∈ {0,1,2}`: 0 = not labeled (stored at 0,0), 1 = labeled but occluded, 2 = labeled and visible. COCO `num_keypoints` is recomputed on write; the keypoint-definition YAML carries names/skeleton/sigmas.

---

## 🔧 Development

For detailed developer guidance including advanced test commands, debugging, and architecture overview, see [CLAUDE.md](CLAUDE.md). Optional Claude Code skills for common tasks (`/maestro:spec`, `/maestro:commit`, `/maestro:release`, `/maestro:claude`) are available via the [maestro plugin](https://github.com/zjykzj/claude-skills) — the project develops normally without them.

### 🧪 Testing

**755 tests, 81% code coverage (6784 statements).**

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
| `dataflow/label/` | 73% | models (88%), utils (86%), base (84%), coco_handler (75%), labelme_handler (73%), yolo_handler (64%) |
| `dataflow/analyse/` | 85% | base (99%), log_templates (94%), sample (88%), split (88%), stats (86%), utils (84%), partition (79%), filter (76%) |
| `dataflow/convert/` | 86% | base (88%), labelme_and_yolo (87%), yolo_and_coco (87%), coco_and_labelme (86%), utils (86%), log_templates (85%), rle (80%) |
| `dataflow/visualize/` | 82% | yolo_vis (100%), coco_vis (91%), labelme_vis (90%), base (79%) |
| `dataflow/evaluate/` | 91% | log_templates (100%), result (100%), evaluator (98%), base (92%), utils (89%), metrics (87%) |
| `dataflow/cli/` | 72% | main (96%), utils (88%), visualize cmd (87%), analyse cmd (71%), evaluate cmd (66%), convert cmd (64%) |
| `dataflow/util/` | 100% | logging (100%) |

</details>

### 🎨 Code Quality

```bash
pip install -e .[dev]                       # Install dev dependencies
ruff check dataflow tests samples           # Lint
ruff format --check dataflow tests samples  # Format check
mypy dataflow                               # Type check
```

### 🔗 Pre-commit Hooks (Optional)

```bash
pip install pre-commit
pre-commit install            # Install git hooks (run once)

# After this, every `git commit` auto-runs:
#   ruff (lint, auto-fix) → ruff format → whitespace checks

pre-commit run --all-files    # Manual run against all files
```

### 📁 Project Structure

```
dataflow/
├── label/           # Annotation handlers + data models
├── analyse/         # Dataset stats, train/val split, category filter, N-way partition, file sampling
├── convert/         # Format converters, RLE utility, log templates
├── visualize/       # OpenCV-based rendering, log templates
├── evaluate/        # pycocotools-based metrics, log templates
├── util/            # Unified logging (LogManager + format helpers)
└── cli/             # CLI entry point, commands, validation
tests/               # Unit & integration tests (755 tests, conftest fixtures)
samples/             # Python API usage examples (analyse, convert, visualize, evaluate, cli)
assets/              # Test data (det/seg by format)
specs/               # Canonical specifications (evaluate/ + formats/ + modules/)
```

---

## 🤝 Contributing

Contributions are welcome! Please review [CLAUDE.md](CLAUDE.md) for architecture and development patterns before contributing.

1. 🍴 Fork the repository
2. 🌿 Create a feature branch
3. ✏️ Make your changes — **spec-first**: if a change affects a contract in [specs/](specs/), update the spec before the code (SDD, see the `/maestro:spec` skill)
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
