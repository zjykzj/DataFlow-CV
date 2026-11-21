# CLI Reference

> dataflow-cv v3.6 — the complete command inventory. The [README](../../README.md)
> shows the most common commands; this page documents every subcommand and option.

## Command Tree

```
dataflow-cv
├── analyse  stats | split | filter | partition | sample
├── convert  yolo2coco | yolo2labelme | labelme2yolo | labelme2coco | coco2yolo | coco2labelme
├── visualize  yolo | labelme | coco
└── evaluate  detection | segmentation | keypoints
```

All required parameters (image directories, label directories, class files,
output paths) are **positional arguments**. Use `--help` on any subcommand for
the authoritative option list.

## Common Options

| Option | Where | Meaning |
|--------|-------|---------|
| `--verbose` | all commands | Enable per-module file logging (DEBUG-level log file under `--log-dir`, default `./logs/`) |
| `--log-dir PATH` | all commands | Log output directory |
| `--no-strict` | convert / visualize | Skip invalid annotations with a warning instead of aborting (analyse is always non-strict) |

## 🔍 Analyse

Format auto-detection: point at your labels — a single `.json` → COCO; a
directory of only `.txt` → YOLO; a directory of `.json` (first checked for
`shapes` → LabelMe, `images` → COCO). Mixed extensions or an empty directory
raise an error.

### `analyse stats`

```bash
dataflow-cv analyse stats yolo_labels/ --image-dir images/ --class-file classes.txt
dataflow-cv analyse stats labelme_json/
dataflow-cv analyse stats coco_annotations.json

# Multiple paths merge into one report (same format); -R recurses subdirectories
dataflow-cv analyse stats yolo_train/ yolo_val/ --recursive

# Sort by count descending (default: class ID ascending)
dataflow-cv analyse stats --sort-by count --descending yolo_labels/

# Pose: keypoint statistics (per-keypoint labeling rates; required for
# YOLO pose sources — it also enables pose-row parsing)
dataflow-cv analyse stats --kpts-config kpts.yaml yolo_pose_labels/

# Verbose logging
dataflow-cv analyse stats --verbose yolo_labels/ --class-file classes.txt
```

### `analyse split` (YOLO / LabelMe only)

```bash
# Labels / images / both modes → outputs/train/ + outputs/val/
dataflow-cv analyse split -l yolo_labels/ outputs/ --ratio 0.8 --seed 42 -c classes.txt
dataflow-cv analyse split -i images/ outputs/ --ratio 0.8
dataflow-cv analyse split -l yolo_labels/ -i images/ outputs/ --ratio 0.8
```

Options: `-l/--label-dir`, `-i/--image-dir`, `-c/--class-file`, `--ratio`
(default 0.8), `--seed` (default 42), `--move` (relocate instead of copy).

### `analyse filter` (all formats)

```bash
# Keep a subset of categories, remap IDs per the new classes.txt
#   → filtered/ receives the filtered labels + the new classes.txt
dataflow-cv analyse filter yolo_labels/ classes.txt classes_new.txt filtered/
dataflow-cv analyse filter coco_annotations.json classes.txt classes_new.txt filtered/
```

### `analyse partition` (YOLO / LabelMe only)

```bash
# N-way partition — labels drive, images follow by stem
#   → parts/part_1/ … parts/part_N/  (each with labels/ + images/ in both mode)
dataflow-cv analyse partition -n 4 --label-dir yolo_labels/ --image-dir images/ parts/
dataflow-cv analyse partition -n 4 --image-dir images/ --shuffle parts/
```

Options: `-n/--num` (≥ 2), `--label-dir`, `--image-dir`, `--shuffle`,
`--seed`, `--class-file`, `--move`.

### `analyse sample` (YOLO / LabelMe only)

```bash
# Collect N files — flat files in output/  (or output/labels/ + output/images/ in both mode)
dataflow-cv analyse sample -l yolo_labels/ output/ -n 10
dataflow-cv analyse sample -i images/ output/ -n 10 --no-shuffle
dataflow-cv analyse sample -l yolo_labels/ -i images/ output/ -n 5 --seed 42
```

Options: `-n/--num`, `--label-dir`, `--image-dir`, `--shuffle` (default on),
`--seed`, `--class-file`, `--move`.

## 🔄 Convert

Six directions — the output of each is one COCO file or one file per image:

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
```

### Options

| Option | Directions | Meaning |
|--------|-----------|---------|
| `--do-rle` | `yolo2coco` | RLE-encode segmentation output |
| `--prediction` | `yolo2coco` | Read YOLO prediction files (extra confidence token); output is a plain JSON list (Variant B) |
| `--pose` | all six | Pose conversion — see below |
| `--kpts-config PATH` | all six | Keypoint-definition YAML (names + skeleton) |
| `--save-kpts-config PATH` | `coco2yolo` / `coco2labelme` | Export a definition YAML from COCO categories |
| `--save-crowd` | `coco2yolo` / `coco2labelme` | Write `iscrowd=1` annotations as ordinary rows/shapes |

### Pose Conversion

Pose processing is **explicit** — a YOLO pose row cannot be told apart from a
segmentation row by token count alone (see the [README](../../README.md) Task
Types section).

```bash
dataflow-cv convert coco2yolo --pose coco_pose.json yolo_pose_labels/
dataflow-cv convert yolo2coco --pose --kpts-config kpts.yaml images/ yolo_pose_labels/ classes.txt pose.json
dataflow-cv convert coco2yolo --pose --save-kpts-config kpts.yaml coco_pose.json yolo_pose_labels/   # export the definition YAML

# Pose predictions (yolo2coco --prediction --pose → keypoints + score)
dataflow-cv convert yolo2coco --prediction --pose images/ yolo_pose_preds/ classes.txt pred.json
```

| Behavior | Non-pose mode (default) | Pose mode (`--pose`) |
|----------|-------------------------|----------------------|
| Objects **with** keypoints | Keypoints dropped with one WARNING per image; bbox/seg converted as usual | Keypoints converted; segmentation dropped for YOLO targets (pose rows cannot carry it) |
| Objects **without** keypoints | Converted as usual | Skipped with one WARNING per image |

> 📂 **COCO→YOLO / COCO→LabelMe output naming**: each `.txt` / `.json` is named
> by the image file stem — leading zeros preserved (`000001.jpg` →
> `000001.txt`). The converter generates `labels/` (or the `.json` files) +
> `classes.txt`; the `images/` directory is created but left **empty** —
> images are never copied, place your image files there yourself.

## 🎨 Visualize

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

Options: `--pose` (YOLO), `--kpts-config` (YOLO / LabelMe — adds skeleton
lines; COCO reads them from its categories), `--display/--no-display`,
`-s/--save PATH`, `--verbose`.

**Keyboard controls** (interactive window): `←`/`↑` previous, `→`/`↓`/`Enter`/
`Space` next, `s` snapshot (saved to `./visualized_snapshots/`), `h` hints,
`q`/`ESC` exit. First/last-image boundaries are no-ops; the title bar shows
`[N/T]` position. The window-manager close button may not work — always use
the keyboard.

## 📊 Evaluate

Evaluation always consumes **COCO-format** GT/DT JSON. Two files are required:

| File | Role | Format | How to create |
|------|------|--------|---------------|
| **`anno.json`** | Ground Truth (GT) | Full COCO dict (`images`, `annotations`, `categories`) | `yolo2coco` (label mode) |
| **`pred.json`** | Detection (DT) | Plain JSON list (with `score`) | `yolo2coco --prediction` |

### ① Prepare Data

```bash
# GT: YOLO labels → COCO
dataflow-cv convert yolo2coco images/ yolo_labels/ classes.txt anno.json

# DT: YOLO predictions → COCO (add --prediction for model output)
dataflow-cv convert yolo2coco --prediction images/ yolo_preds/ classes.txt pred.json
```

> ⚠️ `--prediction` is required for YOLO prediction files — they have an extra
> `confidence` token per line. The flag outputs a **plain JSON list** (not a
> full COCO dict), which is the standard DT format for `loadRes()`. Only
> `yolo2coco` supports `--prediction`; `labelme2coco` does not need it (LabelMe
> has no label vs prediction distinction).

### ② Run Evaluation

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

Options: `--prf1` (single-threshold P/R/F1, skips the COCOeval mAP pipeline;
run twice to get both), `--prf1-iou` (default 0.5), `--prf1-conf` (default
0.0), `--prf1-method macro|micro` (default macro), `-o/--output PATH`,
`--kpts-config` (keypoints sigmas), `--verbose`, `--log-dir`.

### ③ Detection vs Segmentation vs Pose

Three evaluation modes, distinguished by how overlap is measured:

- **Object Detection** — bounding box IoU. GT and DT require `bbox`; DT additionally requires `score`.
- **Instance Segmentation** — mask IoU. GT and DT require `bbox`, `segmentation` (polygon or RLE), and `area`; DT additionally requires `score`.
- **Pose Estimation** — OKS (Object Keypoint Similarity). GT and DT require `keypoints` (flat `[x, y, v, ...]` triplets) and `area`; DT additionally requires `score`. Per-keypoint sigmas resolve via `--kpts-config`, the COCO 17-keypoint official values, or a uniform fallback.

`yolo2coco` (label mode) and `yolo2coco --prediction` (prediction mode)
automatically populate all required fields for all three modes — no manual
editing needed.
