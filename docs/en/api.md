# Python API Reference

> dataflow-cv v3.6 — the Python API mirrors every CLI capability. Examples assume
> an editable or PyPI install (`pip install dataflow-cv`).

## Overview

| Module | Key classes / functions | Purpose |
|--------|-------------------------|---------|
| `dataflow.label` | `YoloAnnotationHandler`, `LabelMeAnnotationHandler`, `CocoAnnotationHandler`, `DatasetAnnotations` | Read / write / validate annotations |
| `dataflow.analyse` | `StatsAnalyser`, `SplitAnalyser`, `FilterAnalyser`, `PartitionAnalyser`, `SampleAnalyser` | Dataset introspection & preparation |
| `dataflow.convert` | `YoloAndCocoConverter`, `LabelMeAndYoloConverter`, `CocoAndLabelMeConverter` | Format conversion |
| `dataflow.visualize` | `YOLOVisualizer`, `LabelMeVisualizer`, `COCOVisualizer` | OpenCV rendering |
| `dataflow.evaluate` | `DetectionEvaluator`, `SegmentationEvaluator`, `KeypointEvaluator`, `compute_pr_f1` | COCO metrics |
| `dataflow.util.logging` | `LogConfig`, `LogManager` | Unified logging |

## Common Patterns

### Logging

Every module accepts `log_config: Optional[LogConfig] = None`. `verbose=True`
adds a per-run log file; `log_dir` controls where it goes (default `./logs/`).

```python
from dataflow.util.logging import LogConfig

log_cfg = LogConfig(name="my-pipeline", verbose=True, log_dir="./logs/")
```

### Result objects

All operations return a result container; check `result.success` before using
`result.data`.

| Result | Fields | Returned by |
|--------|--------|-------------|
| `AnalysisResult` | `success`, `data`, `errors`, `warnings`, `log_path` | `*.analyse()` |
| `ConversionResult` | `success`, `data`, `errors`, `warnings`, `log_path` | `converter.convert()` |
| `EvaluationResult` | `success`, `metrics` (12 COCO metrics + `ar_max_20`), `per_class` (verbose), `errors` | `evaluator.evaluate()` |
| `PRF1Result` | `success`, `overall` (`precision`/`recall`/`f1_score`), `per_class`, `thresholds` | `compute_pr_f1()` |

## Label Handlers

Read/write annotations in their native formats. Handlers store coordinates in
the format's native representation — always check `DatasetAnnotations.format`
to interpret them (see [Data Models](#data-models)).

```python
from dataflow.label import CocoAnnotationHandler, LabelMeAnnotationHandler, YoloAnnotationHandler

# YOLO (detection / segmentation / pose via pose=True)
handler = YoloAnnotationHandler(
    label_dir="yolo_labels/",      # one .txt per image
    class_file="classes.txt",      # class names (line number = class id)
    image_dir="images/",
    pose=True,                     # explicit for YOLO pose rows (5+3k tokens)
    strict_mode=True,
)
result = handler.read()            # -> DatasetAnnotations (all data in memory)
dataset = result.data
print(dataset.format, len(dataset.images), dataset.num_objects)

# LabelMe (per-image .json; kpts_config orders pose keypoints)
handler = LabelMeAnnotationHandler(label_dir="labelme_json/", kpts_config="kpts.yaml")

# COCO (single .json; prediction=True reads/writes the plain-list prediction format)
handler = CocoAnnotationHandler(annotation_file="anno.json", do_rle=False)
```

| Handler | Constructor | Notes |
|---------|-------------|-------|
| `YoloAnnotationHandler` | `(label_dir, class_file, image_dir=None, pose=False, strict_mode=True, **kwargs)` | `prediction=True` (via kwargs) parses prediction rows (extra confidence token) |
| `LabelMeAnnotationHandler` | `(label_dir, class_file=None, kpts_config=None, strict_mode=True, **kwargs)` | Pose = `point` shapes + a `rectangle` sharing a `group_id` |
| `CocoAnnotationHandler` | `(annotation_file, do_rle=False, prediction=False, strict_mode=True, **kwargs)` | `prediction=True` → `write()` outputs a plain JSON list (Variant B) |

Each handler also provides `write()` (batch), `write_one()` (per image; COCO
raises `NotImplementedError`), `validate()`, and `iter_images()` (streaming).

## Analyse

All analysers take `log_config` in the constructor and read annotations with
`strict_mode=False` (read-only, lenient — errors accumulate in
`result.errors`).

```python
from dataflow.analyse import FilterAnalyser, PartitionAnalyser, SampleAnalyser, SplitAnalyser, StatsAnalyser

# Statistics (auto-detects YOLO / LabelMe / COCO from the path)
stats = StatsAnalyser(log_config=log_cfg).analyse(
    "yolo_labels/", image_dir="images/", class_file="classes.txt",
    sort_by="count", descending=True, recursive=False,
    kpts_config="kpts.yaml",   # enables keypoint stats (and YOLO pose-row parsing)
)

# Train/val split (YOLO / LabelMe only — labels / images / both modes)
split = SplitAnalyser().analyse(
    output_dir="split/", ratio=0.8, seed=42,
    label_dir="yolo_labels/", image_dir="images/", class_file="classes.txt",
    move=False,                # copy (False) or move (True) files
)

# Category filter (keep a subset; remap IDs per the new classes.txt)
filtered = FilterAnalyser().analyse(
    "yolo_labels/", original_class_file="classes.txt",
    new_class_file="classes_new.txt", output_dir="filtered/",
)

# N-way partition (YOLO / LabelMe only)
partition = PartitionAnalyser().analyse(
    output_dir="parts/", num=4,
    label_dir="yolo_labels/", image_dir="images/", shuffle=True, seed=42,
)

# File sampling — collect N files (labels / images / both; random or sequential)
sample = SampleAnalyser().analyse(
    output_dir="sampled/", count=10,
    label_dir="yolo_labels/", shuffle=True, seed=42, move=False,
)
```

## Convert

`convert()` auto-dispatches: COCO target → batch pipeline, YOLO/LabelMe
target → streaming pipeline. `source_to_target=True` means "source name →
target name" in the class name (see the table).

```python
from dataflow.convert import CocoAndLabelMeConverter, LabelMeAndYoloConverter, YoloAndCocoConverter

result = YoloAndCocoConverter(source_to_target=True).convert(
    source_path="yolo_labels/", target_path="anno.json",
    class_file="classes.txt", image_dir="images/",
    do_rle=False,               # YOLO→COCO: RLE segmentation output
    pose=True,                  # pose conversion (all six directions)
    kpts_config="kpts.yaml",    # keypoint names/skeleton for the target
)
```

| Converter | `source_to_target=True` | `source_to_target=False` | Extra params |
|-----------|-------------------------|--------------------------|--------------|
| `YoloAndCocoConverter` | YOLO → COCO | COCO → YOLO | `prediction=True` for model output (`yolo2coco --prediction` only); `do_rle`; `save_kpts_config` (COCO→YOLO) |
| `LabelMeAndYoloConverter` | LabelMe → YOLO | YOLO → LabelMe | `kpts_config` |
| `CocoAndLabelMeConverter` | LabelMe → COCO | COCO → LabelMe | `save_crowd`; `save_kpts_config` (COCO→LabelMe) |

All converters accept `pose`, `kpts_config`, and `save_kpts_config` where
applicable. COCO→YOLO / COCO→LabelMe skip `iscrowd=1` annotations by default
(`save_crowd=True` keeps them).

## Visualize

```python
from dataflow.visualize import COCOVisualizer, LabelMeVisualizer, YOLOVisualizer

visualizer = YOLOVisualizer(
    label_dir="yolo_labels/",
    image_dir="images/",
    class_file="classes.txt",
    pose=True,                  # render keypoints + skeleton (YOLO)
    kpts_config="kpts.yaml",    # skeleton source (YOLO / LabelMe)
    is_show=True,               # interactive window (keyboard navigation)
    is_save=True,               # also save each frame
    output_dir="visualized/",
    log_config=log_cfg,
)
result = visualizer.visualize()
```

| Visualizer | Constructor | Notes |
|-----------|-------------|-------|
| `YOLOVisualizer` | `(label_dir, image_dir, class_file, pose=False, kpts_config=None, **kwargs)` | pose rendering needs `pose=True` |
| `LabelMeVisualizer` | `(label_dir, image_dir, class_file=None, kpts_config=None, **kwargs)` | keypoints render automatically when present |
| `COCOVisualizer` | `(annotation_file, image_dir, **kwargs)` | skeleton comes from COCO categories |

Common kwargs (from `BaseVisualizer`): `output_dir`, `is_show=True`,
`is_save=False`, `log_config`. Headless mode: `is_show=False` + `is_save=True`.

## Evaluate

Inputs may be file paths, in-memory dicts/lists, or `DatasetAnnotations`.
GT must be a full COCO dict; DT may be a full dict or a plain JSON list
(Variant B, requires `score`).

```python
from dataflow.evaluate import DetectionEvaluator, KeypointEvaluator, SegmentationEvaluator, compute_pr_f1

evaluator = DetectionEvaluator(log_config=LogConfig(name="eval", verbose=True))
result = evaluator.evaluate("anno.json", "pred.json")
print(f"AP: {result.metrics.ap:.3f}, AP50: {result.metrics.ap50:.3f}")
# SegmentationEvaluator: same API, mask IoU (iouType='segm')
# KeypointEvaluator(sigmas=None): OKS; sigmas resolve via explicit list,
#   COCO 17-keypoint official values, or uniform 0.05 + WARNING

# Quick single-threshold P/R/F1 (skips the COCOeval mAP pipeline)
prf1 = compute_pr_f1(
    "anno.json", "pred.json",
    iou_threshold=0.5,          # IoU threshold (OKS threshold for keypoints)
    confidence_threshold=0.0,
    iou_type="bbox",            # "bbox" | "segm" | "keypoints"
    method="macro",             # "macro" (per-class mean) | "micro" (pooled)
    kpt_sigmas=None,            # per-keypoint sigmas for iou_type="keypoints"
)
print(f"Macro F1: {prf1.overall.f1_score:.3f}")
```

> `KeypointEvaluator` yields 10 COCO stats (no small scale, `maxDets=20`) —
> `metrics.ar_max_20` carries the AR; `ap_small`/`ar_small`/`ar_max_1/10/100`
> stay `-1.0`.

## Data Models

```python
from dataflow.label.models import DatasetAnnotations
```

| Model | Key fields | Notes |
|-------|-----------|-------|
| `DatasetAnnotations` | `format`, `images`, `categories`, `num_objects`, `dataset_info` | `format` defines coordinate semantics: `YOLO` = center + normalized [0,1]; `LABELME` / `COCO` = top-left + absolute pixels |
| `ImageAnnotation` | `width`, `height`, `objects`, `image_path`, `image_id` | one per image |
| `ObjectAnnotation` | `class_id`, `class_name`, `bbox`, `segmentation`, `keypoints`, `confidence`, `is_crowd` | at least one of bbox / segmentation / keypoints is set |
| `BoundingBox` | `x`, `y`, `width`, `height` | **format-native** (see above) |
| `Segmentation` | `points` | polygon in native coordinates |
| `Keypoints` | `points`, `visibility`, `count_labeled()` | `v ∈ {0,1,2}`: 0 = not labeled (stored at 0,0), 1 = occluded, 2 = visible |

## See Also

- [samples/](../../samples/) — runnable examples for every module
- CLI reference: `dataflow-cv --help` and each subcommand's `--help`
- [README](../../README.md) — quick start and end-to-end workflow
