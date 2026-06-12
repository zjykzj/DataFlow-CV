# Evaluate Module Specification

> **Version:** 1.0
> **Status:** Draft
> **Layer:** Modules
> **Dependencies:** Label module (handlers + models), pycocotools

## 1. Module Overview

The Evaluate module (`dataflow/evaluate/`) computes detection and instance segmentation evaluation metrics using pycocotools. It depends **only** on the Label module — it does not import from Convert, Visualize, or CLI.

### 1.1 Key Design: COCO-Eval Wrapper

The module wraps pycocotools' `COCO` and `COCOeval` classes to provide:

- A clean Python API for evaluation (GT + DT in → structured `EvaluationResult` out)
- COCO-standard 12-metric output
- Per-class detailed breakdowns
- Single-threshold P/R/F1 computation (independent of full COCO eval)
- Verbose logging following the project's established contract

### 1.2 Module Contract

- **Input**: Ground truth + detection/prediction annotations in COCO format (file paths or in-memory)
- **Processing**: Load → validate → convert to pycocotools COCO objects → run COCOeval → extract results → compute per-class metrics
- **Output**: `EvaluationResult` (structured metrics container) or `PRF1Result` (single-threshold P/R/F1)
- **Dependency**: Label module only (for COCO handler and data models), pycocotools

### 1.3 File Map

```
dataflow/evaluate/
├── __init__.py             # Public API exports
├── base.py                 # BaseEvaluator abstract class + shared logic
├── evaluator.py            # DetectionEvaluator + SegmentationEvaluator
├── metrics.py              # compute_pr_f1() + per-class metric helpers
├── result.py               # EvaluationResult, EvaluationMetrics, PerClassMetrics, PRF1Result
└── utils.py                # Input normalization, COCO object construction, validation
```

## 2. Data Models (`result.py`)

### 2.1 `EvaluationMetrics`

The 12 COCO standard metrics:

```python
@dataclass
class EvaluationMetrics:
    # Average Precision
    ap: float              # IoU=0.50:0.95, area=all, maxDets=100
    ap50: float            # IoU=0.50,      area=all, maxDets=100
    ap75: float            # IoU=0.75,      area=all, maxDets=100
    ap_small: float        # IoU=0.50:0.95, area=small,  maxDets=100
    ap_medium: float       # IoU=0.50:0.95, area=medium, maxDets=100
    ap_large: float        # IoU=0.50:0.95, area=large,  maxDets=100

    # Average Recall
    ar_max_1: float        # IoU=0.50:0.95, area=all, maxDets=1
    ar_max_10: float       # IoU=0.50:0.95, area=all, maxDets=10
    ar_max_100: float      # IoU=0.50:0.95, area=all, maxDets=100
    ar_small: float        # IoU=0.50:0.95, area=small,  maxDets=100
    ar_medium: float       # IoU=0.50:0.95, area=medium, maxDets=100
    ar_large: float        # IoU=0.50:0.95, area=large,  maxDets=100
```

All values are `float` in [0, 1] or `-1.0` if the metric is undefined (e.g., no GT for that scale).

### 2.2 `PerClassMetrics`

Per-category detailed breakdown (populated in verbose mode):

```python
@dataclass
class PerClassMetrics:
    class_id: int
    class_name: str
    gt_count: int            # Number of GT annotations for this class
    dt_count: int            # Number of DT annotations for this class
    tp: int                  # True Positives (at IoU threshold 0.50:0.95 aggregate)
    fp: int                  # False Positives
    fn: int                  # False Negatives
    ap: float                # AP (IoU=0.50:0.95)
    ap50: float              # AP at IoU=0.50
    ap75: float              # AP at IoU=0.75
    precision: float         # P at IoU=0.50, optimal confidence
    recall: float            # R at IoU=0.50, optimal confidence
    f1_score: float          # F1 at IoU=0.50, optimal confidence
```

### 2.3 `EvaluationResult`

Top-level return type:

```python
@dataclass
class EvaluationResult:
    success: bool
    metrics: Optional[EvaluationMetrics]       # None if evaluation failed
    per_class: Optional[Dict[int, PerClassMetrics]]  # None if not verbose
    iou_type: str                              # "bbox" or "segm"
    gt_stats: Dict[str, int]                   # {"images": N, "annotations": M, "categories": K}
    dt_stats: Dict[str, int]                   # Same structure as gt_stats
    warnings: List[str]                        # Accumulated non-fatal warnings
    errors: List[str]                          # Error messages (non-empty if success=False)
    log_file_path: Optional[str]               # Verbose log file path (None if not verbose)
```

### 2.4 `PRF1Result`

Return type for the single-threshold P/R/F1 API:

```python
@dataclass
class PRF1Result:
    success: bool
    iou_threshold: float
    confidence_threshold: float
    overall: Optional[PRF1Values]
    per_class: Dict[int, PRF1Values]
    warnings: List[str]
    errors: List[str]

@dataclass
class PRF1Values:
    precision: float
    recall: float
    f1_score: float
    tp: int
    fp: int
    fn: int
```

## 3. Core Classes

### 3.1 `BaseEvaluator` (`base.py`)

Abstract base class implementing the template method pattern:

```python
class BaseEvaluator(ABC):
    def __init__(self, strict_mode=True, verbose=False, logger=None): ...

    # Template method — orchestrates the evaluation pipeline
    def evaluate(self, gt_source, dt_source) -> EvaluationResult: ...

    # Hook: Validate inputs before evaluation
    def validate_inputs(self, gt_coco, dt_coco) -> Tuple[bool, List[str]]: ...

    # Hook: Create COCOeval instance (abstract — iouType differs)
    @abstractmethod
    def _create_cocoeval(self, gt_coco, dt_coco) -> COCOeval: ...

    # Hook: Compute per-class metrics from COCOeval results
    def _compute_per_class(self, cocoeval, gt_coco, dt_coco) -> Dict[int, PerClassMetrics]: ...
```

**`evaluate()` pipeline:**

```
1. _load_coco(gt_source)        → pycocotools.COCO (GT)
2. _load_dt(coco_gt, dt_source) → pycocotools.COCO (DT)
   ├─ dt is list or list-file   → coco_gt.loadRes()
   └─ dt is dict / COCO file    → _load_coco()
3. validate_inputs(gt, dt)      → (valid, warnings)
4. _create_cocoeval(gt, dt)     → COCOeval instance
5. cocoeval.evaluate()          → per-image evaluation
6. cocoeval.accumulate()        → accumulate into PR arrays
7. cocoeval.summarize()         → compute 12 stats → self.stats
8. Extract metrics → EvaluationMetrics from cocoeval.stats
9. If verbose: _compute_per_class() → per-class details
10. Build EvaluationResult
```

### 3.2 `DetectionEvaluator` (`evaluator.py`)

```python
class DetectionEvaluator(BaseEvaluator):
    """Object detection evaluation using bbox IoU."""
    def _create_cocoeval(self, gt_coco, dt_coco) -> COCOeval:
        return COCOeval(gt_coco, dt_coco, iouType='bbox')
```

### 3.3 `SegmentationEvaluator` (`evaluator.py`)

```python
class SegmentationEvaluator(BaseEvaluator):
    """Instance segmentation evaluation using mask IoU."""
    def _create_cocoeval(self, gt_coco, dt_coco) -> COCOeval:
        return COCOeval(gt_coco, dt_coco, iouType='segm')
```

### 3.4 Constructor Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `strict_mode` | bool | No | True | If True, validation errors abort immediately. If False, skip invalid annotations with warnings. |
| `verbose` | bool | No | False | If True, compute per-class metrics + file logging. |
| `logger` | Optional[Logger] | No | None | Logger instance. Created internally if not provided. |

## 4. Public API

### 4.1 `evaluate(gt_source, dt_source) → EvaluationResult`

Full COCO evaluation. Computes all 12 standard metrics.

```python
def evaluate(
    self,
    gt_source: Union[str, Path, DatasetAnnotations, Dict],
    dt_source: Union[str, Path, DatasetAnnotations, Dict, List],
) -> EvaluationResult:
```

**GT input normalization** (`utils.py`):

| `gt_source` type | Processing |
|-----------------|-----------|
| `str` / `Path` | Treat as file path → load via `pycocotools.COCO(path)` |
| `DatasetAnnotations` | Convert to COCO dict (if format ≠ COCO, raise error) |
| `Dict` | Use directly as COCO dict |

**DT input normalization** (`utils.py`):

| `dt_source` type | Processing |
|-----------------|-----------|
| `str` / `Path` → file contains dict | Load via `pycocotools.COCO(path)` (full COCO dict with `images`/`annotations`/`categories`) |
| `str` / `Path` → file contains list | Load via `coco_gt.loadRes(path)` (plain annotation array — images/categories sourced from GT) |
| `DatasetAnnotations` | Convert to COCO dict (if format ≠ COCO, raise error) |
| `Dict` | Use directly as COCO dict |
| `List` | Pass directly to `coco_gt.loadRes(list)` |

DT loading is asymmetric from GT loading because prediction files are commonly
JSON arrays (list of annotation dicts with `bbox`/`score`). pycocotools provides
`loadRes()` specifically for this case — it copies `images` and `categories` from
the GT dataset and indexes the annotation list.

### 4.2 `compute_pr_f1()` — Standalone Function (`metrics.py`)

Single-threshold P/R/F1 computation. Does NOT require the full COCOeval pipeline.

```python
def compute_pr_f1(
    gt_source: Union[str, Path, DatasetAnnotations, Dict],
    dt_source: Union[str, Path, DatasetAnnotations, Dict],
    iou_threshold: float = 0.5,
    confidence_threshold: float = 0.0,
    iou_type: str = "bbox",
    verbose: bool = False,
    logger: Optional[logging.Logger] = None,
) -> PRF1Result:
```

**Algorithm:**

```
1. Load GT and DT → COCO objects (same normalization as evaluate())
2. Exclude crowd annotations (iscrowd=1) from GT matching pool.
   Crowd annotations do NOT generate FN when unmatched, and DTs
   matching crowd annotations are ignored (not counted as TP or FP).
   This follows pycocotools behavior (spec_evaluate_fundamentals.md §7.4).
3. For each (image, category):
   a. Filter GT: all non-crowd annotations for this (image, category)
   b. Filter DT: score ≥ confidence_threshold, sorted by score DESCENDING
   c. Run greedy matching (spec_evaluate_fundamentals.md §3.2) with iou_threshold
   d. Accumulate TP, FP, FN
4. Compute per-class P, R, F1 from per-class TP/FP/FN
5. Compute overall P, R via macro averaging across all categories:
     P_overall = (1/K) × Σ P_c
     R_overall = (1/K) × Σ R_c
   Then F1_overall = 2 × P_overall × R_overall / (P_overall + R_overall)
   (Categories with zero TP/FP have P_c=0.0; zero TP/FN have R_c=0.0)
6. Return PRF1Result with per_class dict and overall PRF1Values
```

**Note**: Unlike `evaluate()`, this uses manual matching (not COCOeval). This is intentional — for single-threshold F1, the full COCOeval pipeline (10 IoU thresholds × 101 recall thresholds) is unnecessary overhead.

### 4.3 Module-Level Convenience Functions

```python
# In __init__.py — re-exported for easy imports
from dataflow.evaluate.evaluator import DetectionEvaluator, SegmentationEvaluator
from dataflow.evaluate.metrics import compute_pr_f1
from dataflow.evaluate.result import (
    EvaluationResult, EvaluationMetrics, PerClassMetrics,
    PRF1Result, PRF1Values,
)
```

## 5. Input Format Contract

### 5.1 GT COCO JSON Requirements

Must be a valid COCO annotation file:

- `images` array: each image has `id` (int), `file_name` (str), `width` (int), `height` (int)
- `categories` array: each category has `id` (int), `name` (str)
- `annotations` array: each annotation has `id` (int), `image_id` (int), `category_id` (int), `bbox` ([x, y, w, h])

Additional fields for segmentation evaluation:
- `annotations[].segmentation`: polygon or RLE format
- `annotations[].area`: float (mask area in px²)
- `annotations[].iscrowd`: 0 or 1

### 5.2 DT COCO JSON Requirements

DT can be provided in **either** of two formats:

**Format A — Full COCO dict** (same structure as GT):
- `images` array: each image has `id` (int), `file_name` (str), `width` (int), `height` (int)
- `categories` array: each category has `id` (int), `name` (str)
- `annotations` array: each annotation has `id` (int), `image_id` (int), `category_id` (int), `bbox` ([x, y, w, h]), **`score`** (float ∈ [0,1])

**Format B — Plain annotation list** (JSON array):
- A top-level JSON array of annotation objects, each containing `image_id`, `category_id`, `bbox`, and `score`
- No `images` or `categories` arrays — these are sourced from GT at load time via `loadRes()`
- This is the most common output format from model inference (Detectron2, MMDetection, custom training scripts, etc.)

**Common requirement regardless of format:**
- Every DT annotation must include `score`: float ∈ [0, 1]

If any DT annotation is missing `score`, validation fails.

### 5.3 Validation Rules

The `validate_inputs()` method checks:

1. GT categories is non-empty
2. GT images is non-empty
3. DT's `image_id` values are a subset of GT's `image_id` values (warn if DT contains images not in GT)
4. Every DT annotation has a `score` field
5. DT `category_id` values are a subset of GT `category_id` values (warn on unknown categories)
6. At least one category has both GT and DT annotations
7. If `iouType='segm'`, segmentation data is present on GT and DT annotations that will be matched

## 6. Verbose Mode Contract

### 6.1 Console Output

When `verbose=True`, the evaluator outputs:

**1. Summary header:**
```
Evaluation: detection (bbox)
Ground Truth: 500 images, 3250 annotations, 10 categories
Detections:   500 images, 4100 detections, 10 categories
```

**2. 12 COCO standard metrics (always printed):**
```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.352
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.568
...
```

**3. Per-class breakdown table (verbose only):**
```
Per-Class Breakdown (IoU: 0.50:0.95):
───────────────────────────────────────────────────────────────────────────
 Class          GT    DT     TP    FP    FN     AP     AP50   AP75   P      R      F1
 person         520   610   487   123    33   0.432  0.689  0.451  0.798  0.937  0.862
 car            380   450   342   108    38   0.401  0.634  0.422  0.760  0.900  0.824
 bicycle        150   180   128    52    22   0.321  0.521  0.338  0.711  0.853  0.775
 ...
───────────────────────────────────────────────────────────────────────────
```

### 6.2 File Logging

When `verbose=True`:
- Logger is configured with file output via `VerboseLoggingOperations`
- Log file path is recorded in `EvaluationResult.log_file_path`
- Detailed processing steps logged at DEBUG level: input loading, validation, COCOeval stages, per-class computation
- Evaluation duration is recorded

When `verbose=False`:
- Console-only INFO logging
- Per-class metrics are NOT computed (Performance optimization)
- `EvaluationResult.per_class` is `None`

## 7. Dependency Contract

```
Evaluate module imports FROM:
├── dataflow.label.models             (DatasetAnnotations, AnnotationFormat)
├── dataflow.label.coco_handler       (CocoAnnotationHandler — for loading COCO files)
├── dataflow.util                     (FileOperations, logging)
├── pycocotools.coco                  (COCO)
├── pycocotools.cocoeval              (COCOeval)
└── numpy                             (Array operations for per-class extraction)

Evaluate module does NOT import FROM:
├── dataflow.convert.*                (FORBIDDEN — zero cross-dependency)
├── dataflow.visualize.*              (FORBIDDEN — zero cross-dependency)
├── dataflow.cli.*                    (FORBIDDEN — CLI depends on Evaluate, not vice versa)
├── dataflow.label.yolo_handler       (NOT NEEDED — evaluation works on COCO format only)
└── dataflow.label.labelme_handler    (NOT NEEDED — evaluation works on COCO format only)
```

## 8. Architecture Position

```
┌──────────────────────────────────────────────────────────────┐
│                           CLI                                 │
│  (calls Convert, Visualize & Evaluate public APIs)            │
└──────┬─────────────────────┬──────────────────┬──────────────┘
       │                     │                  │
       ▼                     ▼                  ▼
┌──────────────┐    ┌──────────────────┐    ┌──────────────┐
│   Convert    │    │    Visualize     │    │   Evaluate   │  ← NEW
│  (pipeline)  │    │  (rendering)     │    │  (metrics)   │
└──────┬───────┘    └───────┬──────────┘    └──────┬───────┘
       │                    │                      │
       │    ZERO CROSS-     │    ZERO CROSS-       │
       │    DEPENDENCY      │    DEPENDENCY        │
       │                    │                      │
       ▼                    ▼                      ▼
┌──────────────────────────────────────────────────────────────┐
│                         Label                                 │
│  Data Models + Handlers (read/write/validate)                 │
└──────────────────────────────────────────────────────────────┘
```

**Hard constraints:**

1. **Evaluate ↔ Convert**: Zero dependency. They do not import from each other.
2. **Evaluate ↔ Visualize**: Zero dependency. They do not import from each other.
3. **Evaluate → Label**: Evaluator imports COCO handler and models only through public interfaces.
4. **CLI → Evaluate**: CLI commands only call evaluator public APIs. CLI must NOT import pycocotools directly.

## 9. Error Handling Contract

### 9.1 Error Propagation

```
validate_inputs() fails   → EvaluationResult(success=False, errors=[...])
COCO loading fails        → EvaluationResult(success=False, errors=[...])
COCOeval.evaluate() fails → EvaluationResult(success=False, errors=[...])
COCOeval.accumulate() fails → EvaluationResult(success=False, errors=[...])
```

### 9.2 Specific Error Scenarios

| Error | Behavior |
|-------|----------|
| GT/DT file not found | `EvaluationResult(success=False, errors=["File not found: ..."])` |
| DT missing `score` field | `EvaluationResult(success=False, errors=["N DT annotations missing 'score' field"])` |
| No GT for any category | `EvaluationResult(success=False, errors=["GT contains no annotations"])` |
| pycocotools not installed | `ImportError` with message: "pycocotools is required for evaluation. Install with: pip install pycocotools" |
| Empty category (has GT but no DT) | Warning in `warnings` list; metrics for that category are -1.0 |
| DT image_id not in GT | Warning in `warnings` list; those DTs are excluded |
| `iouType='segm'` but no segmentation data | `EvaluationResult(success=False, errors=["Segmentation data missing for iouType='segm'"])` |
| All categories have zero GT | `EvaluationResult(success=False, errors=["No categories with GT annotations found"])` |

### 9.3 Strict vs Non-Strict

| Error Type | Strict Mode | Non-Strict Mode |
|------------|-------------|-----------------|
| DT annotation missing `score` | Abort with error | Skip DT annotation, log warning, continue |
| DT category_id not in GT | Abort with error | Skip DT annotation, log warning, continue |
| DT image_id not in GT | Abort with error | Skip DT annotation, log warning, continue |
| Invalid bbox (zero area) | Abort with error | Skip annotation, log warning, continue |
| pycocotools not installed | **Always abort** (cannot evaluate without it) | **Always abort** |

## 10. COCO Object Construction (`utils.py`)

### 10.1 `_load_coco(source) → COCO`

Internal utility that normalizes GT input types to a `pycocotools.COCO` instance.
For DT with list format, loading is routed through `coco_gt.loadRes()` instead
(see §4.1 DT input normalization).

```
source is str/Path:
  → Load file via pycocotools.COCO(path)
  → File must contain a valid COCO dict (images, annotations, categories)

source is DatasetAnnotations:
  → Verify format == COCO
  → Reconstruct COCO dict from DatasetAnnotations fields
  → pycocotools.COCO() with in-memory dict (via temporary JSON file round-trip)

source is Dict:
  → Validate required keys ("images", "annotations", "categories")
  → pycocotools.COCO() with in-memory dict (via temporary JSON file round-trip)
```

**DT list-format handling** (in `BaseEvaluator._load_dt()`):

When `dt_source` is a `list` (in-memory) or a file path whose content is a JSON
array (not a dict), loading is delegated to `coco_gt.loadRes()`:

```
dt_source is list:
  → coco_gt.loadRes(list)

dt_source is str/Path → file contains list:
  → coco_gt.loadRes(str(path))

dt_source is str/Path → file contains dict:
  → _load_coco(path)   (standard path, same as GT)
```

`loadRes()` copies `images` and `categories` from the GT COCO object and indexes
the provided annotation list. This is pycocotools' designated API for loading
prediction results in list format.

### 10.2 DT Score Extraction

When loading DT from `DatasetAnnotations`:
- DT must carry `confidence` field on each `ObjectAnnotation`
- This field maps to COCO `score` during conversion

## 11. Verbose Logging Contract

Follows the project's established verbose logging pattern (see `spec_convert.md` §2.7, `spec_visualize.md` §8):

When `verbose=True`:
- Logger is created via `VerboseLoggingOperations.get_verbose_logger()` (console + file)
- Console uses `DEFAULT_FORMAT` (timestamps)
- File logs include filename and line numbers (DEBUG level)
- Per-class metrics are computed and stored in `EvaluationResult.per_class`
- `log_file_path` is recorded in `EvaluationResult`

When `verbose=False`:
- Console-only via `LoggingOperations.get_logger()` (INFO level)
- Per-class metrics are **skipped** to avoid unnecessary computation
- `EvaluationResult.per_class` is `None`

## 12. Summary of Public API

| API | Location | Purpose |
|-----|----------|---------|
| `DetectionEvaluator(verbose, strict_mode, logger)` | `evaluator.py` | Detection evaluation (bbox IoU) |
| `SegmentationEvaluator(verbose, strict_mode, logger)` | `evaluator.py` | Segmentation evaluation (mask IoU) |
| `evaluator.evaluate(gt, dt) → EvaluationResult` | `base.py` | Run full COCO evaluation |
| `compute_pr_f1(gt, dt, iou_thr, conf_thr, iou_type) → PRF1Result` | `metrics.py` | Single-threshold P/R/F1 |
