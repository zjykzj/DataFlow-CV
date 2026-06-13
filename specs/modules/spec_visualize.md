# Visualize Module Specification

> **Version:** 4.3
> **Status:** Draft — industry-standard label positioning (top-left, class-color background, inside-bbox edge flip)
> **Layer:** Modules
> **Dependencies:** Label module (handlers + models) + Logging module (LogManager)

## 1. Module Overview

The Visualize module (`dataflow/visualize/`) renders annotation data onto images for visual
inspection. It depends **only** on the Label module — it does not import from Convert or CLI.

### 1.1 Key Design: Streaming + Unified Rendering Format

Visualizers stream `ImageAnnotation` objects from the Label handler's `iter_images()` and
convert coordinates to a **unified intermediate rendering format** per-image. This means:

- One rendering pipeline handles all source formats
- Coordinate conversion to absolute pixels happens per-image, in one place
- Visualizers don't need to accumulate the entire dataset in memory
- First image appears as soon as the first annotation file is parsed (streaming)

### 1.2 Module Contract

- **Input**: Label handler's `iter_images()` streaming iterator — yields `ImageAnnotation`
  objects (format-native coordinates) one at a time, + image files
- **Processing**: Per-image: convert annotations to absolute-pixel render data → draw → display/save
- **Output**: Rendered images (display window and/or saved files)
- **Dependency**: Label module only (for handlers and data models)

### 1.3 File Map

```
dataflow/visualize/
├── base.py                  # BaseVisualizer + ColorManager + VisualizationResult + RenderData
├── yolo_visualizer.py       # YOLO visualization (wraps YOLO handler + converts to RenderData)
├── labelme_visualizer.py    # LabelMe visualization (wraps LabelMe handler + converts to RenderData)
├── coco_visualizer.py       # COCO visualization (wraps COCO handler + converts to RenderData)
└── utils.py                 # Image scaling, text positioning utilities
```

## 2. Unified Rendering Format

### 2.1 `RenderAnnotation`

All annotations are converted to this unified format before drawing:

```python
@dataclass
class RenderAnnotation:
    class_name: str
    class_id: int
    
    # Bounding box in absolute pixel coordinates [x1, y1, x2, y2]
    # x1, y1 = top-left corner; x2, y2 = bottom-right corner
    bbox: Optional[Tuple[int, int, int, int]]
    
    # Polygon points in absolute pixel coordinates [(x1, y1), (x2, y2), ...]
    polygon: Optional[List[Tuple[int, int]]]
    
    # RLE mask data (COCO only, preserved as-is)
    rle: Optional[Dict[str, Any]]
```

### 2.2 `RenderData`

```python
@dataclass
class RenderData:
    annotations: List[RenderAnnotation]
    image_width: int
    image_height: int
```

### 2.3 Coordinate Conversion to RenderData

The conversion from `ImageAnnotation` (format-native coords) → `RenderData`:

| Source Format | Bounding Box Conversion | Polygon Conversion |
|---------------|------------------------|-------------------|
| YOLO | `cx_norm→cx_abs, cy_norm→cy_abs` → `int(cx-w/2), int(cy-h/2), int(cx+w/2), int(cy+h/2)` | `int(x*W), int(y*H)` per point |
| COCO | `x_tl, y_tl, w, h` → `x_tl, y_tl, x_tl+w, y_tl+h` | Use absolute pixel points as-is |
| LabelMe | `x_tl, y_tl, w, h` → `x_tl, y_tl, x_tl+w, y_tl+h` | Use absolute pixel points as-is |

**All coordinates are truncated to integers via `int()` for OpenCV drawing compatibility.**

**Bbox from polygon fallback**: When `obj.bbox` is `None` but `obj.segmentation` exists (e.g.,
YOLO segmentation format stores polygon points without a separate bbox), the visualizer
**must** compute the bbox from the polygon's axis-aligned bounds:

```python
xs = [p[0] for p in polygon_points_absolute]
ys = [p[1] for p in polygon_points_absolute]
bbox = (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))
```

This ensures every `RenderAnnotation` has a bbox for consistent label positioning (§4.3).

This conversion is lossy (integer truncation), but since it's used only for display (not for
saving annotations), precision loss in the rendering pipeline is acceptable.

### 2.4 Conversion Logic Location

Coordinate conversion to `RenderData` is done **per-image** in each concrete visualizer's
`_convert_to_render_data()` method. The conversion happens on-the-fly during iteration:

1. The Label handler's `iter_images()` yields one `ImageAnnotation` at a time
   (format-native coordinates)
2. `_convert_to_render_data(image_ann)` converts that single image's annotations
   to `RenderData` using the conversion rules above
3. The result is immediately rendered — no accumulation of all images in memory

**Streaming advantage**: The first image is displayed as soon as the first annotation
file is parsed, rather than waiting for all files to load. Memory usage is proportional
to the largest single image's annotations, not the entire dataset.

**No dependency on the Convert module** — each visualizer implements its own lightweight
coordinate conversion to absolute pixels.

## 3. Core Components

### 3.1 `ColorManager`

HSV-based color palette generator that ensures consistent, unique colors per class ID.

**Algorithm:**
1. Generate 1000 unique colors using HSV space
2. Primary: vary hue (0–179) with step = `max(1, 180 // (num_colors // 3))`
3. Secondary: vary saturation (100–200)
4. Tertiary: vary value (155–255)
5. Collision resolution: adjust saturation/value; fallback to hue shift

**Public API:**

| Method | Description |
|--------|-------------|
| `get_color(class_id)` → `(B, G, R)` | Returns cached or newly generated BGR color for a class |
| Color cache | `Dict[int, Tuple[int, int, int]]` — ensures deterministic color per class_id |

**Fallback for class_id ≥ 1000:**
- Deterministic HSV formula: `hue = (class_id * 127) % 180`, `sat = 100 + (class_id * 67) % 100`, `val = 155 + (class_id * 37) % 100`

**Debug mode:** When `debug=True`, log color assignments to stderr.

### 3.2 `VisualizationResult`

| Field | Type | Description |
|-------|------|-------------|
| `success` | bool | Operation succeeded |
| `data` | Optional[Any] | Result data (`{"processed_count": N, "interrupted": bool}`) |
| `message` | str | Status message |
| `errors` | List[str] | Error messages |
| `log_path` | Optional[str] | Log file path (verbose mode) |

### 3.3 `BaseVisualizer`

Abstract base class implementing the template method pattern.

**Constructor parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `label_dir` | Path | **Yes** | — | Annotation directory |
| `image_dir` | Path | **Yes** | — | Image directory |
| `output_dir` | Optional[Path] | No | None | Save directory (required if `is_save=True`) |
| `is_show` | bool | No | True | Display visualization window |
| `is_save` | bool | No | False | Save rendered images |
| `log_config` | Optional[LogConfig] | No | None | Logging configuration (see `spec_logging.md`) |

**Drawing configuration (`self.config`):**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `bbox_thickness` | 2 | Bounding box line width |
| `seg_thickness` | 1 | Segmentation outline width |
| `seg_alpha` | 0.3 | Segmentation fill transparency |
| `text_thickness` | 1 | Text stroke width |
| `text_scale` | 0.5 | Font scale factor |
| `text_padding` | 5 | Text offset from bbox top |
| `font` | `FONT_HERSHEY_SIMPLEX` | OpenCV font |

## 4. Visualization Pipeline

### 4.1 Template Method: `visualize()`

```
visualize()
├── 1. Validate output_dir              # Required if is_save=True
├── 2. Obtain image iterator             # From handler.iter_images()
├── 3. For each image (streaming):
│   ├── handler.iter_images() → ImageAnnotation (format-native)
│   ├── _convert_to_render_data()  → RenderData (absolute pixel)
│   ├── _visualize_single_image()
│   │   ├── Load image (resolve path, cv2.imread)
│   │   ├── For each RenderAnnotation:
│   │   │   ├── _draw_bbox()       # If bbox exists
│   │   │   ├── _draw_polygon()    # If polygon exists
│   │   │   └── _draw_rle_mask()   # If RLE exists
│   │   ├── Display (cv2.imshow)   # If is_show=True
│   │   └── Save (cv2.imwrite)     # If is_save=True
│   └── Handle keyboard input (q/ESC stops iteration)
└── 4. Return VisualizationResult
```

**Streaming semantics**: Images are loaded, converted, and displayed one at a time.
The handler's `iter_images()` yields each `ImageAnnotation` incrementally — no
batch accumulation. This means:
- **Low first-image latency**: The first image appears as soon as parsing completes
  for the first annotation file, not after all files are processed.
- **Low memory**: Only the current image's annotation data is held in memory.
- **Incremental summary**: `total_images` and `total_objects` are updated as images
  are processed, rather than known upfront.
- **Progress display**: Uses counter format (`Processing image 5`) instead of
  percentage-based progress bars (total count unknown until iteration completes).

### 4.2 Abstract Methods: `_create_handler()` and `_convert_to_render_data()`

Each concrete visualizer must implement:

```python
@abstractmethod
def _create_handler(self) -> BaseAnnotationHandler:
    """Create the format-specific Label handler instance."""
    ...

@abstractmethod
def _convert_to_render_data(self, image_ann: ImageAnnotation) -> RenderData:
    """Convert a single ImageAnnotation (format-native coords) to RenderData (absolute pixel coords).

    This is called per-image during the streaming loop. The conversion logic
    uses the format-specific coordinate transform rules (Section 2.3).

    Each concrete visualizer implements format-specific conversion:
    - YOLOVisualizer: normalized center → absolute pixel top-left
    - COCOVisualizer: absolute pixel top-left → absolute pixel top-left (passthrough)
    - LabelMeVisualizer: absolute pixel top-left → absolute pixel top-left (passthrough)

    Args:
        image_ann: Single ImageAnnotation with format-native coordinates.

    Returns:
        RenderData with absolute-pixel annotations ready for drawing.
    """
```

**Streaming loading flow in `visualize()`:**

```python
def visualize(self) -> VisualizationResult:
    handler = self._create_handler()
    image_iter = handler.iter_images()
    for image_ann in image_iter:
        render_data = self._convert_to_render_data(image_ann)
        success = self._visualize_single_image(image_ann.image_path, render_data)
        ...
```

**Key design points:**
1. **Handler creates the iterator** — `_create_handler()` replaces the old
   `load_annotations()` setup step. The handler is created once and its
   `iter_images()` method provides the streaming iterator.
2. **Per-image coordinate conversion** — `_convert_to_render_data()` replaces the
   old bulk conversion loop inside `load_annotations()`. It operates on a single
   `ImageAnnotation` at a time.
3. **No `Dict[str, RenderData]` accumulator** — RenderData is created and consumed
   in the same loop iteration. Nothing is stored for later.
4. **Format awareness** — Each concrete visualizer knows its source format and
   applies the corresponding coordinate transform from Section 2.3.

### 4.3 Drawing Methods

#### Per-Annotation Drawing Flow

For each `RenderAnnotation`, drawing proceeds in this order:

```
1. Determine label position:
   ├── bbox exists → label_pos = bbox top-left (x1, y1 - 2)
   └── bbox is None, polygon exists → label_pos = polygon[0] (fallback)

2. Draw bbox (if present):
   └── cv2.rectangle(image, (x1,y1), (x2,y2), color, thickness)

3. Draw polygon (if present):
   ├── cv2.fillPoly() semi-transparent fill (alpha from config)
   └── cv2.polylines() outline

4. Draw RLE mask (if present):
   └── Decode mask → semi-transparent overlay

5. Draw label (if position determined):
   └── Class-color background rectangle (matches bbox/polygon color)
       White text, top-left aligned with bbox
       Edge case: if label would extend above image top → flip inside bbox
```

**Bbox from polygon fallback**: In `_convert_to_render_data()`, if a `RenderAnnotation`
has a polygon but no bbox (e.g., YOLO segmentation format stores polygon points only),
the visualizer **must** compute the bbox from the polygon's min/max x,y:

```python
if polygon and bbox is None:
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    bbox = (min(xs), min(ys), max(xs), max(ys))
```

This ensures label positioning is always based on a bounding box rather than an
arbitrary polygon vertex, giving consistent placement across all annotation types.

**Label position**: `(x1, y1 - 2)` (above bbox, top-left aligned).
Background rectangle uses the same class color as the bbox/polygon, providing
visual association between label and annotation. Text is white for contrast.
If bbox is None but polygon exists (no bbox computed), falls back to polygon's
first point.

#### `_draw_bbox(image, bbox, color, class_name)`

- `bbox`: `(x1, y1, x2, y2)` in absolute pixels (from `RenderAnnotation` — already in absolute coords)
- Draw rectangle with `cv2.rectangle()`

#### `_draw_polygon(image, polygon, color, class_name)`

- `polygon`: `[(x1, y1), (x2, y2), ...]` in absolute pixels (from `RenderAnnotation`)
- Draw semi-transparent fill with `cv2.fillPoly()` + `cv2.addWeighted()` (alpha from config)
- Draw polygon outline with `cv2.polylines()`
- Label is drawn by the per-annotation flow (not inside `_draw_polygon`) — positioned above the bbox

#### `_draw_rle_mask(image, rle, color)`

1. Decode RLE to binary mask via pycocotools (`coco_mask.decode()`)
2. Create color mask × BGR color values
3. Semi-transparent overlay with `cv2.addWeighted()` + `np.copyto()`

Requires pycocotools — logs error and returns without drawing if unavailable.

#### `_draw_text(image, text, position, color, bbox)`

1. Calculate text bounding box via `cv2.getTextSize()`
2. If text extends above image top edge (`y1 - text_padding - text_height < 0`):
   - Flip label **inside** bbox: `position = (x1, y1 + text_height + text_padding)`
3. Draw **class-color** background rectangle (from `color` parameter, clamped to image boundaries)
4. Draw **white** text with `cv2.putText()` (anti-aliased)

### 4.4 Keyboard Interaction (Display Mode)

When `is_show=True`, each image is shown in a window:

| Key | Action |
|-----|--------|
| Enter / Space | Continue to next image |
| `q` / ESC | Stop visualization (returns `None` from `_visualize_single_image`) |

When the user interrupts, `VisualizationResult.data` includes `{"interrupted": True}`.

### 4.5 Save Mode

When `is_save=True`:
- Rendered images are saved to `output_dir/{image_id}_visualized.jpg`
- JPEG quality: 95
- `output_dir` is created if it doesn't exist

### 4.6 Progress Feedback

When a progress logger is available (verbose mode), progress is reported every 10 images
with a counter-based format (total count is unknown in streaming mode):

```
[====] Processed 40 images, 0 failed
```

Percentage-based progress bars are not used because the total image count is not
known until the iterator exhausts. If the handler provides a total count hint
(e.g., from file enumeration), the visualizer may optionally display both count
and percentage.

## 5. Concrete Visualizers

### 5.1 `YOLOVisualizer`

**Constructor:** `YOLOVisualizer(label_dir, image_dir, class_file, log_config=None, **kwargs)`

- Creates `YoloAnnotationHandler` internally via `_create_handler()`
- `class_file` is required (passed to handler)

**`_create_handler()`:**
```python
def _create_handler(self) -> YoloAnnotationHandler:
    return YoloAnnotationHandler(
        label_dir=str(self.label_dir),
        class_file=str(self.class_file),
        image_dir=str(self.image_dir),
        strict_mode=False,
        logger=self.logger,
    )
```

**`_convert_to_render_data(image_ann)`:**
For each `ObjectAnnotation` in `image_ann.objects`:
1. If `obj.bbox`: convert YOLO-native (normalized center `[cx,cy,w,h]`) to
   absolute pixel `[x1,y1,x2,y2]`:
   - `cx_abs = cx * image_width`
   - `cy_abs = cy * image_height`
   - `half_w = w * image_width / 2`
   - `half_h = h * image_height / 2`
   - `x1 = int(cx_abs - half_w)`, `y1 = int(cy_abs - half_h)`
   - `x2 = int(cx_abs + half_w)`, `y2 = int(cy_abs + half_h)`
2. If `obj.segmentation`: convert each normalized point `(x, y)` to
   absolute pixel `(int(x * width), int(y * height))`
3. **Bbox from polygon fallback**: If polygon exists but bbox is None, compute
   bbox from polygon's axis-aligned bounds: `(min(xs), min(ys), max(xs), max(ys))`
4. Return `RenderData(annotations=[...], image_width, image_height)`

### 5.2 `COCOVisualizer`

**Constructor:** `COCOVisualizer(annotation_file, image_dir, log_config=None, **kwargs)`

- Creates `CocoAnnotationHandler` internally via `_create_handler()`
- `annotation_file` is the COCO JSON file path

**`_create_handler()`:**
```python
def _create_handler(self) -> CocoAnnotationHandler:
    return CocoAnnotationHandler(
        annotation_file=str(self.annotation_file),
        strict_mode=False,
        logger=self.logger,
    )
```

**`_convert_to_render_data(image_ann)`:**
For each `ObjectAnnotation` in `image_ann.objects`:
1. If `obj.bbox`: convert COCO-native (absolute pixel top-left `[x,y,w,h]`) to
   absolute pixel `[x1,y1,x2,y2]`:
   - `x1 = int(x)`, `y1 = int(y)`
   - `x2 = int(x + w)`, `y2 = int(y + h)`
2. If `obj.segmentation`: use absolute pixel polygon points as-is, truncate to int
3. If `obj.segmentation.rle`: preserve RLE dict as-is in `RenderAnnotation.rle`
4. **Bbox from polygon fallback**: If polygon exists but bbox is None, compute
   bbox from polygon's axis-aligned bounds: `(min(xs), min(ys), max(xs), max(ys))`
5. Return `RenderData(annotations=[...], image_width, image_height)`

### 5.3 `LabelMeVisualizer`

**Constructor:** `LabelMeVisualizer(label_dir, image_dir, class_file=None, log_config=None, **kwargs)`

- Creates `LabelMeAnnotationHandler` internally via `_create_handler()`
- `class_file` is optional

**`_create_handler()`:**
```python
def _create_handler(self) -> LabelMeAnnotationHandler:
    kwargs = dict(strict_mode=False, logger=self.logger)
    if self.class_file:
        kwargs["class_file"] = str(self.class_file)
    return LabelMeAnnotationHandler(
        label_dir=str(self.label_dir),
        **kwargs,
    )
```

**`_convert_to_render_data(image_ann)`:**
For each `ObjectAnnotation` in `image_ann.objects`:
1. If `obj.bbox`: LabelMe-native (absolute pixel top-left `[x,y,w,h]`) →
   `x1 = int(x)`, `y1 = int(y)`, `x2 = int(x + w)`, `y2 = int(y + h)`
2. If `obj.segmentation`: use absolute pixel polygon points as-is, truncate to int
3. **Bbox from polygon fallback**: If polygon exists but bbox is None, compute
   bbox from polygon's axis-aligned bounds: `(min(xs), min(ys), max(xs), max(ys))`
4. Return `RenderData(annotations=[...], image_width, image_height)`

## 6. Dependency Contract

```
Visualize module imports FROM:
├── dataflow.label.models         (DatasetAnnotations, AnnotationFormat, ...)
├── dataflow.label.yolo_handler   (YoloAnnotationHandler)
├── dataflow.label.coco_handler   (CocoAnnotationHandler)
├── dataflow.label.labelme_handler (LabelMeAnnotationHandler)
├── dataflow.util.logging         (LogConfig, LogManager)
├── cv2                           (OpenCV rendering)
└── numpy                         (Array operations)

Visualize module does NOT import FROM:
├── dataflow.convert.*            (FORBIDDEN — zero cross-dependency)
└── dataflow.cli.*                (FORBIDDEN — CLI depends on Visualize, not vice versa)
```

## 7. Error Handling Contract

Visualization is a **read-only** operation — no annotation data is produced. Errors are always
logged with the reason and the offending file/line, then processing continues to the next file.
This ensures a single bad annotation file never prevents the user from inspecting all other
images in the dataset.

| Error Type | Behavior |
|------------|----------|
| Annotation parse error (per-line) | Log warning with line number + reason, skip the line, continue parsing the same file |
| Annotation parse error (per-file, handler yields nothing) | Log warning, skip the file, continue to next |
| Image file not found | Log warning, skip image, continue (`failed_images++`) |
| Image failed to load (cv2.imread) | Log warning, skip image, continue (`failed_images++`) |
| Display window error | Log warning, continue without display |
| RLE decode failed (no pycocotools) | Log error, skip RLE mask drawing, continue |
| Handler construction error (missing directory, no categories) | `ValueError` raised before iteration begins — these are structural errors that make the entire dataset unusable |

**Key rules:**
- **Per-line parse errors**: Handle inside the handler — log a warning with the specific
  line number and reason, skip that line, continue with the next line in the same file.
- **Per-file errors**: If the entire file cannot be processed (no paired image, all lines
  invalid), log a warning and skip to the next file. Counted in `failed_images`.
- **Image loading errors**: Always downgraded to warnings. Individual image failures are
  counted in `summary_data["failed_images"]` but processing continues.
- **Structural errors**: Raised immediately before any images are processed — these indicate
  a fundamentally broken dataset (wrong paths, empty class file) that cannot produce useful
  visualizations.
- **Partial results**: When `iter_images()` raises `ValueError` mid-iteration, the visualizer
  catches it and returns a result with `processed_count` reflecting images successfully shown
  before the failure.

## 8. Logging Contract

See [`spec_logging.md`](spec_logging.md) for the full `LogManager` contract. Visualize-specific:

**Constructor**: `BaseVisualizer.__init__(..., log_config=None)`

- If `log_config` is None, a default `LogConfig(name=f"visualize.{class_name}")` is created
- The visualizer creates a `LogManager` from the config
- A child logger is created for progress output: `self._log_manager.child("progress")`

**Log pipeline** (all handled internally by the visualizer):

```
══════════════════════════════════════
Visualize: YOLO
  Labels:  yolo_labels/
  Images:  images/
  Display: yes
  Save:    yes → output/

── Progress ──
  001  image_001.jpg  (5 objects)  ✓
  002  image_002.jpg  (3 objects)  ✓
  ...

── Result ──
  Status:   ✓ Success
  Images:   500 / 500 (0 failed)
  Objects:  3,240
  Duration: 50.12s

  Log saved to: logs/visualize_yolo_20240613_100000.log
══════════════════════════════════════
```

**Log templates** are in `dataflow/visualize/log_templates.py`:
- `format_viz_header()` — header with paths, display/save modes
- `format_viz_progress()` — single-line progress (counter-based for streaming)
- `format_viz_result()` — final result block

When `verbose=True`:
- Console: header + progress (every 10 images, counter-based since total is unknown in streaming)
- File: DEBUG-level per-image details (color assignments, coordinate conversions, drawing operations)
- `log_path` is recorded in `VisualizationResult.log_path`

When `verbose=False`:
- Console: progress (every 10 images, INFO level) + result block
- No file output
- `log_path` is `None`

## 9. Change History

### v4 → v4.1: Remove FileOperations

| Aspect | v4 | v4.1 |
|--------|----|----|
| File I/O | `FileOperations` wrapper class | Inline stdlib `Path` calls |
| `dataflow.util` dependency | `FileOperations` | Removed |

### v3 → v4: Unified Logging via LogManager

| Aspect | v3 | v4 |
|--------|----|----|
| Constructor params | `verbose=False, logger=None, log_file_path=None` | `log_config: Optional[LogConfig] = None` |
| Logger creation | 3-way branch: external `log_file_path` → verboselogging → default | `LogManager(log_config)` — single unified class |
| `VisualizationResult.log_file_path` | Present | Renamed to `log_path` |
| Log templates | None (ad-hoc `self.logger.info(...)` calls) | `log_templates.py` — structured formatting functions |
| CLI interaction | CLI creates logger + passes to visualizer | CLI passes `LogConfig`, visualizer handles all logging |

### v4.2 → v4.3: Industry-Standard Label Positioning

| Aspect | v4.2 | v4.3 |
|--------|------|------|
| Label horizontal alignment | Top-center `(cx, y1-5)` | Top-left `(x1, y1-2)` — matches Ultralytics, Supervision, Detectron2 |
| Label background | Black `(0,0,0)` | Class color — visual association with bbox/polygon |
| Edge handling | Flip below bbox | Flip **inside** bbox at top-left — matches Ultralytics YOLOv8 fallback |
| Rationale | — | No major CV tool uses top-center or black background; left-aligned with class color is industry consensus |

### v4.1 → v4.2: Consistent Label Positioning

| Aspect | v4.1 | v4.2 |
|--------|------|------|
| Label position | Bbox top-left `(x1, y1-5)`; polygon label at first point | Bbox top-center `(cx, y1-5)` |
| Bbox from polygon | Not computed — polygon-only annotations had no bbox | Computed from polygon axis-aligned bounds in `_convert_to_render_data()` |
| Edge handling | Clamp text background to image bounds | Flip label below bbox when text extends above image top |
| Spec §4.3 | No per-annotation drawing flow documented | Explicit per-annotation flow with label positioning rules |
