# Visualize Module Specification

> **Version:** 2.0
> **Status:** Draft — updated for unified rendering format
> **Layer:** Modules
> **Dependencies:** Label module (handlers + models)

## 1. Module Overview

The Visualize module (`dataflow/visualize/`) renders annotation data onto images for visual
inspection. It depends **only** on the Label module — it does not import from Convert or CLI.

### 1.1 Key Design: Unified Rendering Format

Visualizers accept `DatasetAnnotations` in **any format** and convert all coordinates to a
**unified intermediate rendering format** before drawing. This means:

- One rendering pipeline handles all source formats
- Coordinate conversion to absolute pixels happens once, in one place
- Visualizers don't need to know the source format

### 1.2 Module Contract

- **Input**: `DatasetAnnotations` (any format) + image files
- **Processing**: Convert all annotations to absolute-pixel render data → draw → display/save
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

The conversion from `DatasetAnnotations` (any format) → `RenderData`:

| Source Format | Bounding Box Conversion | Polygon Conversion |
|---------------|------------------------|-------------------|
| YOLO | `cx_norm→cx_abs, cy_norm→cy_abs` → `int(cx-w/2), int(cy-h/2), int(cx+w/2), int(cy+h/2)` | `int(x*W), int(y*H)` per point |
| COCO | `x_tl, y_tl, w, h` → `x_tl, y_tl, x_tl+w, y_tl+h` | Use absolute pixel points as-is |
| LabelMe | `x_tl, y_tl, w, h` → `x_tl, y_tl, x_tl+w, y_tl+h` | Use absolute pixel points as-is |

**All coordinates are truncated to integers via `int()` for OpenCV drawing compatibility.**

This conversion is lossy (integer truncation), but since it's used only for display (not for
saving annotations), precision loss in the rendering pipeline is acceptable.

### 2.4 Conversion Logic Location

Coordinate conversion to `RenderData` is done in each concrete visualizer's
`load_annotations()` method. This method:
1. Calls the Label handler's `read()` to get `DatasetAnnotations` in native format
2. Converts all annotations to `List[RenderAnnotation]` using the conversion rules above
3. Stores the `RenderData` for the rendering pipeline

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
| `log_file_path` | Optional[str] | Log file path (verbose mode) |

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
| `strict_mode` | bool | No | True | Strict validation mode |
| `verbose` | bool | No | False | Verbose file logging |
| `logger` | Optional[Logger] | No | None | Logger instance |
| `log_file_path` | Optional[str] | No | None | Pre-configured log file path |

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
├── 1. load_annotations()          # Abstract — subclass returns List[RenderAnnotation]
├── 2. Validate output_dir         # Required if is_save=True
├── 3. For each RenderAnnotation + image:
│   ├── _visualize_single_image()
│   │   ├── Load image (resolve path, cv2.imread)
│   │   ├── For each RenderAnnotation:
│   │   │   ├── _draw_bbox()       # If bbox exists
│   │   │   ├── _draw_polygon()    # If polygon exists
│   │   │   └── _draw_rle_mask()   # If RLE exists
│   │   ├── Display (cv2.imshow)   # If is_show=True
│   │   └── Save (cv2.imwrite)     # If is_save=True
│   └── Handle keyboard input
└── 4. Return VisualizationResult
```

### 4.2 Abstract Method: `load_annotations()`

Each concrete visualizer must implement:

```python
@abstractmethod
def load_annotations(self) -> Dict[str, RenderAnnotations]: ...
```

The implementation:
1. Creates the appropriate Label handler
2. Calls `handler.read()` → `DatasetAnnotations` (format-native coordinates)
3. Converts all annotations to absolute-pixel `RenderAnnotation` objects
4. Returns mapping of `{image_path: [RenderAnnotation, ...]}`
5. Raises `ValueError` on failure

### 4.3 Drawing Methods

#### `_draw_bbox(image, bbox, color, class_name)`

- `bbox`: `(x1, y1, x2, y2)` in absolute pixels (from `RenderAnnotation` — already in absolute coords)
- Draw rectangle with `cv2.rectangle()`
- Draw class label above the rectangle with `_draw_text()`

#### `_draw_polygon(image, polygon, color, class_name)`

- `polygon`: `[(x1, y1), (x2, y2), ...]` in absolute pixels (from `RenderAnnotation`)
- Draw semi-transparent fill with `cv2.fillPoly()` + `cv2.addWeighted()` (alpha from config)
- Draw polygon outline with `cv2.polylines()`
- Draw class label near first point

#### `_draw_rle_mask(image, rle, color)`

1. Decode RLE to binary mask via pycocotools (`coco_mask.decode()`)
2. Create color mask × BGR color values
3. Semi-transparent overlay with `cv2.addWeighted()` + `np.copyto()`

Requires pycocotools — logs error and returns without drawing if unavailable.

#### `_draw_text(image, text, position, color)`

1. Calculate text bounding box
2. Draw black background rectangle (clamped to image boundaries)
3. Draw white text with `cv2.putText()` (anti-aliased)

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
with a text progress bar:

```
[==========>...............................] 25.0% Processing image_025
```

## 5. Concrete Visualizers

### 5.1 `YOLOVisualizer`

**Constructor:** `YOLOVisualizer(label_dir, image_dir, class_file, verbose=False, **kwargs)`

- Creates `YoloAnnotationHandler` internally
- `class_file` is required (passed to handler)
- `load_annotations()`:
  1. Calls `YoloAnnotationHandler.read()` → `DatasetAnnotations(format=YOLO)`
  2. For each `ObjectAnnotation`: converts YOLO-native (normalized center) coords
     to `RenderAnnotation` (absolute pixel `[x1,y1,x2,y2]`)

### 5.2 `COCOVisualizer`

**Constructor:** `COCOVisualizer(annotation_file, image_dir, verbose=False, **kwargs)`

- Creates `CocoAnnotationHandler` internally
- `annotation_file` is the COCO JSON file path
- `load_annotations()`:
  1. Calls `CocoAnnotationHandler.read()` → `DatasetAnnotations(format=COCO)`
  2. For each `ObjectAnnotation`: converts COCO-native (absolute pixel `[x,y,w,h]`)
     to `RenderAnnotation` (absolute pixel `[x1,y1,x2,y2]`)

### 5.3 `LabelMeVisualizer`

**Constructor:** `LabelMeVisualizer(label_dir, image_dir, class_file=None, verbose=False, **kwargs)`

- Creates `LabelMeAnnotationHandler` internally
- `class_file` is optional
- `load_annotations()`:
  1. Calls `LabelMeAnnotationHandler.read()` → `DatasetAnnotations(format=LABELME)`
  2. For each `ObjectAnnotation`: LabelMe coordinates are already in the right form;
     directly converts to `RenderAnnotation`

## 6. Dependency Contract

```
Visualize module imports FROM:
├── dataflow.label.models         (DatasetAnnotations, AnnotationFormat, ...)
├── dataflow.label.yolo_handler   (YoloAnnotationHandler)
├── dataflow.label.coco_handler   (CocoAnnotationHandler)
├── dataflow.label.labelme_handler (LabelMeAnnotationHandler)
├── dataflow.util                 (FileOperations)
├── cv2                           (OpenCV rendering)
└── numpy                         (Array operations)

Visualize module does NOT import FROM:
├── dataflow.convert.*            (FORBIDDEN — zero cross-dependency)
└── dataflow.cli.*                (FORBIDDEN — CLI depends on Visualize, not vice versa)
```

## 7. Error Handling Contract

| Error Type | Strict Mode | Non-Strict Mode |
|------------|-------------|-----------------|
| Image file not found | Log warning, skip image, continue | Log warning, skip image, continue |
| Image failed to load (cv2.imread) | Log warning, skip image, continue | Log warning, skip image, continue |
| Display window error | Log warning, continue without display | Log warning, continue without display |
| RLE decode failed (no pycocotools) | Log error, skip RLE mask drawing | Log error, skip RLE mask drawing |
| Annotation load failed | `ValueError` raised (caught by `visualize()`) | `ValueError` raised |

**Key rule**: Image loading errors never abort the entire visualization — individual image
failures are counted in `summary_data["failed_images"]` but processing continues.

## 8. Verbose Logging Contract

When `verbose=True`:
- Each image processing step is logged at DEBUG level
- Color assignments are logged (class_id → BGR)
- Drawing operations include coordinate verification logs
- Summary statistics are logged at completion (total, success, failed, success rate, duration)
- Log file path is recorded in `VisualizationResult.log_file_path`

When `verbose=False`:
- Only INFO-level console output
- No file logging
