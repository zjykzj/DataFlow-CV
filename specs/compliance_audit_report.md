# 合规性审计报告

> **审计范围**: DataFlow-CV 转换逻辑代码 vs `specs/` 目录格式定义
> **审计日期**: 2026-04-27
> **审计方法**: 逐条比对 specs/ 下的四份规范文档与 `dataflow/label/`、`dataflow/convert/` 中的源码实现

---

## 一、发现问题汇总

| 编号 | 严重级别 | 类别 | 涉及文件 | 简述 |
|------|---------|------|---------|------|
| B1 | **严重** | BUG | `models.py`, `coco_handler.py` | `BoundingBox.xywh_abs()` 中心/左上角坐标语义错误，导致 COCO bbox 输出错误 |
| D1 | **高** | 规范偏离 | `base.py` | `_validate_bbox()` 缺少 width>0/height>0 及边界越界等 6 项校验 |
| D2 | **高** | 规范偏离 | `coco_handler.py` | COCO polygon 单环最低 6 值（3 顶点）校验缺失 |
| D3 | **中** | 规范偏离 | `coco_handler.py` | `read()` 静默覆盖用户显式设置的 `output_rle` 参数 |
| D4 | **中** | 规范偏离 | `coco_handler.py` | `validate()` 缺少 bbox/segmentation/area/iscrowd 必填字段校验、及 referential integrity 校验 |
| D5 | **中** | 规范偏离 | `coco_handler.py` | `licenses` 数组完全不处理（既不读取也不保留） |
| D6 | **中** | 规范偏离 | `labelme_handler.py` | `validate()` 缺少 `imageData` 必填字段校验 |
| D7 | **低** | 规范偏离 | `labelme_handler.py` | `imageHeight`/`imageWidth` 缺失时未尝试从源图读取尺寸 |
| D8 | **低** | 规范偏离 | `base.py` | `AnnotationResult` 无 `warnings` 字段，导致写操作警告静默丢失 |
| D9 | **低** | 规范偏离 | `coco_handler.py` | bbox-only 路径无条件将 `iscrowd` 置 0，可能覆盖原始 crowd 标记 |
| D10 | **低** | 规范偏离 | `labelme_handler.py` | `shape_type` 做了 `.lower()` 大小写不敏感处理（偏离 spec 精确匹配定义） |

---

## 二、详细分析

### B1 [严重] `BoundingBox.xywh_abs()` 中心/左上角语义错误

**涉及文件:** `dataflow/label/models.py:49-56`, `dataflow/label/coco_handler.py:793`

**规范依据:**
- `spec_conversion_logic.md` §2.2: Internal → COCO bbox 公式为
  ```
  x_tl = x_center_norm * img_width - (width_norm * img_width) / 2
  y_tl = y_center_norm * img_height - (height_norm * img_height) / 2
  w_abs = width_norm * img_width
  h_abs = height_norm * img_height
  ```
- `spec_coco_format.md` §4.2: COCO bbox 定义为 `[x, y, width, height]`，其中 `(x, y)` 为**左上角(top-left corner)**绝对像素坐标。

**现状代码:**
```python
# models.py:49-56
def xywh_abs(self, img_width: int, img_height: int) -> Tuple[int, int, int, int]:
    """Convert to absolute pixel coordinates."""
    return (
        int(self.x * img_width),       # ← 输出 x_center 像素值, 而非 x_tl
        int(self.y * img_height),      # ← 输出 y_center 像素值, 而非 y_tl
        int(self.width * img_width),
        int(self.height * img_height),
    )
```

**调用位置:**
```python
# coco_handler.py:793
x_abs, y_abs, w_abs, h_abs = obj.bbox.xywh_abs(img.width, img.height)
bbox = [float(x_abs), float(y_abs), float(w_abs), float(h_abs)]
```

**问题分析:**
由于内部数据模型 `BoundingBox.x` / `BoundingBox.y` 代表**中心点**(center)归一化坐标，`xywh_abs()` 仅做乘以图像尺寸的缩放，输出的 `(x_abs, y_abs)` 仍然是中心点坐标而非左上角坐标。这导致 YOLO→COCO、LabelMe→COCO 转换时生成的 COCO bbox 位置偏移了半个宽高。

**影响范围:** 所有非 original-data 路径的 COCO 写入（即 Priority 2 分支）。当源格式为 YOLO 或 LabelMe，目标为 COCO 时，非 crowd 且非 RLE 的标注会受到影响。Priority 1 (original data) 路径不受影响。

**修复方向:** `xywh_abs` 应返回 `(x_tl, y_tl, width, height)`，或新增一个方法（如 `xy_topleft_abs`）专门用于 COCO 左上角语义，并在 COCO handler 的正确路径中使用。

---

### D1 [高] `_validate_bbox()` 校验不完整

**涉及文件:** `dataflow/label/base.py:132-143`

**规范依据:**
- `spec_conversion_logic.md` §7.2 定义完整的 bbox 校验规则:
  ```
  valid = (0 <= x <= 1) AND (0 <= y <= 1)
          AND (width > 0) AND (height > 0)
          AND (x - width/2 >= 0) AND (x + width/2 <= 1)
          AND (y - height/2 >= 0) AND (y + height/2 <= 1)
  ```
- `spec_yolo_format.md` §2.3: "Any coordinate value falls outside the closed interval [0.0, 1.0]"以及 "The resulting bounding box has zero or negative area (width <= 0 or height <= 0)"

**现状代码:**
```python
# base.py:132-143
def _validate_bbox(self, bbox) -> bool:
    if bbox is None:
        return True
    checks = [
        self._validate_normalized_coordinate(bbox.x, "bbox.x"),      # ✓ 0≤x≤1
        self._validate_normalized_coordinate(bbox.y, "bbox.y"),      # ✓ 0≤y≤1
        self._validate_normalized_coordinate(bbox.width, "bbox.width"),  # ✓ 0≤w≤1
        self._validate_normalized_coordinate(bbox.height, "bbox.height"), # ✓ 0≤h≤1
    ]
    return all(checks)
```

**缺失的校验项 (共 6 项):**

| # | 缺失校验 | 规范条款 |
|---|---------|---------|
| 1 | `width > 0` (零面积拒绝) | `spec_conversion_logic.md` §7.2, `spec_yolo_format.md` §2.3 |
| 2 | `height > 0` (零面积拒绝) | 同上 |
| 3 | `x - width/2 >= 0` (左边界越界检测) | `spec_conversion_logic.md` §7.2 |
| 4 | `x + width/2 <= 1` (右边界越界检测) | 同上 |
| 5 | `y - height/2 >= 0` (上边界越界检测) | 同上 |
| 6 | `y + height/2 <= 1` (下边界越界检测) | 同上 |

**问题分析:**
`_validate_normalized_coordinate` 只校验 `0 ≤ v ≤ 1`，但 width=0 或 height=0 的无效 bbox 能通过校验（因为 0 在 [0,1] 范围内）。此外，超出图像边界的 bbox（例如 x=0.9, width=0.5 时，右边界 x+width/2=1.15>1）也无法被检测到。

**影响范围:** 所有格式的读取路径均使用 `_validate_bbox()`。

---

### D2 [高] COCO polygon 单环最低顶点数校验缺失

**涉及文件:** `dataflow/label/coco_handler.py:439-460`

**规范依据:**
- `spec_coco_format.md` §6.1: "Each inner array MUST have at least 6 values (3 coordinate pairs). An array with fewer than 6 values describes a degenerate polygon and MUST be rejected."

**现状代码:**
```python
# coco_handler.py:447-452
for polygon in seg_data:
    if len(polygon) % 2 != 0:
        self._log_warning(...)
        continue
    # 没有检查 len(polygon) >= 6 !
    for i in range(0, len(polygon), 2):
        ...
```

**问题分析:**
仅校验了坐标数为偶数，未校验最低顶点数。`[[100, 80]]` (2个值=1个顶点) 或 `[[100, 80, 200, 160]]` (4个值=2个顶点) 的退化 polygon 会通过校验、被当作有效 segmentation 处理。

---

### D3 [中] `read()` 静默覆盖用户显式 `output_rle` 参数

**涉及文件:** `dataflow/label/coco_handler.py:110`

**规范依据:**
- `spec_coco_format.md` §6.2 表格: 当 `iscrowd=0` 时 "Polygon format is preferred. RLE is used only when output_rle=True." 即用户显式设置的 `output_rle` 值应被尊重。

**现状代码:**
```python
# coco_handler.py:109-110 (在 read() 方法内)
self.is_rle = self._detect_rle_format(self.annotations)
self.output_rle = self.is_rle  # ← 强制覆盖!
```

**问题分析:**
当用户创建 COCO handler 时显式传入 `do_rle=False`（意即"不要输出 RLE"），但如果源 COCO 文件包含 RLE 格式 annotation，则 `read()` 后 `self.output_rle` 被覆盖为 `True`，后续 `write()` 会输出 RLE。这违背了用户的显式意图。

同样，当用户设置了 `do_rle=True` 但输入没有 RLE，则被覆盖为 `False`，也不符合用户意图。

---

### D4 [中] COCO `validate()` 校验不完整

**涉及文件:** `dataflow/label/coco_handler.py:826-864`

**规范依据:**
- `spec_coco_format.md` §4.1: 每个 annotation 对象需包含 `id`, `image_id`, `category_id`, `bbox`, `segmentation`, `area`, `iscrowd`。
- §4.4 规定了 referential integrity: `image_id` 必须引用 `images[].id`，`category_id` 必须引用 `categories[].id`。

**现状代码 (validate 方法只检查了 3 个字段):**
```python
# coco_handler.py:852-855
for ann in data["annotations"]:
    if "id" not in ann or "image_id" not in ann or "category_id" not in ann:
        self.logger.error(...)
        return False
```

**缺失的校验项:**
- `bbox` / `segmentation` / `area` / `iscrowd` 必填字段存在性检查
- `image_id` → `images[].id` referential integrity
- `category_id` → `categories[].id` referential integrity
- 对 `images` 中 `width` / `height` 必填字段的存在性检查

---

### D5 [中] COCO `licenses` 数组完全不处理

**涉及文件:** `dataflow/label/coco_handler.py`

**规范依据:**
- `spec_coco_format.md` §1: 顶层结构应包含 `licenses` 数组。§2.2 定义了 License 对象结构。
- §8.1 定义了保留策略，包括 full `images[]` entries (含 `license` 引用)。

**现状代码:**
- `read()`: 没有读取/保留 `licenses` 字段（`dataset_info` 提取时仅排除三大数组，`licenses` 进了 `dataset_info` 但未结构化保留）
- `write()`: `_prepare_coco_data()` 中没有生成或恢复 `licenses` 数组
- `_load_images()`: 未提取 `license` 字段

**问题分析:**
COCO JSON 文件若包含 `licenses` 数组，回写后该数组丢失。虽然 spec 标注 `licenses` 为 Optional，但 round-trip 时应保留已有数据。

---

### D6 [中] LabelMe `validate()` 缺少 `imageData` 字段校验

**涉及文件:** `dataflow/label/labelme_handler.py:509`

**规范依据:**
- `spec_labelme_format.md` §2: `imageData` 为 **Required** 字段（值为 string 或 null）。

**现状代码:**
```python
# labelme_handler.py:509 (validate 方法)
required_fields = ["version", "flags", "shapes", "imagePath"]
# 缺少 "imageData" !
```

**对比 `_read_single_file` 中的校验:**
```python
# labelme_handler.py:157 (read 方法)
required_fields = ["version", "flags", "shapes", "imagePath", "imageData"]
```

**问题分析:**
`validate()` 和 `_read_single_file()` 的必填字段列表不一致，validate 缺少 `imageData`。

---

### D7 [低] LabelMe 缺尺寸时未尝试从源图读取

**涉及文件:** `dataflow/label/labelme_handler.py:177-182`

**规范依据:**
- `spec_labelme_format.md` §2.2: "If absent, the consumer MUST attempt to determine dimensions from the source image."

**现状代码:**
```python
# labelme_handler.py:177-182
if image_height is None or image_width is None:
    self._log_warning(f"Image dimensions not in JSON {json_file}, using defaults")
    image_height = 1
    image_width = 1
```

**问题分析:**
当 `imageHeight`/`imageWidth` 缺失时，代码直接使用默认值 `(1, 1)`，没有尝试从 `imagePath` 指向的实际图像文件中读取尺寸。spec 明确要求 MUST attempt。

---

### D8 [低] `AnnotationResult` 缺少 `warnings` 字段

**涉及文件:** `dataflow/label/base.py:15-42`

**规范依据:**
- 非 spec 偏离，属设计缺陷。Converter 代码引用 `write_result.warnings` 但该属性不存在。

**现状代码:**
```python
# base.py:15 — AnnotationResult 无 warnings 字段
@dataclass
class AnnotationResult:
    success: bool
    data: Optional[Any] = None
    message: str = ""
    errors: List[str] = field(default_factory=list)
    # 缺少: warnings: List[str] = field(default_factory=list)
```

**对比:** `ConversionResult` (convert/base.py:19) 同时具有 `errors` 和 `warnings`。

**问题分析:**
handler 层产生的警告信息（如 RLE 解码失败、polygon 校验警告等）在返回到 converter 层时被丢弃，`write_result.warnings` 始终为空列表。

---

### D9 [低] COCO bbox-only 路径无条件重置 `iscrowd`

**涉及文件:** `dataflow/label/coco_handler.py:778-782`

**规范依据:**
- `spec_coco_format.md` §4.3: `iscrowd` 是 COCO annotation 的独立属性，与是否有 segmentation 无关。

**现状代码:**
```python
# coco_handler.py:778-782
elif obj.bbox:
    segmentation = []
    iscrowd = 0  # ← 无条件置 0，忽略 obj.is_crowd
```

**问题分析:**
当 annotation 仅有 bbox 而无 segmentation 时，代码强制将 `iscrowd` 设为 0。但如果原始 COCO 数据中存在 `iscrowd=1` 且仅含 bbox 的标注（虽然不常见，但 spec 允许），此信息会丢失。

---

### D10 [低] LabelMe `shape_type` 大小写不敏感处理

**涉及文件:** `dataflow/label/labelme_handler.py:258`

**规范依据:**
- `spec_labelme_format.md` §3.1: 支持的 `shape_type` 值为 `"rectangle"` 和 `"polygon"`（精确小写）。

**现状代码:**
```python
# labelme_handler.py:258
shape_type = shape.get("shape_type", "").lower()
```

**问题分析:**
代码对 `shape_type` 做了 `.lower()` 处理，这意味着 `"Rectangle"`、`"POLYGON"` 等非规范值也能被接受。虽然这提高了容错性，但从严格合规角度看，偏离了 spec 定义的精确值匹配原则。风险较低，实际场景中这可能是合理的设计选择。

---

## 三、合规性总结

### 严重问题 (必须修复)

| 编号 | 简述 | 建议优先级 |
|------|------|-----------|
| B1 | COCO bbox 输出中心/左上角坐标错误 | P0 |

此 bug 导致所有非 original-data 路径的 COCO 写入产生系统性坐标偏移。修复 `BoundingBox.xywh_abs()` 的语义或调用方式即可解决。

### 高优先级偏离 (建议修复)

| 编号 | 简述 |
|------|------|
| D1 | `_validate_bbox()` 缺少 6 项校验（width>0, height>0, 边界越界） |
| D2 | COCO polygon 单环最低 3 顶点(6 值)校验缺失 |

这两项会导致无效或退化的标注数据被当作有效数据接受，影响数据质量。

### 中优先级偏离 (建议改进)

| 编号 | 简述 |
|------|------|
| D3 | `read()` 覆盖用户 `output_rle` 设置 |
| D4 | COCO `validate()` 缺少多项必填字段和引用完整性校验 |
| D5 | `licenses` 数组未保留 |
| D6 | LabelMe `validate()` 缺少 `imageData` 校验 |

### 低优先级偏离 (可选改进)

| 编号 | 简述 |
|------|------|
| D7 | LabelMe 缺尺寸时未从源图读取 |
| D8 | `AnnotationResult` 缺少 `warnings` 字段 |
| D9 | bbox-only 路径 iscrowd 无条件置 0 |
| D10 | shape_type 大小写不敏感 |

---

## 四、合规项确认 (以下实现与 spec 一致)

- ✅ YOLO 空行忽略、token 数量检测 (detection=5 / segmentation>5且奇数)
- ✅ YOLO 6 位小数精度 (`.6f`)
- ✅ YOLO segmentation 最低 3 点校验
- ✅ COCO RLE 检测 (`isinstance(seg, dict) and "counts" in seg`)
- ✅ COCO RLE encode/decode 流程（Fortran-order, bytes↔string 转换）
- ✅ COCO crowd annotation 的 RLE 强制写入逻辑
- ✅ COCO original data 保留及 Priority 1 回写机制
- ✅ LabelMe rectangle 任意角点顺序处理 (`abs()`)
- ✅ LabelMe polygon 最低 3 点校验
- ✅ LabelMe `imageData=null` 写入
- ✅ LabelMe original data 保留（`raw_data=shape.copy()`）
- ✅ YOLO original data 保留（line + items + is_detection/is_segmentation）
- ✅ `classes.txt` 加载/生成（0-based 顺序，空行跳过）
- ✅ 图像错误始终以 warning 处理（不因 strict_mode 中断）
- ✅ `pycocotools` 可选依赖的优雅降级
