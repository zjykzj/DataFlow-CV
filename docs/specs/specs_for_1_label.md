# DataFlow-CV 标注处理模块开发文档 (label)

## 1. 模块概述

### 功能定位
`dataflow/label` 模块是 DataFlow-CV 项目的核心数据层，负责计算机视觉标注数据的统一表示和格式转换。它提供了 YOLO、LabelMe 和 COCO 三种主流标注格式的读写支持，以及统一的数据模型和坐标系统。

### 核心特性
- **统一数据模型**：`DatasetAnnotations`、`ImageAnnotation`、`ObjectAnnotation` 等核心数据类
- **多格式支持**：YOLO (TXT)、LabelMe (JSON)、COCO (JSON) 格式的读写
- **无损处理**：通过 `OriginalData` 机制支持无损往返转换
- **坐标系统一**：所有坐标归一化为 0-1 范围，支持与绝对像素坐标转换
- **RLE 支持**：COCO 格式的 Run-Length Encoding 编解码（依赖 pycocotools）
- **严格验证**：格式验证、坐标范围检查、数据完整性验证

### 设计原则
1. **数据一致性**：统一的数据模型表示不同格式的标注数据
2. **无损处理**：原始数据保存和优先使用，确保往返转换无精度损失
3. **格式透明**：用户无需关心底层格式差异，统一接口操作
4. **错误容忍**：支持严格模式和非严格模式，灵活处理错误

## 2. 核心架构

### 类图/组件图
```
┌─────────────────────────────────────────┐
│           dataflow.label                │
├─────────────────────────────────────────┤
│  • AnnotationFormat (枚举)              │
│    - LABELME, YOLO, COCO, UNKNOWN       │
│                                         │
│  • 数据模型层                           │
│    - DatasetAnnotations                 │
│    - ImageAnnotation                    │
│    - ObjectAnnotation                   │
│    - BoundingBox, Segmentation          │
│    - OriginalData (无损处理核心)        │
│                                         │
│  • 处理器接口层                         │
│    - BaseAnnotationHandler (抽象基类)   │
│      - read(), write(), validate()      │
│                                         │
│  • 具体处理器实现                       │
│    - YoloAnnotationHandler              │
│    - LabelMeAnnotationHandler           │
│    - CocoAnnotationHandler              │
└─────────────────────────────────────────┘
```

### 数据流
1. **读取流程**：
   ```
   标注文件 → 解析原始数据 → 转换为统一数据模型 → 保存 OriginalData → DatasetAnnotations
               ↳ 格式验证         ↳ 坐标归一化(0-1)
   ```

2. **写入流程**：
   ```
   DatasetAnnotations → 检查 OriginalData → 优先使用原始数据 → 坐标反归一化 → 生成目标格式
                         ↳ 格式匹配检查       ↳ 或使用转换数据
   ```

3. **无损处理机制**：
   ```
   读取: 保存原始数据到 OriginalData → 写入: 检查格式匹配 → 使用原始数据输出
           (优先级最高，确保无损)
   ```

### 依赖关系
- **内部依赖**：`dataflow/util`（文件操作和日志）
- **外部依赖**：
  - 必需：`numpy`, `opencv-python`
  - 可选：`pycocotools`（COCO RLE 支持）
- **被依赖**：`dataflow/convert`（转换模块）, `dataflow/visualize`（可视化模块）

## 3. 主要API

### 数据模型类

#### AnnotationFormat 枚举
**值**：`LABELME`, `YOLO`, `COCO`, `UNKNOWN`

#### DatasetAnnotations 类
**职责**：整个数据集的标注容器

**核心属性**：
- `images: List[ImageAnnotation]`：所有图片的标注列表
- `categories: Dict[int, str]`：类别ID到名称的映射
- `dataset_info: Dict[str, Any]`：数据集元信息

#### ImageAnnotation 类
**职责**：单张图片的标注容器

**核心属性**：
- `image_id: str`：图片唯一标识
- `image_path: Path`：图片文件路径
- `width: int`, `height: int`：图片尺寸
- `objects: List[ObjectAnnotation]`：图片中的对象列表

#### ObjectAnnotation 类
**职责**：单个对象的标注信息

**核心属性**：
- `class_id: int`：类别ID
- `class_name: str`：类别名称
- `bbox: Optional[BoundingBox]`：边界框（检测任务）
- `segmentation: Optional[Segmentation]`：分割多边形（分割任务）
- `confidence: Optional[float]`：置信度分数
- `original_data: Optional[OriginalData]`：原始数据（无损处理关键）

#### OriginalData 类
**职责**：保存原始标注数据，支持无损往返

**核心属性**：
- `format: AnnotationFormat`：原始格式
- `data: Dict[str, Any]`：原始数据字典
- `rle_data: Optional[Dict]`：RLE编码数据（COCO格式）

### 处理器接口

#### BaseAnnotationHandler 抽象基类
**职责**：定义所有标注处理器的统一接口

**核心方法**：
- `read(label_dir: Path, **kwargs) -> AnnotationResult`：读取标注文件
- `write(dataset: DatasetAnnotations, output_path: Path, **kwargs) -> AnnotationResult`：写入标注文件
- `validate(input_path: Path, **kwargs) -> AnnotationResult`：验证标注文件格式

**抽象方法**（子类必须实现）：
- `_read_implementation()`：具体格式的读取实现
- `_write_implementation()`：具体格式的写入实现

#### YoloAnnotationHandler 类
**职责**：处理 YOLO TXT 格式标注

**特性**：
- 支持目标检测（5个值：`class_id x_center y_center width height`）
- 支持实例分割（`class_id x1 y1 x2 y2 ...`）
- 需要类别文件、标签目录和图片目录

#### LabelMeAnnotationHandler 类
**职责**：处理 LabelMe JSON 格式标注

**特性**：
- 支持矩形（rectangle）和多边形（polygon）形状
- 类别文件可选，可从标注中提取类别
- 每个标注保存为单独的 JSON 文件

#### CocoAnnotationHandler 类
**职责**：处理 COCO JSON 格式标注

**特性**：
- 支持多边形和 RLE 分割格式
- 可选依赖 pycocotools 进行 RLE 编解码
- 支持 crowd 标注（iscrowd 标志）
- 单个 JSON 文件包含整个数据集的标注

### 配置参数

**通用参数**：
- `strict_mode: bool = True`：严格模式（错误时抛出异常）
- `verbose: bool = False`：详细日志模式

**处理器特定参数**：
- `class_file: Optional[Path]`：类别文件路径（YOLO、LabelMe 需要）
- `image_dir: Optional[Path]`：图片目录路径
- `use_rle: bool = False`：是否使用 RLE 格式（COCO）

## 4. 使用指南

### 快速开始

```python
# 1. YOLO 格式读取
from dataflow.label import YoloAnnotationHandler

yolo_handler = YoloAnnotationHandler()
result = yolo_handler.read(
    label_dir=Path("yolo_labels"),
    image_dir=Path("images"),
    class_file=Path("classes.txt")
)

if result.success:
    dataset = result.data
    print(f"Loaded {len(dataset.images)} images")

# 2. LabelMe 格式写入
from dataflow.label import LabelMeAnnotationHandler

labelme_handler = LabelMeAnnotationHandler()
write_result = labelme_handler.write(
    dataset=dataset,
    output_path=Path("labelme_output")
)

# 3. COCO 格式处理（带 RLE）
from dataflow.label import CocoAnnotationHandler

coco_handler = CocoAnnotationHandler(use_rle=True)
coco_result = coco_handler.read(Path("annotations.json"))
```

### 常见场景

#### 场景1：格式转换基础
```python
# 读取 YOLO 格式
yolo_handler = YoloAnnotationHandler()
yolo_result = yolo_handler.read(Path("yolo_data"), ...)

# 转换为 LabelMe 格式
if yolo_result.success:
    labelme_handler = LabelMeAnnotationHandler()
    labelme_handler.write(yolo_result.data, Path("labelme_output"))
```

#### 场景2：无损往返验证
```python
# 读取原始数据
handler = LabelMeAnnotationHandler()
result1 = handler.read(Path("original_labelme"))

if result1.success:
    # 写回同一格式
    result2 = handler.write(result1.data, Path("restored_labelme"))

    # 验证文件是否完全相同
    # （由于 OriginalData 机制，应该完全一致）
```

#### 场景3：混合数据处理
```python
# 创建新的数据集
from dataflow.label.models import DatasetAnnotations, ImageAnnotation, ObjectAnnotation

dataset = DatasetAnnotations()
image = ImageAnnotation(
    image_id="001",
    image_path=Path("images/001.jpg"),
    width=640,
    height=480
)

# 添加对象（可以是来自不同格式的混合数据）
obj = ObjectAnnotation(
    class_id=0,
    class_name="person",
    bbox=BoundingBox(...),
    # original_data 可包含原始格式信息
)

image.objects.append(obj)
dataset.images.append(image)
```

#### 场景4：坐标系统转换
```python
# 归一化坐标 → 绝对像素坐标
from dataflow.label.models import BoundingBox

bbox = BoundingBox(x_center=0.5, y_center=0.5, width=0.2, height=0.3)
abs_bbox = bbox.to_absolute(640, 480)  # 图片尺寸
# 结果: (x_center=320, y_center=240, width=128, height=144)

# 绝对像素坐标 → 归一化坐标
norm_bbox = BoundingBox.from_absolute(320, 240, 128, 144, 640, 480)
# 结果: (x_center=0.5, y_center=0.5, width=0.2, height=0.3)
```

## 5. 开发步骤

### 阶段1：设计核心数据模型
1. **分析格式需求**：研究 YOLO、LabelMe、COCO 格式的数据结构
2. **设计统一数据模型**：
   - `DatasetAnnotations`：数据集级容器
   - `ImageAnnotation`：图片级容器
   - `ObjectAnnotation`：对象级容器
   - `BoundingBox`、`Segmentation`：几何数据
3. **设计坐标系统**：
   - 确定归一化坐标（0-1 范围）作为内部标准
   - 设计坐标转换方法（归一化 ↔ 绝对像素）
4. **设计 `OriginalData` 机制**：
   - 定义保存原始数据的结构
   - 设计优先级写入策略

### 阶段2：实现 BaseAnnotationHandler 基类
1. **定义抽象接口**：
   ```python
   class BaseAnnotationHandler(ABC):
       @abstractmethod
       def read(self, label_dir: Path, **kwargs) -> AnnotationResult:
           pass

       @abstractmethod
       def write(self, dataset: DatasetAnnotations,
                output_path: Path, **kwargs) -> AnnotationResult:
           pass
   ```
2. **实现通用功能**：
   - 参数验证和预处理
   - 错误处理和结果封装
   - 日志记录集成
3. **设计 `AnnotationResult` 类**：
   - 统一的结果返回格式
   - 支持成功/失败状态、消息、错误列表

### 阶段3：实现 YoloAnnotationHandler
1. **解析 YOLO 格式**：
   - 目标检测格式：`class_id x_center y_center width height`
   - 实例分割格式：`class_id x1 y1 x2 y2 ...`
2. **实现读取逻辑**：
   ```python
   def _read_implementation(self, label_dir, image_dir, class_file):
       # 1. 加载类别文件
       categories = self._load_categories(class_file)

       # 2. 遍历标签文件
       for label_file in label_dir.glob("*.txt"):
           # 3. 解析每行数据
           with open(label_file) as f:
               for line in f:
                   parts = line.strip().split()
                   if len(parts) == 5:
                       # 检测任务
                       self._parse_detection_line(parts)
                   else:
                       # 分割任务
                       self._parse_segmentation_line(parts)
   ```
3. **实现写入逻辑**：
   - 将归一化坐标转换为 YOLO 格式坐标
   - 生成 TXT 文件（每行一个对象）

### 阶段4：实现 LabelMeAnnotationHandler
1. **解析 LabelMe JSON 格式**：
   - 解析形状数据（矩形、多边形）
   - 提取类别标签和坐标
2. **实现读取逻辑**：
   - 遍历 JSON 文件
   - 解析每个形状的 points 和 label
   - 转换为统一数据模型
3. **实现写入逻辑**：
   - 生成符合 LabelMe 格式的 JSON
   - 包含图片信息、形状列表、版本信息

### 阶段5：实现 CocoAnnotationHandler
1. **解析 COCO JSON 格式**：
   - 解析 categories、images、annotations 结构
   - 支持多边形和 RLE 格式
2. **实现 RLE 支持**：
   ```python
   if self.use_rle and HAS_PYCOCOTOOLS:
       # 使用 pycocotools 进行 RLE 编解码
       import pycocotools.mask as mask_util
       rle = mask_util.encode(np.array(mask))
   else:
       # 降级为多边形格式
       self._logger.warning("RLE not available, using polygon format")
   ```
3. **实现读取/写入逻辑**：
   - 处理 COCO 的特殊字段（iscrowd、area 等）
   - 支持从图片文件名匹配标注

### 阶段6：实现无损处理机制
1. **完善 `OriginalData` 类**：
   - 设计序列化/反序列化方法
   - 支持多种数据格式保存
2. **实现优先级写入策略**：
   ```python
   def _get_write_data(self, obj: ObjectAnnotation, target_format):
       # 1. 检查是否有匹配的原始数据
       if (obj.original_data and
           obj.original_data.format == target_format):
           return obj.original_data.data

       # 2. 检查是否有 RLE 数据（COCO 格式）
       if (target_format == AnnotationFormat.COCO and
           obj.segmentation and obj.segmentation.rle_data):
           return obj.segmentation.rle_data

       # 3. 使用转换后的数据
       return self._convert_data(obj, target_format)
   ```
3. **集成到各个处理器**：
   - 读取时保存原始数据
   - 写入时优先使用原始数据

### 阶段7：集成验证和错误处理
1. **实现验证方法**：
   - 格式验证（文件结构、字段完整性）
   - 坐标验证（范围检查、有效性检查）
   - 数据一致性验证
2. **实现严格/非严格模式**：
   ```python
   if self.strict_mode:
       raise AnnotationError(f"Invalid coordinate: {coord}")
   else:
       self._logger.warning(f"Ignoring invalid coordinate: {coord}")
       return None
   ```
3. **编写完整的单元测试**：
   - 各种格式的读取/写入测试
   - 无损往返测试
   - 错误处理测试

### 注意事项
1. **坐标精度**：浮点数精度问题，使用适当的数据类型
2. **内存管理**：大型数据集的处理，考虑流式处理
3. **性能优化**：批量操作、缓存机制
4. **向后兼容**：格式更新时保持向后兼容性

## 6. 开发要点

### 扩展指南
**添加新格式支持**：
1. 创建新的处理器类继承 `BaseAnnotationHandler`
2. 实现 `_read_implementation()` 和 `_write_implementation()`
3. 在 `AnnotationFormat` 枚举中添加新格式
4. 添加对应的单元测试

**自定义数据验证**：
```python
class CustomValidationHandler(BaseAnnotationHandler):
    def validate(self, input_path, **kwargs):
        result = super().validate(input_path, **kwargs)

        # 添加自定义验证规则
        if not self._check_custom_rule(input_path):
            result.add_error("Custom validation failed")

        return result
```

**性能优化建议**：
1. **批量处理**：使用生成器或分批处理大型数据集
2. **缓存机制**：缓存已解析的类别映射、图片信息等
3. **并行处理**：多图片读取/写入可使用多进程/多线程

### 调试技巧
1. **启用详细日志**：处理器支持 `verbose=True` 参数
2. **验证数据完整性**：使用 `validate()` 方法检查数据
3. **检查坐标转换**：验证归一化 ↔ 绝对像素转换的正确性
4. **测试无损往返**：读取后立即写回，比较文件差异

### 性能优化
1. **减少文件IO**：批量读取/写入，减少文件打开/关闭次数
2. **内存优化**：对于大型数据集，考虑流式处理或分块处理
3. **坐标计算优化**：向量化坐标转换计算
4. **缓存利用**：重复使用的数据（如类别映射）进行缓存

## 7. 测试指南

### 测试策略
**单元测试目标**：
- 各个处理器的读取/写入功能
- 数据模型的方法和属性
- 坐标转换的正确性
- 无损处理机制
- 错误处理逻辑

**集成测试目标**：
- 多格式间的数据一致性
- 与 convert 模块的集成
- 完整工作流的验证

### 测试数据准备
1. **小型测试数据集**：
   ```
   test_data/
   ├── yolo/
   │   ├── images/ (测试图片)
   │   ├── labels/ (YOLO标签)
   │   └── classes.txt
   ├── labelme/ (LabelMe JSON文件)
   └── coco/ (COCO JSON文件)
   ```
2. **边界情况数据**：
   - 空标注文件
   - 无效坐标数据
   - 缺失字段的数据
   - 大尺寸图片标注

### 示例测试用例

```python
# 测试 YOLO 处理器
def test_yolo_handler_read_detection():
    handler = YoloAnnotationHandler()
    result = handler.read(
        label_dir=TEST_YOLO_DIR,
        image_dir=TEST_IMAGE_DIR,
        class_file=TEST_CLASS_FILE
    )

    assert result.success is True
    assert len(result.data.images) > 0
    assert 0 in result.data.categories  # 类别ID 0存在

# 测试无损往返
def test_lossless_roundtrip_labelme():
    # 读取原始数据
    handler = LabelMeAnnotationHandler()
    read_result = handler.read(ORIGINAL_LABELME_DIR)

    # 写入到临时目录
    temp_dir = Path("temp_labelme")
    write_result = handler.write(read_result.data, temp_dir)

    # 验证文件内容相同
    original_files = list(ORIGINAL_LABELME_DIR.glob("*.json"))
    temp_files = list(temp_dir.glob("*.json"))

    assert len(original_files) == len(temp_files)
    for orig, temp in zip(original_files, temp_files):
        assert filecmp.cmp(orig, temp, shallow=False)

    # 清理
    shutil.rmtree(temp_dir)

# 测试坐标转换
def test_coordinate_conversion():
    # 测试归一化 → 绝对像素
    bbox = BoundingBox(x_center=0.5, y_center=0.5, width=0.2, height=0.3)
    abs_bbox = bbox.to_absolute(640, 480)

    assert abs(abs_bbox.x_center - 320) < 0.001
    assert abs(abs_bbox.y_center - 240) < 0.001
    assert abs(abs_bbox.width - 128) < 0.001
    assert abs(abs_bbox.height - 144) < 0.001

    # 测试绝对像素 → 归一化
    norm_bbox = BoundingBox.from_absolute(320, 240, 128, 144, 640, 480)
    assert abs(norm_bbox.x_center - 0.5) < 0.001
    assert abs(norm_bbox.y_center - 0.5) < 0.001

# 测试错误处理
def test_error_handling_non_strict_mode():
    handler = YoloAnnotationHandler(strict_mode=False)
    # 提供无效的标签文件
    result = handler.read(INVALID_LABEL_DIR, ...)

    assert result.success is False
    assert len(result.errors) > 0
    assert "Invalid coordinate" in result.errors[0]
```

### 测试覆盖率目标
- 处理器核心方法：100% 行覆盖
- 数据模型方法：100% 行覆盖
- 坐标转换逻辑：100% 分支覆盖
- 错误处理路径：90% 以上覆盖
- 边界条件测试：所有格式的边界情况

---

**文档版本**：1.0
**最后更新**：2026-03-29
**基于实现版本**：DataFlow-CV 当前实现