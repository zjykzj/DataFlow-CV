# DataFlow-CV 转换模块开发文档 (convert)

## 1. 模块概述

### 功能定位
`dataflow/convert` 模块是 DataFlow-CV 项目的格式转换引擎，负责在不同标注格式（YOLO、LabelMe、COCO）之间进行双向转换。它构建在 `label` 模块之上，提供完整的格式转换工作流，支持无损转换和批量处理。

### 核心特性
- **六种双向转换**：YOLO↔COCO、LabelMe↔YOLO、COCO↔LabelMe 全向转换
- **无损转换保证**：通过 `OriginalData` 机制支持无损往返转换
- **RLE 支持**：COCO 格式的 Run-Length Encoding 转换（可选）
- **目录结构管理**：自动创建和维护转换所需的目录结构
- **批量处理**：支持整个数据集的批量转换
- **详细日志**：完整的转换过程日志和进度报告
- **错误处理**：严格模式和非严格模式，支持错误收集和继续

### 设计原则
1. **模块化设计**：每个转换方向独立实现，便于维护和扩展
2. **依赖注入**：转换器通过工厂方法创建 label 处理器，降低耦合
3. **模板方法模式**：统一转换流程，确保一致性
4. **用户友好**：清晰的错误信息、进度反馈和结果报告
5. **性能优化**：批量处理、缓存机制和资源管理

## 2. 核心架构

### 类图/组件图
```
┌─────────────────────────────────────────┐
│          dataflow.convert               │
├─────────────────────────────────────────┤
│  • BaseConverter (抽象基类)             │
│    - convert() (模板方法)               │
│    - validate_inputs()                  │
│    - create_source_handler() (工厂方法) │
│    - create_target_handler() (工厂方法) │
│                                         │
│  • 具体转换器实现                       │
│    - YoloAndCocoConverter               │
│    - LabelMeAndYoloConverter            │
│    - CocoAndLabelMeConverter            │
│                                         │
│  • 辅助组件                             │
│    - ConversionResult (结果容器)        │
│    - RLEConverter (RLE转换工具)         │
│    - utils (路径处理等工具函数)         │
└─────────────────────────────────────────┘
```

### 数据流
1. **标准转换流程**（模板方法）：
   ```
   输入验证 → 创建源处理器 → 读取源数据 → 格式转换 → 创建目标处理器 → 写入目标格式 → 返回结果
         ↳ 参数检查        ↳ 路径解析          ↳ 坐标转换          ↳ 目录创建
   ```

2. **与 label 模块集成**：
   ```
   转换器 → 创建处理器 → label模块读取 → DatasetAnnotations → label模块写入 → 目标格式
             (工厂方法)                     (统一数据模型)
   ```

3. **无损转换机制**：
   ```
   源格式 → 读取(保存OriginalData) → DatasetAnnotations → 写入(优先使用OriginalData) → 目标格式
                                                              (如果格式匹配)
   ```

### 依赖关系
- **内部依赖**：`dataflow/label`（标注处理）, `dataflow/util`（日志和文件操作）
- **外部依赖**：
  - 必需：`numpy`, `opencv-python`
  - 可选：`pycocotools`（COCO RLE 支持）
- **被依赖**：`dataflow/cli`（CLI转换命令）

## 3. 主要API

### BaseConverter 抽象基类
**职责**：定义转换流程的模板方法

**核心方法**：
- `convert(source_path, target_path, **kwargs) -> ConversionResult`：执行转换（模板方法）
- `validate_inputs(source_path, target_path, **kwargs) -> bool`：验证输入参数
- `create_source_handler(**kwargs) -> BaseAnnotationHandler`：创建源处理器（工厂方法）
- `create_target_handler(**kwargs) -> BaseAnnotationHandler`：创建目标处理器（工厂方法）
- `convert_annotations(dataset, **kwargs) -> DatasetAnnotations`：执行格式特定的转换（抽象方法）

**模板方法实现**：
```python
def convert(self, source_path, target_path, **kwargs):
    # 1. 输入验证
    self.validate_inputs(source_path, target_path, **kwargs)

    # 2. 创建源处理器并读取数据
    source_handler = self.create_source_handler(**kwargs)
    read_result = source_handler.read(source_path, **kwargs)

    if not read_result.success:
        return ConversionResult.fail(f"Failed to read source: {read_result.errors}")

    # 3. 执行格式转换
    dataset = self.convert_annotations(read_result.data, **kwargs)

    # 4. 创建目标处理器并写入数据
    target_handler = self.create_target_handler(**kwargs)
    write_result = target_handler.write(dataset, target_path, **kwargs)

    # 5. 返回结果
    return ConversionResult(
        success=write_result.success,
        source_path=source_path,
        target_path=target_path,
        images_converted=len(dataset.images),
        errors=read_result.errors + write_result.errors,
        warnings=read_result.warnings + write_result.warnings
    )
```

### 具体转换器类

#### YoloAndCocoConverter 类
**职责**：YOLO 和 COCO 格式之间的双向转换

**转换方向**：
- YOLO → COCO：需要 `image_dir`, `class_file` 参数
- COCO → YOLO：需要 `class_file` 参数（可选）

**特殊处理**：
- COCO → YOLO：自动创建 `images/` 和 `labels/` 目录结构
- YOLO → COCO：支持 RLE 编码（`do_rle=True`）

#### LabelMeAndYoloConverter 类
**职责**：LabelMe 和 YOLO 格式之间的双向转换

**转换方向**：
- LabelMe → YOLO：需要 `class_file`, `image_dir`（可选）参数
- YOLO → LabelMe：需要 `class_file`, `image_dir` 参数

**特殊处理**：
- LabelMe → YOLO：从 JSON 文件中提取类别（如未提供 class_file）
- YOLO → LabelMe：为每张图片生成单独的 JSON 文件

#### CocoAndLabelMeConverter 类
**职责**：COCO 和 LabelMe 格式之间的双向转换

**转换方向**：
- COCO → LabelMe：需要 `class_file`（可选）参数
- LabelMe → COCO：需要 `class_file` 参数

**特殊处理**：
- COCO → LabelMe：crowd 标注的特殊处理
- LabelMe → COCO：支持 RLE 编码（`do_rle=True`）

### RLEConverter 工具类
**职责**：处理 COCO RLE 格式的编码和解码

**核心方法**：
- `encode_polygons_to_rle(polygons, height, width) -> Dict`：多边形转 RLE
- `decode_rle_to_polygons(rle_data) -> List[List[float]]`：RLE 转多边形
- `is_rle_available() -> bool`：检查 pycocotools 是否可用

**降级处理**：pycocotools 不可用时，自动降级为多边形格式并记录警告

### ConversionResult 类
**职责**：封装转换结果

**属性**：
- `success: bool`：转换是否成功
- `source_path: Path`：源路径
- `target_path: Path`：目标路径
- `images_converted: int`：转换的图片数量
- `errors: List[str]`：错误信息列表
- `warnings: List[str]`：警告信息列表
- `metadata: Dict[str, Any]`：转换元数据

### 配置参数

**通用参数**：
- `verbose: bool = False`：详细日志模式
- `strict_mode: bool = True`：严格错误处理模式
- `class_file: Optional[Path]`：类别文件路径（大多数转换需要）
- `image_dir: Optional[Path]`：图片目录路径

**转换特定参数**：
- `do_rle: bool = False`：是否使用 RLE 格式（COCO相关转换）
- `overwrite: bool = False`：是否覆盖已存在的目标文件

## 4. 使用指南

### 快速开始

```python
# 1. YOLO → COCO 转换
from dataflow.convert import YoloAndCocoConverter

converter = YoloAndCocoConverter(verbose=True)
result = converter.convert(
    source_path=Path("yolo_labels"),
    target_path=Path("coco_annotations.json"),
    image_dir=Path("images"),
    class_file=Path("classes.txt"),
    do_rle=True  # 使用 RLE 编码
)

if result.success:
    print(f"Converted {result.images_converted} images successfully")

# 2. COCO → YOLO 转换
result = converter.convert(
    source_path=Path("coco_annotations.json"),
    target_path=Path("yolo_output"),
    class_file=Path("classes.txt")
)

# 3. LabelMe → YOLO 转换
from dataflow.convert import LabelMeAndYoloConverter

converter = LabelMeAndYoloConverter()
result = converter.convert(
    source_path=Path("labelme_json"),
    target_path=Path("yolo_labels"),
    class_file=Path("classes.txt")
)

# 4. 完整转换链示例
# LabelMe → YOLO → COCO → LabelMe（无损往返验证）
```

### 常见场景

#### 场景1：完整数据集转换
```python
from dataflow.convert import YoloAndCocoConverter

converter = YoloAndCocoConverter(
    verbose=True,      # 详细日志
    strict_mode=False  # 非严格模式，继续处理错误
)

result = converter.convert(
    source_path=Path("datasets/yolo_train"),
    target_path=Path("datasets/coco_train.json"),
    image_dir=Path("datasets/images"),
    class_file=Path("datasets/classes.txt"),
    do_rle=True
)

print(f"Conversion summary:")
print(f"  Images: {result.images_converted}")
print(f"  Errors: {len(result.errors)}")
print(f"  Warnings: {len(result.warnings)}")

if result.warnings:
    print("\nWarnings:")
    for warn in result.warnings[:5]:  # 显示前5个警告
        print(f"  - {warn}")
```

#### 场景2：批量转换工作流
```python
import shutil
from pathlib import Path
from dataflow.convert import (
    YoloAndCocoConverter,
    LabelMeAndYoloConverter,
    CocoAndLabelMeConverter
)

# 工作流：YOLO → COCO → LabelMe → YOLO（验证无损）
yolo_dir = Path("original_yolo")
coco_file = Path("converted_coco.json")
labelme_dir = Path("converted_labelme")
final_yolo_dir = Path("final_yolo")

# 步骤1: YOLO → COCO
converter1 = YoloAndCocoConverter(verbose=True)
result1 = converter1.convert(
    yolo_dir, coco_file,
    image_dir=Path("images"),
    class_file=Path("classes.txt")
)

# 步骤2: COCO → LabelMe
converter2 = CocoAndLabelMeConverter(verbose=True)
result2 = converter2.convert(
    coco_file, labelme_dir,
    class_file=Path("classes.txt")
)

# 步骤3: LabelMe → YOLO
converter3 = LabelMeAndYoloConverter(verbose=True)
result3 = converter3.convert(
    labelme_dir, final_yolo_dir,
    class_file=Path("classes.txt")
)

# 验证转换准确性
print(f"Original YOLO images: {len(list(yolo_dir.glob('*.txt')))}")
print(f"Final YOLO images: {len(list(final_yolo_dir.glob('*.txt')))}")
```

#### 场景3：错误处理和调试
```python
converter = YoloAndCocoConverter(
    verbose=True,      # 启用详细日志
    strict_mode=False  # 收集所有错误，不立即停止
)

result = converter.convert(
    source_path=Path("problematic_data"),
    target_path=Path("output"),
    image_dir=Path("images"),
    class_file=Path("classes.txt")
)

if not result.success or result.errors:
    print("Conversion completed with issues:")
    print(f"  Success: {result.success}")
    print(f"  Images converted: {result.images_converted}")
    print(f"  Errors: {len(result.errors)}")
    print(f"  Warnings: {len(result.warnings)}")

    # 将错误保存到文件
    if result.errors:
        with open("conversion_errors.txt", "w") as f:
            for err in result.errors:
                f.write(f"{err}\n")
        print("Errors saved to conversion_errors.txt")

    # 检查日志文件（verbose模式会生成）
    log_files = list(Path("logs").glob("*.log"))
    if log_files:
        print(f"Detailed logs available in: {log_files[-1]}")
```

#### 场景4：自定义转换配置
```python
from dataflow.convert import LabelMeAndYoloConverter

# 自定义转换器配置
class CustomConverter(LabelMeAndYoloConverter):
    def convert_annotations(self, dataset, **kwargs):
        # 调用父类转换
        converted = super().convert_annotations(dataset, **kwargs)

        # 自定义处理：过滤低置信度标注
        if "min_confidence" in kwargs:
            min_conf = kwargs["min_confidence"]
            for image in converted.images:
                image.objects = [
                    obj for obj in image.objects
                    if obj.confidence is None or obj.confidence >= min_conf
                ]

        return converted

# 使用自定义转换器
converter = CustomConverter(verbose=True)
result = converter.convert(
    source_path=Path("labelme_data"),
    target_path=Path("filtered_yolo"),
    class_file=Path("classes.txt"),
    min_confidence=0.5  # 自定义参数
)
```

## 5. 开发步骤

### 阶段1：设计 BaseConverter 基类
1. **分析转换需求**：确定通用转换流程和接口
2. **设计模板方法模式**：
   ```python
   class BaseConverter(ABC):
       def convert(self, source_path, target_path, **kwargs):
           # 1. 输入验证
           self.validate_inputs(source_path, target_path, **kwargs)

           # 2. 读取源数据
           source_handler = self.create_source_handler(**kwargs)
           read_result = source_handler.read(source_path, **kwargs)

           # 3. 格式转换
           dataset = self.convert_annotations(read_result.data, **kwargs)

           # 4. 写入目标数据
           target_handler = self.create_target_handler(**kwargs)
           write_result = target_handler.write(dataset, target_path, **kwargs)

           # 5. 返回结果
           return self._create_result(read_result, write_result, dataset)
   ```
3. **设计工厂方法模式**：
   - `create_source_handler()`：创建源格式处理器
   - `create_target_handler()`：创建目标格式处理器
4. **设计结果封装**：`ConversionResult` 类设计

### 阶段2：实现 YoloAndCocoConverter
1. **实现双向转换逻辑**：
   ```python
   class YoloAndCocoConverter(BaseConverter):
       def convert_annotations(self, dataset, **kwargs):
           # 根据转换方向执行不同的转换逻辑
           if self._is_yolo_to_coco:
               return self._yolo_to_coco(dataset, **kwargs)
           else:
               return self._coco_to_yolo(dataset, **kwargs)
   ```
2. **实现 YOLO → COCO 转换**：
   - 处理 YOLO 特有的数据结构
   - 构建 COCO 格式的 categories、images、annotations
   - 支持 RLE 编码（如果启用）
3. **实现 COCO → YOLO 转换**：
   - 解析 COCO 的复杂结构
   - 创建 YOLO 目录结构（images/, labels/）
   - 复制图片文件到目标目录
4. **实现参数验证**：
   - 检查必需的参数（class_file, image_dir）
   - 验证路径存在性和权限

### 阶段3：实现 LabelMeAndYoloConverter
1. **实现 LabelMe → YOLO 转换**：
   - 解析 LabelMe JSON 文件
   - 提取形状数据和类别信息
   - 转换为 YOLO 格式（归一化坐标）
2. **实现 YOLO → LabelMe 转换**：
   - 为每张图片生成独立的 JSON 文件
   - 构建 LabelMe 格式的形状列表
   - 添加必要的元数据（version, flags等）
3. **处理类别文件**：
   - 支持从 LabelMe 标注中提取类别
   - 验证类别一致性

### 阶段4：实现 CocoAndLabelMeConverter
1. **实现 COCO → LabelMe 转换**：
   - 处理 COCO 的 crowd 标注（iscrowd）
   - 转换多边形和 RLE 数据
   - 生成每张图片的 JSON 文件
2. **实现 LabelMe → COCO 转换**：
   - 合并多个 JSON 文件为单个 COCO 文件
   - 构建 COCO 格式的数据结构
   - 支持 RLE 编码（如果启用）
3. **实现 RLE 支持**：
   - 集成 `RLEConverter` 工具类
   - 处理 pycocotools 不可用的情况

### 阶段5：实现 RLEConverter 工具类
1. **设计 RLE 编解码接口**：
   ```python
   class RLEConverter:
       @staticmethod
       def encode_polygons_to_rle(polygons, height, width):
           if not HAS_PYCOCOTOOLS:
               raise ImportError("pycocotools required for RLE encoding")

           import pycocotools.mask as mask_util
           # 创建二值掩码
           mask = np.zeros((height, width), dtype=np.uint8)
           # 绘制多边形...
           # 编码为 RLE
           rle = mask_util.encode(mask)
           return rle
   ```
2. **实现降级处理**：
   - 检查 pycocotools 可用性
   - 不可用时记录警告并降级为多边形格式
   - 提供清晰的错误信息

### 阶段6：集成目录结构管理和文件复制
1. **实现目录创建逻辑**：
   ```python
   def _prepare_output_directory(self, target_path, **kwargs):
       # 创建输出目录
       FileOperations().ensure_dir(target_path)

       # 对于 YOLO 格式，创建 images/ 和 labels/ 子目录
       if self._needs_yolo_structure:
           FileOperations().ensure_dir(target_path / "images")
           FileOperations().ensure_dir(target_path / "labels")
   ```
2. **实现文件复制逻辑**：
   - 转换时复制图片文件到目标目录
   - 支持覆盖选项（overwrite）
   - 进度报告和错误处理
3. **实现类别文件处理**：
   - 复制类别文件到目标目录
   - 验证类别文件格式和内容

### 阶段7：实现详细日志和错误处理
1. **集成日志系统**：
   ```python
   if self.verbose:
       self._logger = VerboseLoggingOperations().get_verbose_logger(
           f"dataflow.convert.{self.__class__.__name__}",
           verbose=True
       )
   else:
       self._logger = LoggingOperations().get_logger(
           f"dataflow.convert.{self.__class__.__name__}"
       )
   ```
2. **实现错误收集机制**：
   - 严格模式：错误时立即停止
   - 非严格模式：收集错误并继续
   - 结果中包含所有错误和警告
3. **实现进度报告**：
   - 批量转换时的进度显示
   - 详细日志中的处理详情

### 注意事项
1. **内存管理**：大型数据集的转换，考虑流式处理
2. **文件IO优化**：批量文件操作，减少系统调用
3. **错误恢复**：部分失败时的清理和恢复
4. **性能监控**：转换过程中的性能指标收集

## 6. 开发要点

### 扩展指南
**添加新的转换方向**：
1. 创建新的转换器类继承 `BaseConverter`
2. 实现 `create_source_handler()` 和 `create_target_handler()`
3. 实现 `convert_annotations()` 方法
4. 在 CLI 模块中添加对应的命令

**自定义转换逻辑**：
```python
class CustomYoloCocoConverter(YoloAndCocoConverter):
    def convert_annotations(self, dataset, **kwargs):
        # 调用父类转换
        converted = super().convert_annotations(dataset, **kwargs)

        # 添加自定义处理
        if "filter_classes" in kwargs:
            filter_set = set(kwargs["filter_classes"])
            for image in converted.images:
                image.objects = [
                    obj for obj in image.objects
                    if obj.class_name in filter_set
                ]

        return converted
```

**性能优化建议**：
1. **批量处理优化**：
   - 使用多进程处理大型数据集
   - 批量文件复制操作
2. **内存优化**：
   - 流式处理大型 COCO 文件
   - 及时释放不再需要的数据
3. **缓存利用**：
   - 缓存已解析的类别映射
   - 重复使用的数据预计算

### 调试技巧
1. **启用详细日志**：`verbose=True` 查看完整转换过程
2. **单步调试**：使用小数据集测试转换逻辑
3. **无损验证**：测试往返转换验证数据一致性
4. **错误收集**：使用 `strict_mode=False` 收集所有问题

### 性能优化
1. **减少文件IO**：
   - 批量读取/写入操作
   - 使用内存缓存频繁访问的数据
2. **算法优化**：
   - 向量化坐标计算
   - 避免重复的类型转换
3. **并行处理**：
   - 多图片转换可并行化
   - 文件复制操作可并行化

## 7. 测试指南

### 测试策略
**单元测试目标**：
- 各个转换器的双向转换功能
- 参数验证逻辑
- 错误处理机制
- 目录结构管理

**集成测试目标**：
- 与 label 模块的完整集成
- 无损往返转换验证
- 完整工作流测试

**性能测试目标**：
- 大型数据集的转换性能
- 内存使用情况
- 错误恢复能力

### 测试数据准备
1. **小型测试数据集**：
   ```
   test_convert/
   ├── images/ (测试图片，5-10张)
   ├── yolo/ (YOLO标签)
   ├── labelme/ (LabelMe JSON)
   ├── coco/ (COCO JSON)
   └── classes.txt
   ```
2. **边界情况数据**：
   - 空数据集
   - 大尺寸图片
   - 复杂标注（密集、嵌套等）
   - 缺失文件/目录

### 示例测试用例

```python
# 测试 YOLO → COCO 转换
def test_yolo_to_coco_conversion():
    converter = YoloAndCocoConverter()
    result = converter.convert(
        source_path=TEST_YOLO_DIR,
        target_path=Path("test_output.json"),
        image_dir=TEST_IMAGE_DIR,
        class_file=TEST_CLASS_FILE
    )

    assert result.success is True
    assert result.images_converted == len(list(TEST_IMAGE_DIR.glob("*.jpg")))
    assert Path("test_output.json").exists()

    # 验证输出文件是有效的 JSON
    import json
    with open("test_output.json") as f:
        data = json.load(f)
        assert "images" in data
        assert "annotations" in data
        assert "categories" in data

    # 清理
    Path("test_output.json").unlink()

# 测试无损往返
def test_lossless_roundtrip_yolo_coco():
    # YOLO → COCO → YOLO
    converter1 = YoloAndCocoConverter()
    result1 = converter1.convert(
        TEST_YOLO_DIR, Path("temp_coco.json"),
        image_dir=TEST_IMAGE_DIR,
        class_file=TEST_CLASS_FILE
    )

    converter2 = YoloAndCocoConverter()
    result2 = converter2.convert(
        Path("temp_coco.json"), Path("temp_yolo"),
        class_file=TEST_CLASS_FILE
    )

    assert result1.success is True
    assert result2.success is True

    # 比较原始和转换后的 YOLO 文件
    original_files = list(TEST_YOLO_DIR.glob("*.txt"))
    converted_files = list(Path("temp_yolo/labels").glob("*.txt"))

    assert len(original_files) == len(converted_files)

    # 清理
    Path("temp_coco.json").unlink()
    shutil.rmtree(Path("temp_yolo"))

# 测试错误处理
def test_converter_error_handling():
    converter = YoloAndCocoConverter(strict_mode=False)

    # 提供无效的源路径
    result = converter.convert(
        source_path=Path("non_existent"),
        target_path=Path("output"),
        image_dir=TEST_IMAGE_DIR,
        class_file=TEST_CLASS_FILE
    )

    assert result.success is False
    assert len(result.errors) > 0
    assert "not found" in result.errors[0].lower()

# 测试 RLE 支持
def test_rle_conversion():
    converter = YoloAndCocoConverter()

    result = converter.convert(
        source_path=TEST_YOLO_DIR,
        target_path=Path("test_rle.json"),
        image_dir=TEST_IMAGE_DIR,
        class_file=TEST_CLASS_FILE,
        do_rle=True
    )

    assert result.success is True

    # 验证输出包含 RLE 数据
    import json
    with open("test_rle.json") as f:
        data = json.load(f)
        # 检查是否有 annotations 包含 segmentation 字段
        has_rle = any(
            "segmentation" in ann and isinstance(ann["segmentation"], dict)
            for ann in data.get("annotations", [])
        )
        # RLE 可能不可用，取决于 pycocotools
        if HAS_PYCOCOTOOLS:
            assert has_rle or len(data.get("annotations", [])) == 0

    # 清理
    Path("test_rle.json").unlink()

# 测试目录结构创建
def test_directory_structure_creation():
    converter = YoloAndCocoConverter()
    output_dir = Path("test_yolo_output")

    result = converter.convert(
        source_path=TEST_COCO_FILE,
        target_path=output_dir,
        class_file=TEST_CLASS_FILE
    )

    assert result.success is True
    assert (output_dir / "images").exists()
    assert (output_dir / "labels").exists()
    assert len(list((output_dir / "images").glob("*"))) > 0

    # 清理
    shutil.rmtree(output_dir)
```

### 测试覆盖率目标
- 转换器核心方法：100% 行覆盖
- 参数验证逻辑：100% 分支覆盖
- 错误处理路径：所有错误情况测试
- 边界条件：各种边界情况测试
- 集成测试：主要转换路径集成测试

---

**文档版本**：1.0
**最后更新**：2026-03-29
**基于实现版本**：DataFlow-CV 当前实现