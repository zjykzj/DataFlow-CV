# DataFlow-CV 转换模块开发规范

## 1. 项目概述

### 1.1 项目背景和目标

DataFlow-CV已经实现了dataflow/label模块（支持LabelMe、YOLO、COCO格式的读写）和dataflow/visualize模块（可视化功能）。随着计算机视觉项目的复杂化，数据标注格式转换成为常见需求。不同的算法和框架使用不同的标注格式，导致数据预处理工作重复且容易出错。

本模块的核心目标是实现一个标准化、模块化的格式转换系统，支持：
- LabelMe、YOLO、COCO三种主流标注格式之间的双向转换（共6种转换方向）
- 目标检测和实例分割数据的统一转换
- RLE mask格式支持（COCO特有）
- 无损转换保证（除RLE精度损失外）
- 完整的测试和示例体系

### 1.2 核心功能特性

1. **全面转换支持**：支持LabelMe↔YOLO、YOLO↔COCO、COCO↔LabelMe六种转换方向
2. **双任务支持**：同时支持目标检测和实例分割算法数据标注转换
3. **RLE支持**：对COCO格式支持RLE mask格式转换，提供`do_rle`参数控制输出格式
4. **无损转换**：利用`OriginalData`机制实现格式间无损转换（除RLE精度损失外）
5. **类别映射**：智能处理格式间类别信息差异（COCO内嵌类别 vs YOLO/LabelMe外部类别文件）
6. **批量处理**：支持文件夹级别的批量转换
7. **严格验证**：严格的错误处理和格式验证机制，支持严格模式和宽松模式

### 1.3 设计原则

1. **统一性**：所有转换器遵循相同的接口规范，基于`BaseConverter`抽象基类
2. **无损性**：优先使用原始数据，确保转换回原始格式时得到相同文件（除RLE外）
3. **类型安全**：使用Python数据类（dataclass）确保类型安全
4. **错误处理**：严格模式（默认）遇错停止，宽松模式跳过错误继续处理
5. **可扩展性**：易于添加新的格式转换支持
6. **平台兼容**：确保在Windows、Linux、macOS上的完全兼容

## 2. 整体架构

### 2.1 模块组织结构图

```
DataFlow-CV/
├── dataflow/
│   ├── __init__.py
│   ├── label/                    # 现有标签模块
│   │   ├── base.py
│   │   ├── labelme_handler.py
│   │   ├── yolo_handler.py
│   │   ├── coco_handler.py
│   │   └── models.py
│   ├── convert/                  # 新转换模块
│   │   ├── __init__.py
│   │   ├── base.py               # BaseConverter抽象基类
│   │   ├── labelme_and_yolo.py   # LabelMe↔YOLO转换器
│   │   ├── yolo_and_coco.py      # YOLO↔COCO转换器
│   │   ├── coco_and_labelme.py   # COCO↔LabelMe转换器
│   │   ├── utils.py              # 转换工具函数
│   │   └── rle_converter.py      # RLE转换工具（可选依赖）
│   ├── util/                     # 现有工具模块
│   │   ├── file_util.py
│   │   └── logging_util.py
│   └── visualize/                # 现有可视化模块
├── tests/
│   ├── __init__.py
│   ├── convert/                  # 转换模块测试
│   │   ├── __init__.py
│   │   ├── test_base.py
│   │   ├── test_labelme_and_yolo.py
│   │   ├── test_yolo_and_coco.py
│   │   ├── test_coco_and_labelme.py
│   │   ├── test_rle_converter.py
│   │   └── test_utils.py
├── samples/
│   ├── convert/                  # 转换模块示例
│   │   ├── __init__.py
│   │   ├── labelme_to_yolo_demo.py
│   │   ├── yolo_to_labelme_demo.py
│   │   ├── yolo_to_coco_demo.py
│   │   ├── coco_to_yolo_demo.py
│   │   ├── coco_to_labelme_demo.py
│   │   ├── labelme_to_coco_demo.py
│   │   └── full_conversion_demo.py
└── assets
    └── test_data/                # 测试数据
```

### 2.2 文件树结构说明

- **dataflow/convert/**：核心转换模块目录
  - `base.py`：转换器抽象基类和结果数据类
  - `labelme_and_yolo.py`：LabelMe与YOLO格式互转
  - `yolo_and_coco.py`：YOLO与COCO格式互转
  - `coco_and_labelme.py`：COCO与LabelMe格式互转
  - `utils.py`：转换工具函数
  - `rle_converter.py`：RLE格式转换工具（可选依赖pycocotools）
- **tests/convert/**：转换模块测试目录
- **samples/convert/**：转换模块示例代码目录
- **assets/test_data/**：测试数据资源目录

### 2.3 依赖关系

```
dataflow.convert
├── dataflow.label (核心依赖)
├── dataflow.util (文件操作和日志工具)
├── pycocotools (可选依赖，RLE转换)
├── opencv-python (图像处理，可选)
├── numpy (数值计算)
└── Python标准库
```

## 3. dataflow/convert模块详细设计

### 3.1 基础组件

#### 3.1.1 BaseConverter抽象基类

所有格式转换器的基类，定义统一的转换接口规范。

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import logging
from pathlib import Path

from dataflow.label.base import AnnotationResult
from dataflow.label.models import DatasetAnnotations, AnnotationFormat
from dataflow.util.file_util import FileOperations

@dataclass
class ConversionResult:
    """转换操作结果"""
    success: bool
    source_format: str
    target_format: str
    source_path: str
    target_path: str
    num_images_converted: int = 0
    num_objects_converted: int = 0
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_warning(self, warning: str):
        """添加警告消息"""
        self.warnings.append(warning)

    def add_error(self, error: str):
        """添加错误消息"""
        self.errors.append(error)
        self.success = False

    def add_metadata(self, key: str, value: Any):
        """添加元数据键值对"""
        self.metadata[key] = value

    def get_summary(self) -> str:
        """获取转换结果摘要"""
        if self.success:
            return (f"成功转换 {self.num_images_converted} 张图片 "
                    f"共 {self.num_objects_converted} 个对象 "
                    f"从 {self.source_format} 到 {self.target_format}")
        else:
            return f"转换失败，共 {len(self.errors)} 个错误"

class BaseConverter(ABC):
    """格式转换器抽象基类"""

    def __init__(self,
                 source_format: str,
                 target_format: str,
                 strict_mode: bool = True,
                 logger: Optional[logging.Logger] = None):
        """
        初始化基础转换器

        Args:
            source_format: 源标注格式名称
            target_format: 目标标注格式名称
            strict_mode: 是否在错误时停止（默认True）
            logger: 可选日志器实例
        """
        self.source_format = source_format
        self.target_format = target_format
        self.strict_mode = strict_mode
        self.logger = logger or logging.getLogger(__name__)
        self.file_ops = FileOperations(logger=self.logger)

    @abstractmethod
    def convert(self, source_path: str, target_path: str, **kwargs) -> ConversionResult:
        """
        将标注从源格式转换为目标格式

        Args:
            source_path: 源标注路径
            target_path: 目标标注路径
            **kwargs: 附加转换参数

        Returns:
            ConversionResult包含转换状态和详情
        """
        pass

    def validate_inputs(self, source_path: str, target_path: str, kwargs: Dict) -> bool:
        """
        验证转换输入参数

        Args:
            source_path: 源标注路径
            target_path: 目标标注路径
            kwargs: 附加转换参数

        Returns:
            True如果输入有效，False否则
        """
        # 实现细节：验证路径存在性、权限等
        pass

    @abstractmethod
    def create_source_handler(self, source_path: str, kwargs: Dict) -> Any:
        """
        创建源标注处理器

        Args:
            source_path: 源标注路径
            kwargs: 附加转换参数

        Returns:
            BaseAnnotationHandler子类实例
        """
        pass

    @abstractmethod
    def create_target_handler(self, target_path: str, kwargs: Dict) -> Any:
        """
        创建目标标注处理器

        Args:
            target_path: 目标标注路径
            kwargs: 附加转换参数

        Returns:
            BaseAnnotationHandler子类实例
        """
        pass

    def convert_annotations(self,
                           source_annotations: DatasetAnnotations,
                           kwargs: Dict) -> DatasetAnnotations:
        """
        转换标注数据（格式特定转换）

        Args:
            source_annotations: 从源格式读取的标注数据
            kwargs: 附加转换参数

        Returns:
            转换后的DatasetAnnotations，准备写入目标格式
        """
        # 默认实现：直接返回（类别和数据已正确）
        return source_annotations

    def _log_info(self, message: str):
        """记录信息日志"""
        self.logger.info(message)

    def _log_warning(self, message: str):
        """记录警告日志"""
        self.logger.warning(message)

    def _log_error(self, message: str):
        """记录错误日志，严格模式下抛出异常"""
        self.logger.error(message)
        if self.strict_mode:
            raise ValueError(message)
```

#### 3.1.2 通用转换流程

所有转换器遵循相同的转换流程：

```python
def convert(self, source_path, target_path, **kwargs):
    # 1. 验证输入参数
    if not self.validate_inputs(source_path, target_path, kwargs):
        return ConversionResult(
            success=False,
            source_format=self.source_format,
            target_format=self.target_format,
            source_path=source_path,
            target_path=target_path,
            errors=["输入参数验证失败"]
        )

    # 2. 使用源handler读取数据
    source_handler = self.create_source_handler(source_path, kwargs)
    read_result = source_handler.read()
    if not read_result.success:
        return self._create_conversion_result(
            success=False,
            source_path=source_path,
            target_path=target_path,
            errors=read_result.errors
        )

    # 3. 转换数据（格式特定转换，如类别映射等）
    annotations = read_result.data
    converted_annotations = self.convert_annotations(annotations, kwargs)

    # 4. 使用目标handler写入数据
    target_handler = self.create_target_handler(target_path, kwargs)
    write_result = target_handler.write(converted_annotations, target_path)

    # 5. 返回结果
    return self._create_conversion_result(
        success=write_result.success,
        source_path=source_path,
        target_path=target_path,
        annotations=converted_annotations,
        write_result=write_result
    )
```

### 3.2 转换器设计

#### 3.2.1 LabelMeAndYoloConverter

处理LabelMe与YOLO格式之间的双向转换。

**类定义**：
```python
class LabelMeAndYoloConverter(BaseConverter):
    """LabelMe与YOLO格式转换器"""

    def __init__(self, source_to_target: bool, **kwargs):
        """
        初始化转换器

        Args:
            source_to_target: True表示LabelMe→YOLO，False表示YOLO→LabelMe
            **kwargs: 传递给BaseConverter的参数
        """
        if source_to_target:
            source_format = "labelme"
            target_format = "yolo"
        else:
            source_format = "yolo"
            target_format = "labelme"

        super().__init__(source_format, target_format, **kwargs)
        self.source_to_target = source_to_target
```

**LabelMe→YOLO转换参数**：
- `source_path`: LabelMe标签目录路径（包含JSON文件）
- `target_path`: 输出目录路径（将创建`images/`和`labels/`子目录）
- `class_file`: 类别文件路径（必需，每行一个类别名）
- `image_dir`: 源图片目录路径（可选，默认与JSON文件同目录）

**YOLO→LabelMe转换参数**：
- `source_path`: YOLO标签目录路径（包含.txt文件）
- `target_path`: 输出目录路径（将生成JSON文件）
- `class_file`: 类别文件路径（必需，每行一个类别名）
- `image_dir`: 图片目录路径（必需，用于获取图片尺寸）

**关键特性**：
1. 自动处理类别文件生成和使用
2. 支持目标检测和实例分割标注
3. 坐标系统正确转换（归一化↔绝对坐标）
4. 保留原始数据确保无损转换

#### 3.2.2 YoloAndCocoConverter

处理YOLO与COCO格式之间的双向转换。

**类定义**：
```python
class YoloAndCocoConverter(BaseConverter):
    """YOLO与COCO格式转换器"""

    def __init__(self, source_to_target: bool, **kwargs):
        """
        初始化转换器

        Args:
            source_to_target: True表示YOLO→COCO，False表示COCO→YOLO
            **kwargs: 传递给BaseConverter的参数
        """
        if source_to_target:
            source_format = "yolo"
            target_format = "coco"
        else:
            source_format = "coco"
            target_format = "yolo"

        super().__init__(source_format, target_format, **kwargs)
        self.source_to_target = source_to_target
```

**YOLO→COCO转换参数**：
- `source_path`: YOLO标签目录路径
- `target_path`: 输出COCO JSON文件路径
- `class_file`: 类别文件路径（必需）
- `image_dir`: 图片目录路径（必需）
- `do_rle`: 是否输出RLE格式（默认False，输出多边形点列表）

**COCO→YOLO转换参数**：
- `source_path`: COCO JSON文件路径
- `target_path`: 输出目录路径（将创建`classes.txt`和`labels/`子目录）
- `image_dir`: 图片目录路径（可选，从COCO中提取图片路径）

**关键特性**：
1. 自动从COCO JSON提取类别信息并生成classes.txt
2. 支持RLE格式转换（需要pycocotools）
3. 正确处理COCO特有字段（is_crowd, area等）
4. 图片信息完整转换

#### 3.2.3 CocoAndLabelMeConverter

处理COCO与LabelMe格式之间的双向转换。

**类定义**：
```python
class CocoAndLabelMeConverter(BaseConverter):
    """COCO与LabelMe格式转换器"""

    def __init__(self, source_to_target: bool, **kwargs):
        """
        初始化转换器

        Args:
            source_to_target: True表示COCO→LabelMe，False表示LabelMe→COCO
            **kwargs: 传递给BaseConverter的参数
        """
        if source_to_target:
            source_format = "coco"
            target_format = "labelme"
        else:
            source_format = "labelme"
            target_format = "coco"

        super().__init__(source_format, target_format, **kwargs)
        self.source_to_target = source_to_target
```

**COCO→LabelMe转换参数**：
- `source_path`: COCO JSON文件路径
- `target_path`: 输出目录路径（将生成JSON文件和classes.txt）
- `image_dir`: 图片目录路径（可选，从COCO中提取图片路径）

**LabelMe→COCO转换参数**：
- `source_path`: LabelMe标签目录路径
- `target_path`: 输出COCO JSON文件路径
- `class_file`: 类别文件路径（必需）
- `do_rle`: 是否输出RLE格式（默认False）

**关键特性**：
1. 复杂的格式映射（COCO JSON ↔ LabelMe JSON集合）
2. 数据集级别信息转换
3. RLE支持（LabelMe→COCO时可用）
4. 元数据完整保留

### 3.3 类别处理策略

不同格式有不同的类别管理方式，转换时需要智能处理：

#### 3.3.1 类别信息来源

| 格式 | 类别信息来源 | 转换输出 |
|------|-------------|----------|
| COCO | JSON内嵌的categories数组 | 提取并生成classes.txt（COCO→其他时） |
| YOLO | 外部classes.txt文件 | 使用提供的classes.txt，或嵌入到COCO JSON |
| LabelMe | 外部classes.txt或从标注提取 | 使用提供的classes.txt，或从标注提取 |

#### 3.3.2 转换时的类别映射

```python
def _extract_categories_from_coco(self, coco_data: Dict) -> Dict[int, str]:
    """从COCO数据中提取类别信息"""
    categories = {}
    for cat in coco_data.get("categories", []):
        cat_id = cat.get("id")
        cat_name = cat.get("name", "")
        if cat_id is not None:
            categories[cat_id] = cat_name
    return categories

def _generate_classes_file(self, categories: Dict[int, str], output_path: Path) -> bool:
    """生成classes.txt文件"""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            # 按ID排序写入
            for cat_id in sorted(categories.keys()):
                f.write(f"{categories[cat_id]}\n")
        return True
    except Exception as e:
        self._log_error(f"写入classes文件失败 {output_path}: {e}")
        return False

def _load_classes_file(self, class_file: Path) -> Dict[int, str]:
    """从classes.txt文件加载类别映射"""
    categories = {}
    try:
        with open(class_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if line:  # 跳过空行
                    categories[i] = line
    except Exception as e:
        self._log_error(f"加载classes文件失败 {class_file}: {e}")
    return categories
```

#### 3.3.3 类别冲突处理

当类别ID或名称冲突时，转换器需要：
1. 记录警告信息
2. 尝试自动解决冲突（如重新编号）
3. 在严格模式下抛出错误

### 3.4 RLE支持

#### 3.4.1 RLEConverter工具类

```python
class RLEConverter:
    """RLE格式转换工具类"""

    def __init__(self):
        self.has_coco_mask = False
        try:
            from pycocotools import mask as coco_mask
            self.coco_mask = coco_mask
            self.has_coco_mask = True
        except ImportError:
            self._log_warning("pycocotools未安装，RLE支持禁用")

    def polygon_to_rle(self, points: List[Tuple[float, float]],
                      img_width: int, img_height: int) -> Dict:
        """将多边形点列表转换为RLE格式"""
        if not self.has_coco_mask:
            raise ImportError("需要pycocotools进行RLE转换")

        # 将归一化坐标转换为绝对坐标
        abs_points = [(int(x * img_width), int(y * img_height)) for x, y in points]

        # 创建二值掩码
        import numpy as np
        import cv2
        mask = np.zeros((img_height, img_width), dtype=np.uint8)
        contour = np.array(abs_points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [contour], 1)

        # 编码为RLE
        rle = self.coco_mask.encode(np.asfortranarray(mask))
        return rle

    def rle_to_polygon(self, rle: Dict,
                      img_width: int, img_height: int) -> List[Tuple[float, float]]:
        """将RLE格式解码为多边形点列表"""
        if not self.has_coco_mask:
            raise ImportError("需要pycocotools进行RLE解码")

        # 解码RLE为二值掩码
        binary_mask = self.coco_mask.decode(rle)

        # 从掩码提取轮廓
        import cv2
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return []

        # 取最大轮廓
        largest_contour = max(contours, key=cv2.contourArea)

        # 转换为归一化坐标
        points = []
        for point in largest_contour:
            x, y = point[0]
            points.append((x / img_width, y / img_height))

        return points
```

#### 3.4.2 do_rle参数控制

对于YOLO/LabelMe→COCO转换，`do_rle`参数控制输出格式：
- `do_rle=False`（默认）：输出多边形点列表，适合人工编辑
- `do_rle=True`：输出RLE mask格式，适合存储和计算

**精度损失警告**：RLE转换会有精度损失，转换器需要记录警告信息。

#### 3.4.3 可选依赖处理

当pycocotools未安装时：
1. `do_rle=True`会引发ImportError
2. `do_rle=False`可以正常工作
3. 转换器记录警告信息但继续执行

### 3.5 无损转换保证

#### 3.5.1 OriginalData机制

利用dataflow/label模块的`OriginalData`类实现无损转换：

```python
@dataclass
class OriginalData:
    """原始标注数据，用于无损转换"""
    format: str  # 原始格式名称
    raw_data: Any  # 原始数据（通常是字典或列表）
    metadata: Dict[str, Any] = field(default_factory=dict)
```

#### 3.5.2 转换链中的原始数据传递

1. **读取时保留**：每个`ObjectAnnotation`和`ImageAnnotation`都保存其原始数据
2. **转换时传递**：格式转换过程中，原始数据随标注对象一起传递
3. **写入时优先使用**：当转换回原始格式时，优先使用保存的原始数据

#### 3.5.3 无损验证

```python
def verify_lossless_conversion(self, original_path: str, converted_path: str) -> bool:
    """
    验证转换是否无损

    Args:
        original_path: 原始文件/目录路径
        converted_path: 转换后文件/目录路径

    Returns:
        True如果转换无损（除RLE外），False否则
    """
    # 实现细节：比较原始文件和转换后文件的差异
    # 注意：RLE转换会有精度损失，需要特殊处理
    pass
```

### 3.6 错误处理策略

#### 3.6.1 严格模式 vs 宽松模式

- **严格模式**（默认）：遇到任何错误立即停止，返回失败结果
- **宽松模式**：记录错误但继续处理，跳过有问题的文件

#### 3.6.2 错误类型分类

1. **输入错误**：路径不存在、权限不足、格式不匹配
2. **格式错误**：无效的JSON、缺少必需字段、坐标越界
3. **转换错误**：类别映射失败、坐标转换失败、RLE编码失败
4. **输出错误**：写入失败、磁盘空间不足

#### 3.6.3 错误恢复机制

在宽松模式下，转换器需要：
1. 记录详细的错误信息
2. 跳过当前文件继续处理下一个
3. 返回部分成功的结果
4. 提供错误统计信息

## 4. dataflow/convert/utils模块详细设计

### 4.1 工具函数集合

`utils.py`提供转换过程中常用的工具函数：

```python
def extract_categories_from_annotations(annotations: Any) -> Dict[int, str]:
    """从DatasetAnnotations提取类别映射"""
    return annotations.categories.copy()

def generate_classes_file(categories: Dict[int, str], output_path: Path) -> bool:
    """从类别映射生成classes.txt文件"""

def load_classes_file(class_file: Path) -> Dict[int, str]:
    """从classes.txt文件加载类别映射"""

def extract_categories_from_coco(coco_data: Dict) -> Dict[int, str]:
    """从COCO数据中提取类别信息"""

def ensure_categories_in_annotations(annotations: Any,
                                    categories: Dict[int, str]) -> Any:
    """确保标注数据包含指定的类别映射"""

def get_image_dimensions_from_handler(handler: Any, image_path: str) -> Tuple[int, int]:
    """使用处理器的内部方法获取图片尺寸"""

def normalize_path(path: str, base_dir: Path) -> Path:
    """规范化路径（相对路径转为绝对路径）"""

def validate_conversion_chain(source_format: str, target_format: str,
                             allowed_chains: List[Tuple[str, str]]) -> bool:
    """验证转换链是否允许"""

def create_conversion_chain(chain: List[str]) -> List[Tuple[str, str]]:
    """从格式链创建转换步骤列表"""
```

### 4.2 路径处理工具

```python
def resolve_image_paths(annotations: DatasetAnnotations,
                       source_dir: Path,
                       target_dir: Path) -> DatasetAnnotations:
    """
    解析和规范化图片路径

    Args:
        annotations: 标注数据
        source_dir: 源目录（用于解析相对路径）
        target_dir: 目标目录（用于生成新路径）

    Returns:
        更新了图片路径的标注数据
    """
    updated_images = []
    for image_ann in annotations.images:
        # 解析源路径
        source_path = normalize_path(image_ann.image_path, source_dir)

        # 生成目标路径（保持相对结构）
        if source_path.is_relative_to(source_dir):
            relative_path = source_path.relative_to(source_dir)
            target_path = target_dir / relative_path
        else:
            # 无法确定相对路径，使用文件名
            target_path = target_dir / Path(image_ann.image_path).name

        # 更新图片标注
        updated_ann = ImageAnnotation(
            image_id=image_ann.image_id,
            image_path=str(target_path),
            width=image_ann.width,
            height=image_ann.height,
            objects=image_ann.objects,
            original_data=image_ann.original_data
        )
        updated_images.append(updated_ann)

    return DatasetAnnotations(
        images=updated_images,
        categories=annotations.categories,
        dataset_info=annotations.dataset_info
    )
```

## 5. 技术实现要求

### 5.1 核心依赖

**必需依赖**：
- `dataflow.label>=1.0.0`：标签处理模块（项目内部）
- `dataflow.util>=1.0.0`：工具模块（项目内部）
- `numpy>=1.19.0`：数值计算
- `opencv-python>=4.5.0`：图像处理（用于RLE转换）

**可选依赖**：
- `pycocotools>=2.0.0`：RLE格式支持（COCO专用）

**Python版本**：
- Python 3.8+（确保类型提示支持）

### 5.2 跨平台兼容性保证

1. **路径处理**：统一使用`pathlib.Path`，避免字符串拼接
2. **文件编码**：统一使用UTF-8编码
3. **行尾符**：使用Python的通用换行模式
4. **临时文件**：使用`tempfile`模块确保跨平台兼容
5. **权限处理**：正确处理文件权限和异常

### 5.3 性能优化

1. **批量处理**：支持文件夹级别的批量转换
2. **惰性加载**：大文件支持流式读取
3. **内存优化**：避免不必要的数据复制

### 5.4 代码质量要求

1. **类型提示**：所有公共API必须有完整的类型提示
2. **文档字符串**：所有类和方法必须有完整的docstring（Google风格）
3. **代码风格**：遵循PEP 8规范，使用black格式化
4. **单元测试**：核心功能必须有单元测试覆盖
5. **错误处理**：全面的错误处理和异常捕获

## 6. 测试策略

### 6.1 测试目录结构

```
tests/convert/
├── __init__.py
├── conftest.py                    # 测试配置和fixture
├── test_base.py                   # BaseConverter测试
├── test_labelme_and_yolo.py       # LabelMe↔YOLO转换测试
├── test_yolo_and_coco.py          # YOLO↔COCO转换测试
├── test_coco_and_labelme.py       # COCO↔LabelMe转换测试
├── test_rle_converter.py          # RLE转换工具测试
├── test_utils.py                  # 工具函数测试
└── assets/                        # 测试数据（符号链接到主assets）
```

### 6.2 测试数据准备

位于`assets/test_data/`中的现有数据：

```
assets/
├── test_data
│   ├── det
│   │   ├── coco
│   │   │   ├── annotations.json
│   │   │   ├── annotations_converted.json
│   │   │   ├── conversion_test
│   │   │   │   ├── original.json
│   │   │   │   └── rle.json
│   │   │   └── images
│   │   │       ├── image1.jpg
│   │   │       └── image2.jpg
│   │   ├── labelme
│   │   │   ├── classes.txt
│   │   │   ├── image1.jpg
│   │   │   ├── image1.json
│   │   │   ├── image2.jpg
│   │   │   └── image2.json
│   │   └── yolo
│   │       ├── classes.txt
│   │       ├── images
│   │       │   ├── image1.jpg
│   │       │   └── image2.jpg
│   │       ├── labels
│   │       │   ├── image1.txt
│   │       │   └── image2.txt
│   │       └── output
│   │           ├── image1.txt
│   │           └── image2.txt
│   └── seg
│       ├── coco
│       │   ├── annotations-rle.json
│       │   ├── annotations.json
│       │   └── images
│       │       ├── image1.jpg
│       │       └── image2.jpg
│       ├── labelme
│       │   ├── classes.txt
│       │   ├── image1.jpg
│       │   ├── image1.json
│       │   ├── image2.jpg
│       │   └── image2.json
│       └── yolo
│           ├── classes.txt
│           ├── images
│           │   ├── image1.jpg
│           │   └── image2.jpg
│           ├── labels
│           │   ├── image1.txt
│           │   └── image2.txt
│           └── output
│               ├── image1.txt
│               └── image2.txt
```

### 6.3 测试用例设计

#### 6.3.1 基础功能测试

```python
def test_labelme_to_yolo_detection():
    """测试LabelMe→YOLO目标检测转换"""
    # 1. 准备测试数据
    # 2. 执行转换
    # 3. 验证输出文件存在
    # 4. 验证类别文件正确
    # 5. 验证标注内容正确

def test_yolo_to_coco_with_rle():
    """测试YOLO→COCO转换（带RLE）"""
    # 1. 准备测试数据
    # 2. 执行转换（do_rle=True）
    # 3. 验证RLE格式正确
    # 4. 验证精度损失警告记录

def test_coco_to_labelme_roundtrip():
    """测试COCO→LabelMe往返转换"""
    # 1. 准备原始COCO数据
    # 2. COCO→LabelMe转换
    # 3. LabelMe→COCO转换
    # 4. 验证无损性（比较原始和最终COCO数据）
```

#### 6.3.2 边界条件测试

```python
def test_empty_annotations():
    """测试空标注转换"""

def test_large_file_conversion():
    """测试大文件转换性能"""

def test_invalid_input_path():
    """测试无效输入路径的错误处理"""

def test_missing_class_file():
    """测试缺少类别文件的错误处理"""
```

#### 6.3.3 错误处理测试

```python
def test_strict_mode_errors():
    """测试严格模式的错误处理"""

def test_lenient_mode_skip():
    """测试宽松模式的错误跳过"""

def test_rle_without_pycocotools():
    """测试未安装pycocotools时的RLE处理"""
```

### 6.4 测试覆盖率目标

- 总体覆盖率：≥90%
- 核心转换逻辑覆盖率：≥95%
- 错误处理覆盖率：≥85%
- RLE相关功能覆盖率：≥80%（如果pycocotools可用）

### 6.5 集成测试

```python
def test_full_conversion_chain():
    """测试完整转换链：LabelMe→YOLO→COCO→LabelMe"""
    # 1. 准备原始LabelMe数据
    # 2. LabelMe→YOLO转换
    # 3. YOLO→COCO转换
    # 4. COCO→LabelMe转换
    # 5. 验证最终LabelMe数据与原始数据一致（除RLE精度损失外）
```

## 7. 示例代码

### 7.1 示例目录结构

```
samples/convert/
├── __init__.py
├── labelme_to_yolo_demo.py      # LabelMe→YOLO示例
├── yolo_to_labelme_demo.py      # YOLO→LabelMe示例
├── yolo_to_coco_demo.py         # YOLO→COCO示例
├── coco_to_yolo_demo.py         # COCO→YOLO示例
├── coco_to_labelme_demo.py      # COCO→LabelMe示例
├── labelme_to_coco_demo.py      # LabelMe→COCO示例
└── full_conversion_demo.py      # 完整转换链示例
```

### 7.2 示例代码编写规范

1. **完整性**：展示完整的使用流程
2. **简洁性**：聚焦核心功能，避免冗余代码
3. **注释**：详细的步骤说明和注意事项
4. **错误处理**：展示正确的错误处理方式
5. **可执行性**：确保示例可以直接运行（使用项目内测试数据）

### 7.3 LabelMe→YOLO使用示例

```python
#!/usr/bin/env python3
"""
LabelMe到YOLO格式转换示例

展示如何使用LabelMeAndYoloConverter将LabelMe格式标注转换为YOLO格式。
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dataflow.convert import LabelMeAndYoloConverter
from dataflow.util import LoggingOperations

def main():
    """主函数"""
    # 配置日志
    log_ops = LoggingOperations()
    logger = log_ops.get_logger("labelme_to_yolo_demo", level="INFO")

    # 示例数据路径
    data_dir = project_root / "assets" / "test_data" / "det" / "labelme"
    class_file = data_dir / "classes.txt"
    output_dir = project_root / "samples" / "convert" / "output" / "labelme_to_yolo"

    if not data_dir.exists():
        logger.error(f"数据目录不存在: {data_dir}")
        logger.info("请确保已准备示例数据")
        return

    logger.info("=" * 50)
    logger.info("LabelMe到YOLO格式转换示例")
    logger.info("=" * 50)

    # 创建转换器（LabelMe→YOLO方向）
    logger.info("创建LabelMe→YOLO转换器")
    converter = LabelMeAndYoloConverter(
        source_to_target=True,  # LabelMe→YOLO
        strict_mode=True,
        logger=logger
    )

    # 执行转换
    logger.info(f"执行转换:")
    logger.info(f"  源目录: {data_dir}")
    logger.info(f"  目标目录: {output_dir}")
    logger.info(f"  类别文件: {class_file}")

    result = converter.convert(
        source_path=str(data_dir),
        target_path=str(output_dir),
        class_file=str(class_file)
    )

    # 显示结果
    logger.info("\n转换结果:")
    logger.info(f"  成功: {result.success}")
    logger.info(f"  转换图片数: {result.num_images_converted}")
    logger.info(f"  转换对象数: {result.num_objects_converted}")

    if result.success:
        logger.info(f"  输出目录: {output_dir}")
        logger.info(f"  生成的文件:")
        for file in output_dir.rglob("*"):
            if file.is_file():
                logger.info(f"    - {file.relative_to(output_dir)}")
    else:
        logger.error(f"  错误数: {len(result.errors)}")
        for error in result.errors:
            logger.error(f"    - {error}")

    if result.warnings:
        logger.warning(f"  警告数: {len(result.warnings)}")
        for warning in result.warnings:
            logger.warning(f"    - {warning}")

    logger.info("\n示例完成！")

if __name__ == "__main__":
    main()
```

### 7.4 完整转换链示例

```python
#!/usr/bin/env python3
"""
完整格式转换链示例

展示如何将LabelMe格式转换为YOLO，再转换为COCO，最后转回LabelMe，
验证无损转换（除RLE精度损失外）。
"""

import sys
from pathlib import Path
import tempfile
import shutil

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dataflow.convert import (
    LabelMeAndYoloConverter,
    YoloAndCocoConverter,
    CocoAndLabelMeConverter
)
from dataflow.util import LoggingOperations

def main():
    """主函数"""
    # 配置日志
    log_ops = LoggingOperations()
    logger = log_ops.get_logger("full_conversion_demo", level="INFO")

    # 创建临时工作目录
    temp_dir = Path(tempfile.mkdtemp(prefix="dataflow_convert_"))
    logger.info(f"创建临时工作目录: {temp_dir}")

    try:
        # 示例数据路径
        data_dir = project_root / "assets" / "test_data" / "det" / "labelme"
        class_file = data_dir / "classes.txt"

        if not data_dir.exists():
            logger.error(f"数据目录不存在: {data_dir}")
            return

        logger.info("=" * 50)
        logger.info("完整格式转换链示例")
        logger.info("=" * 50)

        # 第1步：LabelMe → YOLO
        logger.info("\n1. LabelMe → YOLO 转换")
        yolo_dir = temp_dir / "yolo_output"

        converter1 = LabelMeAndYoloConverter(
            source_to_target=True,
            strict_mode=True,
            logger=logger
        )

        result1 = converter1.convert(
            source_path=str(data_dir),
            target_path=str(yolo_dir),
            class_file=str(class_file)
        )

        if not result1.success:
            logger.error("LabelMe→YOLO转换失败")
            return

        # 第2步：YOLO → COCO
        logger.info("\n2. YOLO → COCO 转换")
        coco_file = temp_dir / "coco_output.json"

        converter2 = YoloAndCocoConverter(
            source_to_target=True,
            strict_mode=True,
            logger=logger
        )

        result2 = converter2.convert(
            source_path=str(yolo_dir / "labels"),
            target_path=str(coco_file),
            class_file=str(yolo_dir / "classes.txt"),
            image_dir=str(yolo_dir / "images"),
            do_rle=False  # 不使用RLE以确保无损
        )

        if not result2.success:
            logger.error("YOLO→COCO转换失败")
            return

        # 第3步：COCO → LabelMe
        logger.info("\n3. COCO → LabelMe 转换")
        labelme_dir = temp_dir / "labelme_output"

        converter3 = CocoAndLabelMeConverter(
            source_to_target=True,
            strict_mode=True,
            logger=logger
        )

        result3 = converter3.convert(
            source_path=str(coco_file),
            target_path=str(labelme_dir)
        )

        if not result3.success:
            logger.error("COCO→LabelMe转换失败")
            return

        # 验证结果
        logger.info("\n转换链完成！")
        logger.info(f"原始LabelMe文件数: {len(list(data_dir.glob('*.json')))}")
        logger.info(f"最终LabelMe文件数: {len(list(labelme_dir.glob('*.json')))}")

        # 简单验证：文件数量一致
        original_count = len(list(data_dir.glob('*.json')))
        final_count = len(list(labelme_dir.glob('*.json')))

        if original_count == final_count:
            logger.info("✓ 文件数量一致，转换链完整")
        else:
            logger.warning(f"⚠ 文件数量不一致: 原始={original_count}, 最终={final_count}")

        logger.info(f"\n临时工作目录: {temp_dir}")
        logger.info("（程序结束后会自动清理）")

    finally:
        # 清理临时目录（在实际使用中可能保留用于调试）
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            logger.info(f"已清理临时目录: {temp_dir}")

    logger.info("\n示例完成！")

if __name__ == "__main__":
    main()
```

## 8. 开发计划

### 8.1 阶段划分（共5个阶段）

#### 阶段一：基础框架（预计1-2天）
**目标**：建立转换模块基础架构，实现基础类和工具函数

**任务清单**：
1. 创建convert模块目录结构
2. 实现`BaseConverter`抽象基类和`ConversionResult`数据类
3. 实现基础工具函数（`utils.py`）
4. 集成`dataflow/util`模块的文件和日志工具
5. 编写基础测试框架

**验收标准**：
- 基础类设计合理，接口清晰
- 工具函数完整可用
- 测试框架可运行

#### 阶段二：核心转换器实现（预计3-4天）
**目标**：实现三个核心转换器

**任务清单**：
1. 实现`LabelMeAndYoloConverter`
   - LabelMe→YOLO转换逻辑
   - YOLO→LabelMe转换逻辑
   - 类别文件处理
   - 坐标系统验证
2. 实现`YoloAndCocoConverter`
   - YOLO→COCO转换逻辑
   - COCO→YOLO转换逻辑
   - 类别信息提取和生成
   - 图片信息处理
3. 实现`CocoAndLabelMeConverter`
   - COCO→LabelMe转换逻辑
   - LabelMe→COCO转换逻辑
   - RLE mask处理框架
4. 编写核心转换器单元测试

**验收标准**：
- 所有6种转换方向基本功能正常
- 类别处理正确
- 坐标转换准确
- 单元测试通过率≥80%

#### 阶段三：RLE支持（预计1-2天）
**目标**：实现RLE格式转换支持

**任务清单**：
1. 实现`RLEConverter`工具类
2. 集成pycocotools依赖处理
3. 实现`do_rle`参数逻辑
4. 添加RLE转换精度警告
5. 编写RLE相关测试

**验收标准**：
- RLE转换功能正常
- 可选依赖处理正确
- 精度损失警告记录完整
- RLE测试覆盖所有场景

#### 阶段四：label模块调整（预计1-2天）
**目标**：增强现有label模块以支持转换需求

**任务清单**：
1. 增强`CocoAnnotationHandler`支持`do_rle`参数
2. 添加类别提取辅助方法
3. 确保OriginalData机制在转换流程中正常工作
4. 更新相关测试用例
5. 验证向后兼容性

**验收标准**：
- label模块功能增强不影响现有使用
- OriginalData机制完整支持无损转换
- 所有现有测试仍然通过

#### 阶段五：测试和文档（预计2-3天）
**目标**：完善测试、文档和示例

**任务清单**：
1. 编写完整的单元测试套件
2. 创建使用示例（samples/convert/）
4. 集成测试和性能测试
5. 最终代码审查和质量检查

**验收标准**：
- 单元测试覆盖率≥90%
- 示例代码可正常运行
- 性能指标满足要求

### 8.2 详细任务清单

#### 阶段一详细任务
- [ ] 创建目录：`dataflow/convert/`, `tests/convert/`, `samples/convert/`
- [ ] 编写`dataflow/convert/__init__.py`导出必要接口
- [ ] 实现`BaseConverter`抽象基类
- [ ] 实现`ConversionResult`数据类
- [ ] 实现`utils.py`中的工具函数
- [ ] 编写`test_base.py`单元测试
- [ ] 编写`test_utils.py`单元测试
- [ ] 验证与现有模块的集成

#### 阶段二详细任务
- [ ] 实现`LabelMeAndYoloConverter`类
- [ ] 实现`YoloAndCocoConverter`类
- [ ] 实现`CocoAndLabelMeConverter`类
- [ ] 编写`test_labelme_and_yolo.py`单元测试
- [ ] 编写`test_yolo_and_coco.py`单元测试
- [ ] 编写`test_coco_and_labelme.py`单元测试
- [ ] 准备转换测试数据
- [ ] 验证所有6种转换方向

#### 阶段三详细任务
- [ ] 实现`RLEConverter`工具类
- [ ] 集成pycocotools可选依赖处理
- [ ] 在转换器中实现`do_rle`参数逻辑
- [ ] 添加RLE精度损失警告机制
- [ ] 编写`test_rle_converter.py`单元测试
- [ ] 进行coco rle相关操作时，如果没有检测到pycocotools，直接报错处理（ImportError）

#### 阶段四详细任务
- [ ] 增强`CocoAnnotationHandler.write()`支持`do_rle`参数
- [ ] 在label模块添加类别提取辅助方法

#### 阶段五详细任务
- [ ] 创建7个示例文件（samples/convert/）
- [ ] 确保所有测试通过

### 8.3 验收标准

#### 代码质量
- 单元测试覆盖率≥90%
- 类型提示覆盖率100%
- 文档字符串覆盖率100%
- 遵循PEP 8代码规范，通过black格式化

#### 功能完整性
- 所有6种转换方向功能完整
- 目标检测和实例分割支持完整
- RLE支持完整（包括可选依赖处理）
- 无损转换验证通过（除RLE外）
- 错误处理完善（严格和宽松模式）

#### 用户体验
- 清晰的API设计，易于使用
- 完整的示例代码，可直接运行
- 详细的错误信息和日志
- 合理的默认参数设置

#### 性能要求
- 转换100张图片的时间在合理范围内（<30秒）
- 内存使用合理，无内存泄漏
- 支持大文件处理

#### 兼容性要求
- 与现有label模块完全兼容
- 跨平台兼容（Windows、Linux、macOS）
- Python 3.8+支持

### 8.4 风险评估和缓解措施

#### 风险1：RLE转换精度损失
- **风险**：多边形↔RLE转换有精度损失，无法实现完全无损转换
- **缓解**：
  - 明确文档说明RLE转换的精度损失
  - 添加警告信息提醒用户
  - 提供`do_rle=False`选项使用多边形格式

#### 风险2：类别映射冲突
- **风险**：不同格式间类别ID或名称冲突，导致转换错误
- **缓解**：
  - 实现智能类别冲突检测和解决
  - 提供详细的警告和错误信息
  - 允许用户指定类别映射规则

#### 风险3：大文件内存使用
- **风险**：处理大型COCO JSON文件时内存占用过高
- **缓解**：
  - 实现流式读取和分批处理
  - 添加内存使用监控和警告
  - 提供处理大文件的指导文档

#### 风险4：向后兼容性破坏
- **风险**：修改label模块可能影响现有功能
- **缓解**：
  - 充分测试现有功能不受影响
  - 逐步修改，每次修改后运行完整测试套件
  - 保持API向后兼容

## 9. 质量保证

### 9.1 代码规范检查

使用以下工具确保代码质量：
1. **black**：代码格式化
2. **isort**：导入排序
3. **flake8**：代码风格检查
4. **mypy**：类型检查
5. **pylint**：代码质量分析

配置预提交钩子（pre-commit hooks）自动检查。

### 9.2 测试覆盖率要求

使用`pytest-cov`进行测试覆盖率统计：

```bash
# 运行测试并生成覆盖率报告
pytest tests/convert/ --cov=dataflow.convert --cov-report=term --cov-report=html

# 最小覆盖率要求
pytest tests/convert/ --cov=dataflow.convert --cov-fail-under=90
```

### 9.3 持续集成

配置GitHub Actions或类似CI系统，确保每次提交：
- 运行完整的测试套件
- 检查代码规范和类型提示
- 生成测试覆盖率报告
- 验证跨平台兼容性

### 9.4 发布前检查清单

发布前必须完成以下检查：
- [ ] 所有单元测试通过
- [ ] 测试覆盖率≥90%
- [ ] 类型检查通过（mypy）
- [ ] 代码风格检查通过（flake8）
- [ ] 示例代码可正常运行
- [ ] 跨平台兼容性验证通过
- [ ] 性能测试通过
- [ ] 文档完整且准确

## 总结

本规范文档为DataFlow-CV转换模块的开发提供了完整的蓝图，包括：

1. **详细的设计方案**：涵盖架构、接口、数据流、错误处理
2. **完整的实现指南**：每个转换器的具体实现要求和技术细节
3. **严格的测试策略**：确保代码质量和可靠性
4. **实用的示例代码**：帮助用户快速上手和使用
5. **可行的开发计划**：分阶段实施，降低风险
6. **全面的质量保证**：从代码规范到持续集成

转换模块的完成将使DataFlow-CV成为一个完整的计算机视觉数据处理套件，支持：
- **格式读取**：通过label模块读取多种标注格式
- **格式转换**：通过convert模块在格式间灵活转换
- **数据可视化**：通过visualize模块查看和验证标注

遵循本规范进行开发，可以确保DataFlow-CV转换模块的高质量、可维护性和易用性，为用户提供稳定可靠的计算机视觉数据格式转换工具。

---

**文档版本**：1.0
**最后更新**：2026-03-22
**作者**：Claude Code