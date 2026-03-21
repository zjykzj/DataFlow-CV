# DataFlow-CV 标签模块开发规范

## 1. 项目概述

### 1.1 项目背景和目标

DataFlow-CV是一个专门处理计算机视觉数据集的Python库，旨在提供统一的标签文件格式转换、可视化和数据处理功能。随着计算机视觉应用的普及，不同的算法和框架使用不同的标注格式（如LabelMe、YOLO、COCO等），导致数据预处理工作复杂且重复。

本项目的核心目标是实现一个标准化、模块化的标签处理系统，支持：
- 多种主流标注格式的批量读取和写入
- 目标检测和实例分割数据的统一处理
- 自动检测标注类型和格式
- 严格的跨平台兼容性
- 完整的测试和示例体系

### 1.2 核心功能特性

1. **多格式支持**：支持LabelMe、YOLO、COCO三种主流标注格式
2. **双任务支持**：同时支持目标检测和实例分割算法数据标注
3. **自动检测**：自动识别标注类型（目标检测 vs 实例分割）
4. **RLE支持**：对COCO格式支持RLE mask格式和点列表格式
5. **批量处理**：支持文件夹级别的批量读取和写入
6. **工具模块**：提供统一的文件和日志操作工具
7. **严格验证**：严格的错误处理和格式验证机制

### 1.3 设计原则

1. **统一性**：所有处理器遵循相同的接口规范
2. **类型安全**：使用Python数据类（dataclass）确保类型安全
3. **错误处理**：严格模式，遇到错误立即引发异常
4. **可扩展性**：易于添加新的标注格式支持
5. **平台兼容**：确保在Windows、Linux、macOS上的完全兼容
6. **文档完整**：完整的API文档、使用示例和测试用例

## 2. 整体架构

### 2.1 模块组织结构图

```
DataFlow-CV/
├── dataflow/
│   ├── __init__.py
│   ├── label/
│   │   ├── __init__.py
│   │   ├── base.py          # BaseAnnotationHandler抽象基类
│   │   ├── labelme_handler.py
│   │   ├── yolo_handler.py
│   │   ├── coco_handler.py
│   │   └── models.py        # 数据类定义
│   └── util/
│       ├── __init__.py
│       ├── file_util.py     # FileOperations类
│       └── logging_util.py  # LoggingOperations类
├── tests/
│   ├── __init__.py
│   ├── label/
│   │   ├── __init__.py
│   │   ├── test_base.py
│   │   ├── test_labelme.py
│   │   ├── test_yolo.py
│   │   └── test_coco.py
│   └── util/
│       ├── __init__.py
│       ├── test_file_util.py
│       └── test_logging_util.py
├── samples/
│   ├── label/
│   │   ├── labelme_demo.py
│   │   ├── yolo_demo.py
│   │   └── coco_demo.py
│   └── util/
│       ├── file_util_demo.py
│       └── logging_util_demo.py
└── assets/
    ├── test_data/
    │   ├── labelme/
    │   ├── yolo/
    │   └── coco/
    └── sample_data/
        ├── labelme/
        ├── yolo/
        └── coco/
```

### 2.2 文件树结构说明

- **dataflow/**：核心模块目录
  - `label/`：标签处理模块
  - `util/`：工具模块
- **tests/**：测试目录，与核心模块结构对应
- **samples/**：示例代码目录，提供使用范例
- **assets/**：资源目录
  - `test_data/`：测试数据
  - `sample_data/`：示例数据

### 2.3 依赖关系

```
dataflow.label
├── dataflow.util (依赖)
├── opencv-python (图像处理)
├── pycocotools (RLE处理，可选)
├── numpy (数值计算)
└── Python标准库
```

## 3. dataflow/label模块详细设计

### 3.1 基础组件

#### 3.1.1 BaseAnnotationHandler抽象基类

所有标注处理器的基类，定义统一的接口规范。

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import logging

@dataclass
class AnnotationResult:
    """标注处理结果"""
    success: bool
    data: Optional[Any] = None
    message: str = ""
    errors: List[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []

class BaseAnnotationHandler(ABC):
    """标注处理器抽象基类"""

    def __init__(self, strict_mode: bool = True, logger: Optional[logging.Logger] = None):
        self.strict_mode = strict_mode
        self.logger = logger or logging.getLogger(__name__)
        self.is_det = False  # 是否为目标检测
        self.is_seg = False  # 是否为实例分割
        self.is_rle = False  # 是否为RLE格式（仅COCO）

    @abstractmethod
    def read(self, *args, **kwargs) -> AnnotationResult:
        """读取标注文件"""
        pass

    @abstractmethod
    def write(self, *args, **kwargs) -> AnnotationResult:
        """写入标注文件"""
        pass

    @abstractmethod
    def validate(self, *args, **kwargs) -> bool:
        """验证标注格式"""
        pass

    def _log_info(self, message: str):
        """记录信息日志"""
        self.logger.info(message)

    def _log_error(self, message: str):
        """记录错误日志"""
        self.logger.error(message)
        if self.strict_mode:
            raise ValueError(message)
```

#### 3.1.2 数据类定义

```python
from dataclasses import dataclass
from typing import List, Tuple, Any, Dict
import numpy as np

@dataclass
class BoundingBox:
    """边界框"""
    x: float  # 中心点x坐标（归一化）
    y: float  # 中心点y坐标（归一化）
    width: float  # 宽度（归一化）
    height: float  # 高度（归一化）

    @property
    def xywh_abs(self, img_width: int, img_height: int) -> Tuple[int, int, int, int]:
        """转换为绝对坐标"""
        return (
            int(self.x * img_width),
            int(self.y * img_height),
            int(self.width * img_width),
            int(self.height * img_height)
        )

    @property
    def xyxy(self, img_width: int, img_height: int) -> Tuple[int, int, int, int]:
        """转换为左上-右下坐标"""
        x_center = self.x * img_width
        y_center = self.y * img_height
        w = self.width * img_width
        h = self.height * img_height
        return (
            int(x_center - w/2),
            int(y_center - h/2),
            int(x_center + w/2),
            int(y_center + h/2)
        )

@dataclass
class Segmentation:
    """分割标注"""
    points: List[Tuple[float, float]]  # 多边形点列表（归一化坐标）

    @property
    def points_abs(self, img_width: int, img_height: int) -> List[Tuple[int, int]]:
        """转换为绝对坐标"""
        return [(int(x * img_width), int(y * img_height)) for x, y in self.points]

@dataclass
class ObjectAnnotation:
    """单个对象标注"""
    class_id: int  # 类别ID
    class_name: str  # 类别名称
    bbox: Optional[BoundingBox] = None  # 边界框（目标检测）
    segmentation: Optional[Segmentation] = None  # 分割标注（实例分割）
    confidence: float = 1.0  # 置信度
    is_crowd: bool = False  # 是否为人群标注（COCO特有）

    def __post_init__(self):
        # 自动设置标注类型标志
        if self.bbox is not None:
            self.is_detection = True
        if self.segmentation is not None:
            self.is_segmentation = True

@dataclass
class ImageAnnotation:
    """单张图片的标注"""
    image_id: str  # 图片ID（文件名或唯一标识）
    image_path: str  # 图片文件路径
    width: int  # 图片宽度
    height: int  # 图片高度
    objects: List[ObjectAnnotation]  # 对象标注列表

    def __post_init__(self):
        if self.objects is None:
            self.objects = []

@dataclass
class DatasetAnnotations:
    """数据集标注集合"""
    images: List[ImageAnnotation]  # 图片标注列表
    categories: Dict[int, str]  # 类别映射（ID -> 名称）
    dataset_info: Dict[str, Any] = None  # 数据集元信息

    def __post_init__(self):
        if self.images is None:
            self.images = []
        if self.categories is None:
            self.categories = {}
        if self.dataset_info is None:
            self.dataset_info = {}
```

#### 3.1.3 统一数据格式说明

所有处理器最终都转换为`DatasetAnnotations`格式，确保内部数据统一：

1. **坐标系统**：所有坐标使用归一化值（0-1范围）
2. **类别映射**：统一的`categories: Dict[int, str]`格式
3. **多任务支持**：同时包含`bbox`和`segmentation`字段
4. **元数据**：保留原始格式的元信息

### 3.2 标签处理器设计

#### 3.2.1 LabelMeAnnotationHandler

处理LabelMe JSON格式标注文件。

**输入参数**：
- `label_dir`: 标签文件夹目录（包含多个JSON文件）
- `class_file`: 类别文件路径（可选，每行一个类别名）

**API设计**：
```python
class LabelMeAnnotationHandler(BaseAnnotationHandler):
    def __init__(self, label_dir: str, class_file: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.label_dir = label_dir
        self.class_file = class_file
        self.categories = self._load_categories()

    def read(self) -> AnnotationResult:
        """读取LabelMe标注文件夹"""
        # 实现细节...

    def write(self, annotations: DatasetAnnotations, output_dir: str) -> AnnotationResult:
        """将标注写入LabelMe格式"""
        # 实现细节...

    def validate(self, annotation_file: str) -> bool:
        """验证单个LabelMe JSON文件"""
        # 实现细节...
```

**特性**：
1. 支持批量读取文件夹中的所有JSON文件
2. 自动从标注文件中提取类别信息
3. 支持多边形（分割）和矩形（检测）标注
4. 严格验证JSON格式和必需字段

#### 3.2.2 YoloAnnotationHandler

处理YOLO格式标注文件。

**输入参数**：
- `label_dir`: 标签文件夹目录（包含多个.txt文件）
- `class_file`: 类别文件路径（必需，每行一个类别名）
- `image_dir`: 图片文件夹目录（用于获取图片尺寸）

**API设计**：
```python
class YoloAnnotationHandler(BaseAnnotationHandler):
    def __init__(self, label_dir: str, class_file: str, image_dir: str, **kwargs):
        super().__init__(**kwargs)
        self.label_dir = label_dir
        self.class_file = class_file
        self.image_dir = image_dir
        self.categories = self._load_categories()

    def read(self) -> AnnotationResult:
        """读取YOLO标注文件夹"""
        # 实现细节...

    def write(self, annotations: DatasetAnnotations, output_dir: str) -> AnnotationResult:
        """将标注写入YOLO格式"""
        # 实现细节...

    def validate(self, annotation_file: str) -> bool:
        """验证单个YOLO TXT文件"""
        # 实现细节...
```

**特性**：
1. 自动检测标注类型（目标检测 vs 实例分割）
2. 支持归一化坐标验证（0-1范围）
3. 自动匹配图片和标签文件
4. 严格的格式验证（每行5个值表示检测，多个值表示分割）

#### 3.2.3 CocoAnnotationHandler

处理COCO格式标注文件。

**输入参数**：
- `annotation_file`: 单个COCO标签文件路径（JSON格式）

**API设计**：
```python
class CocoAnnotationHandler(BaseAnnotationHandler):
    def __init__(self, annotation_file: str, **kwargs):
        super().__init__(**kwargs)
        self.annotation_file = annotation_file

    def read(self) -> AnnotationResult:
        """读取COCO标注文件"""
        # 实现细节...

    def write(self, annotations: DatasetAnnotations, output_file: str) -> AnnotationResult:
        """将标注写入COCO格式"""
        # 实现细节...

    def validate(self) -> bool:
        """验证COCO JSON文件"""
        # 实现细节...
```

**特性**：
1. 支持标准COCO JSON格式
2. 自动检测RLE格式和点列表格式
3. 支持`is_crowd`标志处理
4. 完整的类别和图片信息提取

### 3.3 自动检测机制

#### 3.3.1 YOLO格式检测逻辑

YOLO处理器需要自动判断标注类型：

```python
def _detect_annotation_type(self, line_items: List[str]) -> Tuple[bool, bool]:
    """
    检测标注类型

    Args:
        line_items: 分割后的行数据项

    Returns:
        Tuple[is_detection, is_segmentation]
    """
    if len(line_items) == 5:
        # 目标检测格式: class_id x_center y_center width height
        return True, False
    elif len(line_items) > 5 and len(line_items) % 2 == 1:
        # 实例分割格式: class_id x1 y1 x2 y2 ... xn yn
        # 第一个是class_id，后面是成对的x,y坐标
        return False, True
    else:
        raise ValueError(f"Invalid YOLO format: {len(line_items)} items")
```

#### 3.3.2 标志变量使用方式

处理器级别的标志设置：

```python
def _set_annotation_flags(self, annotations: DatasetAnnotations):
    """根据标注数据设置处理器标志"""
    has_detection = any(
        obj.bbox is not None
        for img in annotations.images
        for obj in img.objects
    )
    has_segmentation = any(
        obj.segmentation is not None
        for img in annotations.images
        for obj in img.objects
    )

    self.is_det = has_detection
    self.is_seg = has_segmentation

    # 记录日志
    if self.is_det and self.is_seg:
        self._log_info("检测到混合标注类型：目标检测 + 实例分割")
    elif self.is_det:
        self._log_info("检测到目标检测标注")
    elif self.is_seg:
        self._log_info("检测到实例分割标注")
```

### 3.4 RLE支持

#### 3.4.1 pycocotools集成方案

```python
try:
    from pycocotools import mask as coco_mask
    HAS_COCO_MASK = True
except ImportError:
    HAS_COCO_MASK = False
    self._log_warning("pycocotools not installed, RLE support disabled")

class CocoAnnotationHandler(BaseAnnotationHandler):
    def __init__(self, annotation_file: str, **kwargs):
        super().__init__(**kwargs)
        self.annotation_file = annotation_file
        self.is_rle = False

    def _detect_rle_format(self, annotations: List[Dict]) -> bool:
        """检测是否包含RLE格式标注"""
        for ann in annotations:
            if 'segmentation' in ann:
                if isinstance(ann['segmentation'], dict) and 'counts' in ann['segmentation']:
                    return True
        return False
```

#### 3.4.2 RLE编码/解码流程

```python
def _decode_rle_to_polygon(self, rle: Dict, img_width: int, img_height: int) -> List[Tuple[float, float]]:
    """将RLE解码为多边形点列表"""
    if not HAS_COCO_MASK:
        raise ImportError("pycocotools required for RLE decoding")

    # 解码RLE为二值掩码
    binary_mask = coco_mask.decode(rle)

    # 从掩码提取轮廓
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

def _encode_polygon_to_rle(self, points: List[Tuple[float, float]],
                          img_width: int, img_height: int) -> Dict:
    """将多边形编码为RLE格式"""
    if not HAS_COCO_MASK:
        raise ImportError("pycocotools required for RLE encoding")

    # 将归一化坐标转换为绝对坐标
    abs_points = [(int(x * img_width), int(y * img_height)) for x, y in points]

    # 创建二值掩码
    mask = np.zeros((img_height, img_width), dtype=np.uint8)
    contour = np.array(abs_points, dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [contour], 1)

    # 编码为RLE
    rle = coco_mask.encode(np.asfortranarray(mask))

    return rle
```

#### 3.4.3 与多边形格式的互转换

COCO处理器应支持两种格式：
1. **多边形格式**：点列表，适合人工编辑
2. **RLE格式**：压缩格式，适合存储和计算

处理器自动检测并统一转换为多边形格式进行内部处理，写入时根据`is_rle`标志选择输出格式。

## 4. dataflow/util模块详细设计

### 4.1 file_util.py - FileOperations类

提供跨平台文件操作工具。

```python
import os
import shutil
import tempfile
from pathlib import Path
from typing import List, Tuple, Optional, Union
import logging

class FileOperations:
    """跨平台文件操作工具类"""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

    def ensure_dir(self, dir_path: Union[str, Path]) -> Path:
        """确保目录存在，如果不存在则创建"""
        path = Path(dir_path)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Created directory: {path}")
        return path

    def safe_remove(self, file_path: Union[str, Path]) -> bool:
        """安全删除文件，如果文件存在"""
        path = Path(file_path)
        if path.exists():
            try:
                if path.is_file():
                    path.unlink()
                else:
                    shutil.rmtree(path)
                self.logger.info(f"Removed: {path}")
                return True
            except Exception as e:
                self.logger.error(f"Failed to remove {path}: {e}")
                return False
        return False

    def copy_files(self, src_pattern: str, dst_dir: Union[str, Path],
                  overwrite: bool = False) -> List[Tuple[Path, bool]]:
        """批量复制文件"""
        results = []
        dst_path = Path(dst_dir)
        self.ensure_dir(dst_path)

        for src_file in Path().glob(src_pattern):
            dst_file = dst_path / src_file.name
            if dst_file.exists() and not overwrite:
                self.logger.warning(f"File exists, skipping: {dst_file}")
                results.append((dst_file, False))
                continue

            try:
                shutil.copy2(src_file, dst_file)
                results.append((dst_file, True))
                self.logger.info(f"Copied: {src_file} -> {dst_file}")
            except Exception as e:
                self.logger.error(f"Failed to copy {src_file}: {e}")
                results.append((dst_file, False))

        return results

    def find_files(self, directory: Union[str, Path],
                  pattern: str = "*", recursive: bool = True) -> List[Path]:
        """查找匹配模式的文件"""
        dir_path = Path(directory)
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        if recursive:
            return list(dir_path.rglob(pattern))
        else:
            return list(dir_path.glob(pattern))

    def create_temp_dir(self, prefix: str = "dataflow_") -> Path:
        """创建临时目录"""
        temp_dir = tempfile.mkdtemp(prefix=prefix)
        self.logger.info(f"Created temp directory: {temp_dir}")
        return Path(temp_dir)

    def get_file_size(self, file_path: Union[str, Path]) -> int:
        """获取文件大小（字节）"""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        return path.stat().st_size

    def read_lines(self, file_path: Union[str, Path],
                  encoding: str = "utf-8") -> List[str]:
        """读取文件所有行"""
        path = Path(file_path)
        with open(path, 'r', encoding=encoding) as f:
            return [line.strip() for line in f.readlines()]

    def write_lines(self, file_path: Union[str, Path],
                   lines: List[str], encoding: str = "utf-8") -> bool:
        """写入多行文本"""
        path = Path(file_path)
        try:
            with open(path, 'w', encoding=encoding) as f:
                f.write('\n'.join(lines))
            self.logger.info(f"Written {len(lines)} lines to: {path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to write to {path}: {e}")
            return False
```

### 4.2 logging_util.py - LoggingOperations类

提供统一的日志配置。

```python
import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import datetime

class LoggingOperations:
    """日志操作工具类"""

    DEFAULT_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    DEFAULT_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

    def __init__(self):
        self.loggers = {}

    def get_logger(self, name: str = "dataflow",
                  level: int = logging.INFO,
                  log_file: Optional[str] = None,
                  console: bool = True) -> logging.Logger:
        """获取配置好的日志器"""
        if name in self.loggers:
            return self.loggers[name]

        logger = logging.getLogger(name)
        logger.setLevel(level)

        # 清除现有处理器
        logger.handlers.clear()

        # 创建格式化器
        formatter = logging.Formatter(
            self.DEFAULT_FORMAT,
            datefmt=self.DEFAULT_DATE_FORMAT
        )

        # 控制台处理器
        if console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        # 文件处理器
        if log_file:
            self._ensure_log_dir(log_file)
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        self.loggers[name] = logger
        return logger

    def _ensure_log_dir(self, log_file: str):
        """确保日志目录存在"""
        log_path = Path(log_file)
        if log_path.parent:
            log_path.parent.mkdir(parents=True, exist_ok=True)

    def setup_root_logger(self, level: int = logging.INFO,
                         log_file: Optional[str] = None):
        """设置根日志器配置"""
        handlers = []

        # 控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)

        # 文件处理器
        if log_file:
            self._ensure_log_dir(log_file)
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(level)
            handlers.append(file_handler)

        handlers.append(console_handler)

        # 配置根日志器
        logging.basicConfig(
            level=level,
            format=self.DEFAULT_FORMAT,
            datefmt=self.DEFAULT_DATE_FORMAT,
            handlers=handlers
        )

    def create_log_file(self, base_name: str = "dataflow",
                       directory: str = "./logs") -> str:
        """创建带时间戳的日志文件路径"""
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = dir_path / f"{base_name}_{timestamp}.log"

        return str(log_file)

    def set_log_level(self, logger_name: str, level: str):
        """设置日志级别"""
        level_map = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR,
            'CRITICAL': logging.CRITICAL
        }

        if level not in level_map:
            raise ValueError(f"Invalid log level: {level}")

        logger = logging.getLogger(logger_name)
        logger.setLevel(level_map[level])

        # 更新所有处理器的级别
        for handler in logger.handlers:
            handler.setLevel(level_map[level])
```

## 5. 技术实现要求

### 5.1 核心依赖

**必需依赖**：
- `numpy>=1.24.0`：数值计算
- `opencv-python>=4.6.0.66`：图像处理和轮廓提取
- `click>=7.0.0`：命令行界面（未来扩展）

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

### 5.3 错误处理策略

1. **严格模式**：默认启用，遇到错误立即引发异常
2. **详细错误信息**：包含上下文信息的错误消息
3. **类型验证**：使用类型提示和运行时验证
4. **资源清理**：确保文件句柄、临时目录的正确清理

### 5.4 性能优化

1. **批量处理**：支持文件夹级别的批量操作
2. **惰性加载**：大文件支持流式读取
3. **内存管理**：及时释放大对象内存
4. **缓存机制**：适当缓存频繁访问的数据

### 5.5 代码质量要求

1. **类型提示**：所有公共API必须有完整的类型提示
2. **文档字符串**：所有类和方法必须有完整的docstring
3. **代码风格**：遵循PEP 8规范
4. **单元测试**：核心功能必须有单元测试覆盖

## 6. 测试策略

### 6.1 测试目录结构

```
tests/
├── __init__.py
├── conftest.py           # 测试配置和fixture
├── label/
│   ├── __init__.py
│   ├── test_base.py      # 基类测试
│   ├── test_labelme.py   # LabelMe处理器测试
│   ├── test_yolo.py      # YOLO处理器测试
│   └── test_coco.py      # COCO处理器测试
└── util/
    ├── __init__.py
    ├── test_file_util.py    # 文件工具测试
    └── test_logging_util.py # 日志工具测试
```

### 6.2 测试数据准备

在`assets/test_data/`目录下准备测试数据：

```
assets/test_data/
├── labelme/
│   ├── classes.txt
│   ├── image1.jpg
│   ├── image1.json
│   ├── image2.jpg
│   └── image2.json
├── yolo/
│   ├── classes.txt
│   ├── images/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   └── labels/
│       ├── image1.txt
│       └── image2.txt
└── coco/
    ├── annotations.json
    └── images/
        ├── image1.jpg
        └── image2.jpg
```

### 6.3 测试编写规范

1. **测试类命名**：`Test{ClassName}`
2. **测试方法命名**：`test_{function_name}_{scenario}`
3. **测试覆盖**：正常流程、边界条件、异常情况
4. **测试隔离**：每个测试独立，不依赖执行顺序
5. **测试清理**：测试后清理创建的临时文件

### 6.4 测试用例示例

```python
import pytest
from pathlib import Path
from dataflow.label import LabelMeAnnotationHandler
from dataflow.util import FileOperations

class TestLabelMeAnnotationHandler:
    """LabelMe处理器测试"""

    @pytest.fixture
    def labelme_handler(self, tmp_path):
        """创建测试用的LabelMe处理器"""
        # 创建测试数据
        test_data_dir = tmp_path / "labelme_test"
        test_data_dir.mkdir()

        # 复制或创建测试文件
        # ...

        handler = LabelMeAnnotationHandler(
            label_dir=str(test_data_dir),
            strict_mode=True
        )
        return handler

    def test_read_success(self, labelme_handler):
        """测试成功读取LabelMe标注"""
        result = labelme_handler.read()

        assert result.success is True
        assert result.data is not None
        assert len(result.data.images) > 0
        assert len(result.data.categories) > 0

    def test_read_invalid_json(self, labelme_handler, tmp_path):
        """测试读取无效JSON文件"""
        invalid_file = tmp_path / "invalid.json"
        invalid_file.write_text("{invalid json}")

        with pytest.raises(ValueError):
            labelme_handler.read()

    def test_write_success(self, labelme_handler, tmp_path):
        """测试成功写入LabelMe标注"""
        # 先读取测试数据
        read_result = labelme_handler.read()
        assert read_result.success is True

        # 写入到新目录
        output_dir = tmp_path / "output"
        write_result = labelme_handler.write(read_result.data, str(output_dir))

        assert write_result.success is True
        assert output_dir.exists()
        assert len(list(output_dir.glob("*.json"))) > 0
```

### 6.5 测试覆盖率目标

- 总体覆盖率：≥90%
- 核心模块覆盖率：≥95%
- 异常处理覆盖率：≥85%

## 7. 示例代码

### 7.1 示例目录结构

```
samples/
├── __init__.py
├── label/
│   ├── __init__.py
│   ├── labelme_demo.py    # LabelMe使用示例
│   ├── yolo_demo.py       # YOLO使用示例
│   └── coco_demo.py       # COCO使用示例
└── util/
    ├── __init__.py
    ├── file_util_demo.py     # 文件工具使用示例
    └── logging_util_demo.py  # 日志工具使用示例
```

### 7.2 示例代码编写规范

1. **完整性**：展示完整的使用流程
2. **简洁性**：聚焦核心功能，避免冗余代码
3. **注释**：详细的步骤说明和注意事项
4. **错误处理**：展示正确的错误处理方式
5. **可执行性**：确保示例可以直接运行

### 7.3 LabelMe使用示例

```python
#!/usr/bin/env python3
"""
LabelMe标注格式处理示例

展示如何使用LabelMeAnnotationHandler读取、处理和写入LabelMe格式标注。
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dataflow.label import LabelMeAnnotationHandler
from dataflow.util import LoggingOperations

def main():
    """主函数"""
    # 配置日志
    log_ops = LoggingOperations()
    logger = log_ops.get_logger("labelme_demo", level="INFO")

    # 示例数据路径（假设在assets/sample_data/labelme目录下）
    data_dir = project_root / "assets" / "sample_data" / "labelme"
    class_file = data_dir / "classes.txt"

    if not data_dir.exists():
        logger.error(f"数据目录不存在: {data_dir}")
        logger.info("请确保已准备示例数据")
        return

    logger.info("=" * 50)
    logger.info("LabelMe标注处理示例")
    logger.info("=" * 50)

    # 创建LabelMe处理器
    logger.info(f"创建LabelMe处理器，数据目录: {data_dir}")
    handler = LabelMeAnnotationHandler(
        label_dir=str(data_dir),
        class_file=str(class_file) if class_file.exists() else None,
        strict_mode=True,
        logger=logger
    )

    # 读取标注数据
    logger.info("正在读取LabelMe标注...")
    result = handler.read()

    if not result.success:
        logger.error(f"读取失败: {result.message}")
        if result.errors:
            for error in result.errors:
                logger.error(f"  - {error}")
        return

    logger.info(f"成功读取 {len(result.data.images)} 张图片的标注")
    logger.info(f"类别数量: {len(result.data.categories)}")

    # 显示部分信息
    for i, image_ann in enumerate(result.data.images[:3]):  # 只显示前3张
        logger.info(f"\n图片 {i+1}: {image_ann.image_id}")
        logger.info(f"  路径: {image_ann.image_path}")
        logger.info(f"  尺寸: {image_ann.width}x{image_ann.height}")
        logger.info(f"  对象数量: {len(image_ann.objects)}")

        for j, obj in enumerate(image_ann.objects[:2]):  # 每张图片显示前2个对象
            logger.info(f"    对象 {j+1}: {obj.class_name} (ID: {obj.class_id})")
            if obj.bbox:
                logger.info(f"      边界框: x={obj.bbox.x:.3f}, y={obj.bbox.y:.3f}, "
                          f"w={obj.bbox.width:.3f}, h={obj.bbox.height:.3f}")

    logger.info("\n示例完成！")

if __name__ == "__main__":
    main()
```

### 7.4 YOLO使用示例

```python
#!/usr/bin/env python3
"""
YOLO标注格式处理示例

展示如何使用YoloAnnotationHandler读取、处理和写入YOLO格式标注。
支持自动检测目标检测和实例分割格式。
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dataflow.label import YoloAnnotationHandler
from dataflow.util import LoggingOperations

def main():
    """主函数"""
    # 配置日志
    log_ops = LoggingOperations()
    logger = log_ops.get_logger("yolo_demo", level="INFO")

    # 示例数据路径
    data_dir = project_root / "assets" / "sample_data" / "yolo"
    class_file = data_dir / "classes.txt"
    image_dir = data_dir / "images"
    label_dir = data_dir / "labels"

    if not data_dir.exists():
        logger.error(f"数据目录不存在: {data_dir}")
        logger.info("请确保已准备示例数据")
        return

    logger.info("=" * 50)
    logger.info("YOLO标注处理示例")
    logger.info("=" * 50)

    # 创建YOLO处理器
    logger.info(f"创建YOLO处理器:")
    logger.info(f"  标签目录: {label_dir}")
    logger.info(f"  类别文件: {class_file}")
    logger.info(f"  图片目录: {image_dir}")

    handler = YoloAnnotationHandler(
        label_dir=str(label_dir),
        class_file=str(class_file),
        image_dir=str(image_dir),
        strict_mode=True,
        logger=logger
    )

    # 读取标注数据
    logger.info("\n正在读取YOLO标注...")
    result = handler.read()

    if not result.success:
        logger.error(f"读取失败: {result.message}")
        if result.errors:
            for error in result.errors:
                logger.error(f"  - {error}")
        return

    # 显示标注类型检测结果
    logger.info(f"标注类型检测:")
    logger.info(f"  目标检测: {handler.is_det}")
    logger.info(f"  实例分割: {handler.is_seg}")

    logger.info(f"\n成功读取 {len(result.data.images)} 张图片的标注")
    logger.info(f"类别数量: {len(result.data.categories)}")

    # 格式转换示例：YOLO -> 统一格式 -> 新的YOLO
    logger.info("\n进行格式转换测试...")
    output_dir = data_dir / "output"
    output_dir.mkdir(exist_ok=True)

    write_result = handler.write(result.data, str(output_dir))

    if write_result.success:
        logger.info(f"成功写入到: {output_dir}")
        logger.info(f"生成文件数量: {len(list(output_dir.glob('*.txt')))}")
    else:
        logger.error(f"写入失败: {write_result.message}")

    logger.info("\n示例完成！")

if __name__ == "__main__":
    main()
```

### 7.5 COCO使用示例

```python
#!/usr/bin/env python3
"""
COCO标注格式处理示例

展示如何使用CocoAnnotationHandler读取、处理和写入COCO格式标注。
支持RLE格式和多边形格式。
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dataflow.label import CocoAnnotationHandler
from dataflow.util import LoggingOperations

def main():
    """主函数"""
    # 配置日志
    log_ops = LoggingOperations()
    logger = log_ops.get_logger("coco_demo", level="INFO")

    # 示例数据路径
    data_dir = project_root / "assets" / "sample_data" / "coco"
    annotation_file = data_dir / "annotations.json"

    if not annotation_file.exists():
        logger.error(f"标注文件不存在: {annotation_file}")
        logger.info("请确保已准备示例数据")
        return

    logger.info("=" * 50)
    logger.info("COCO标注处理示例")
    logger.info("=" * 50)

    # 创建COCO处理器
    logger.info(f"创建COCO处理器，标注文件: {annotation_file}")
    handler = CocoAnnotationHandler(
        annotation_file=str(annotation_file),
        strict_mode=True,
        logger=logger
    )

    # 读取标注数据
    logger.info("正在读取COCO标注...")
    result = handler.read()

    if not result.success:
        logger.error(f"读取失败: {result.message}")
        if result.errors:
            for error in result.errors:
                logger.error(f"  - {error}")
        return

    # 显示RLE格式检测结果
    logger.info(f"RLE格式检测: {handler.is_rle}")

    logger.info(f"\n成功读取标注信息:")
    logger.info(f"  图片数量: {len(result.data.images)}")
    logger.info(f"  标注数量: {sum(len(img.objects) for img in result.data.images)}")
    logger.info(f"  类别数量: {len(result.data.categories)}")

    # 显示数据集信息
    if result.data.dataset_info:
        logger.info("\n数据集信息:")
        for key, value in result.data.dataset_info.items():
            logger.info(f"  {key}: {value}")

    # 格式转换示例
    logger.info("\n进行格式转换测试...")
    output_file = data_dir / "annotations_converted.json"

    write_result = handler.write(result.data, str(output_file))

    if write_result.success:
        logger.info(f"成功写入到: {output_file}")

        # 验证写入的文件
        logger.info("验证写入的文件...")
        verify_handler = CocoAnnotationHandler(
            annotation_file=str(output_file),
            strict_mode=False  # 验证时使用非严格模式
        )
        verify_result = verify_handler.read()

        if verify_result.success:
            logger.info("验证通过！写入的文件可以正确读取。")
        else:
            logger.warning(f"验证警告: {verify_result.message}")
    else:
        logger.error(f"写入失败: {write_result.message}")

    logger.info("\n示例完成！")

if __name__ == "__main__":
    main()
```

## 8. 开发计划

### 8.1 阶段划分（共4个阶段）

#### 阶段一：基础架构和工具模块（预计3-4天）
**目标**：建立项目基础架构，实现工具模块

**任务清单**：
1. 创建项目目录结构
2. 实现`dataflow/util/file_util.py` - `FileOperations`类
3. 实现`dataflow/util/logging_util.py` - `LoggingOperations`类
4. 编写工具模块的单元测试
5. 编写工具模块的使用示例
6. 更新`requirements.txt`文件

**验收标准**：
- 工具模块功能完整，测试通过
- 跨平台兼容性验证通过
- 示例代码可以正常运行

#### 阶段二：基础数据模型和LabelMe处理器（预计3-4天）
**目标**：实现基础数据模型和LabelMe格式支持

**任务清单**：
1. 实现`dataflow/label/models.py` - 数据类定义
2. 实现`dataflow/label/base.py` - `BaseAnnotationHandler`抽象基类
3. 实现`dataflow/label/labelme_handler.py` - LabelMe处理器
4. 编写LabelMe处理器的单元测试
5. 准备LabelMe测试数据
6. 编写LabelMe使用示例

**验收标准**：
- LabelMe格式读写功能完整
- 数据模型设计合理，类型安全
- 单元测试覆盖率≥90%

#### 阶段三：YOLO处理器和自动检测（预计3-4天）
**目标**：实现YOLO格式支持和自动检测机制

**任务清单**：
1. 实现`dataflow/label/yolo_handler.py` - YOLO处理器
2. 实现自动检测机制（目标检测 vs 实例分割）
3. 编写YOLO处理器的单元测试
4. 准备YOLO测试数据（检测和分割两种格式）
5. 编写YOLO使用示例
6. 优化错误处理和验证逻辑

**验收标准**：
- YOLO格式读写功能完整
- 自动检测机制准确可靠
- 支持混合标注类型处理

#### 阶段四：COCO处理器和RLE支持（预计4-5天）
**目标**：实现COCO格式支持和RLE处理

**任务清单**：
1. 实现`dataflow/label/coco_handler.py` - COCO处理器
2. 集成pycocotools，实现RLE编码/解码
3. 实现RLE与多边形格式的互转换
4. 编写COCO处理器的单元测试
5. 准备COCO测试数据（含RLE格式）
6. 编写COCO使用示例
7. 整体测试和文档完善

**验收标准**：
- COCO格式读写功能完整
- RLE支持完整，正确处理人群标注
- 可选依赖处理正确（pycocotools未安装时的降级处理）

### 8.2 详细任务清单

#### 阶段一详细任务
- [ ] 创建项目目录：`dataflow/`, `tests/`, `samples/`
- [ ] 编写`dataflow/__init__.py`和子模块`__init__.py`
- [ ] 实现`FileOperations`类的所有方法
- [ ] 实现`LoggingOperations`类的所有方法
- [ ] 编写`test_file_util.py`单元测试
- [ ] 编写`test_logging_util.py`单元测试
- [ ] 创建`file_util_demo.py`示例
- [ ] 创建`logging_util_demo.py`示例
- [ ] 验证跨平台兼容性
- [ ] 更新`requirements.txt`和`setup.py`

#### 阶段二详细任务
- [ ] 设计并实现所有数据类
- [ ] 实现`BaseAnnotationHandler`抽象基类
- [ ] 实现`LabelMeAnnotationHandler`类
- [ ] 编写`test_base.py`单元测试
- [ ] 编写`test_labelme.py`单元测试
- [ ] 准备LabelMe测试数据
- [ ] 创建`labelme_demo.py`示例
- [ ] 验证数据模型设计合理性
- [ ] 确保类型提示完整

#### 阶段三详细任务
- [ ] 实现`YoloAnnotationHandler`类
- [ ] 实现自动检测算法
- [ ] 编写`test_yolo.py`单元测试
- [ ] 准备YOLO测试数据（检测格式）
- [ ] 准备YOLO测试数据（分割格式）
- [ ] 创建`yolo_demo.py`示例
- [ ] 测试自动检测准确性
- [ ] 优化性能和大文件处理

#### 阶段四详细任务
- [ ] 实现`CocoAnnotationHandler`类
- [ ] 集成pycocotools，处理RLE格式
- [ ] 编写`test_coco.py`单元测试
- [ ] 准备COCO测试数据（标准格式）
- [ ] 准备COCO测试数据（RLE格式）
- [ ] 创建`coco_demo.py`示例
- [ ] 测试可选依赖处理
- [ ] 整体集成测试
- [ ] 完善文档和注释

### 8.3 验收标准

#### 代码质量
- 单元测试覆盖率≥90%
- 类型提示覆盖率100%
- 文档字符串覆盖率100%
- 遵循PEP 8代码规范

#### 功能完整性
- 三种格式的读写功能完整
- 自动检测机制准确
- RLE支持完整
- 错误处理完善

#### 用户体验
- 清晰的API设计
- 完整的示例代码
- 详细的错误信息
- 友好的日志输出

#### 跨平台兼容性
- Windows、Linux、macOS测试通过
- 路径处理正确
- 编码处理正确
- 权限处理正确

### 8.4 风险评估和缓解措施

#### 风险1：RLE处理复杂性
- **风险**：pycocotools安装复杂，RLE编解码容易出错
- **缓解**：
  - 提供详细的使用说明
  - 实现降级处理（pycocotools未安装时禁用RLE功能）
  - 提供RLE与多边形格式的互转换工具

#### 风险2：跨平台兼容性问题
- **风险**：不同操作系统下的路径、编码、权限问题
- **缓解**：
  - 统一使用`pathlib.Path`处理路径
  - 强制使用UTF-8编码
  - 充分的跨平台测试

#### 风险3：性能问题
- **风险**：大文件或大批量处理时内存占用过高
- **缓解**：
  - 实现流式读取和分批处理
  - 提供内存使用警告
  - 优化数据结构，减少内存占用

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
pytest --cov=dataflow --cov-report=term --cov-report=html

# 最小覆盖率要求
pytest --cov=dataflow --cov-fail-under=90
```

### 9.3 文档完整性要求

1. **API文档**：使用Sphinx或pdoc自动生成
2. **使用示例**：每个模块必须有完整的使用示例
3. **开发指南**：包含设置、开发、测试指南
4. **故障排除**：常见问题解决方案

### 9.4 发布前检查清单

发布前必须完成以下检查：

- [ ] 所有单元测试通过
- [ ] 测试覆盖率≥90%
- [ ] 类型检查通过（mypy）
- [ ] 代码风格检查通过（flake8）
- [ ] 文档完整且更新
- [ ] 示例代码可正常运行
- [ ] 跨平台兼容性验证通过
- [ ] 性能测试通过
- [ ] 安全扫描通过

### 9.5 持续集成配置

配置GitHub Actions实现自动化测试：

```yaml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
        python-version: ['3.8', '3.9', '3.10', '3.11']

    steps:
    - uses: actions/checkout@v2

    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-cov black isort flake8 mypy

    - name: Code style check
      run: |
        black --check .
        isort --check .
        flake8 .

    - name: Type check
      run: mypy dataflow/

    - name: Run tests
      run: |
        pytest --cov=dataflow --cov-report=xml

    - name: Upload coverage
      uses: codecov/codecov-action@v2
      with:
        file: ./coverage.xml
```

## 总结

本规范文档为DataFlow-CV标签模块的开发提供了完整的蓝图，包括：

1. **详细的设计方案**：涵盖架构、接口、数据模型
2. **完整的实现指南**：每个模块的具体实现要求
3. **严格的测试策略**：确保代码质量和可靠性
4. **实用的示例代码**：帮助用户快速上手
5. **可行的开发计划**：分阶段实施，降低风险
6. **全面的质量保证**：从代码规范到持续集成

遵循本规范进行开发，可以确保DataFlow-CV标签模块的高质量、可维护性和易用性，为用户提供稳定可靠的计算机视觉数据处理工具。

---

**文档版本**：1.0
**最后更新**：2026-03-21
**作者**：Claude Code
**状态**：草案（待评审）