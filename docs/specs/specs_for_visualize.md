# DataFlow-CV 可视化模块开发规范

## 1. 项目概述

### 1.1 项目背景和目标

DataFlow-CV已经完成了dataflow/label模块的开发，支持LabelMe、YOLO、COCO格式的标签文件读写和格式转换。dataflow/util模块提供了文件操作和日志记录工具。现在需要开发dataflow/visualize模块，用于可视化标签文件中的标注结果。

本模块的核心目标是实现一个标准化、模块化的可视化系统，支持：
- 多种主流标注格式的可视化（LabelMe、YOLO、COCO）
- 目标检测和实例分割标注的统一绘制
- 自动识别标注类型并选择相应绘制方式
- 高质量的颜色管理和可视化效果
- 灵活的输出方式（显示窗口或保存图像）
- 严格的跨平台兼容性
- 完整的测试和示例体系

### 1.2 核心功能特性

1. **多格式支持**：支持LabelMe、YOLO、COCO三种主流标注格式的可视化
2. **双任务支持**：同时支持目标检测边界框和实例分割多边形的绘制
3. **RLE支持**：对COCO格式的RLE mask支持解码和绘制
4. **颜色管理**：自动分配高对比度颜色，确保同一类别颜色一致
5. **交互模式**：支持窗口显示（按Enter键下一张，按q键退出）
6. **保存模式**：支持将可视化结果保存为JPEG图像文件
7. **批量处理**：支持文件夹级别的批量可视化
8. **严格验证**：严格的错误处理和格式验证机制

### 1.3 设计原则

1. **统一性**：所有可视化器遵循相同的接口规范
2. **复用性**：充分利用已有的label模块和util模块
3. **类型安全**：使用Python类型提示确保类型安全
4. **错误处理**：严格模式，遇到错误立即引发异常
5. **可扩展性**：易于添加新的可视化功能或格式支持
6. **平台兼容**：确保在Windows、Linux、macOS上的完全兼容

## 2. 整体架构

### 2.1 模块组织结构图

```
DataFlow-CV/
├── dataflow/
│   ├── __init__.py
│   ├── label/              # 已存在的标签处理模块
│   ├── util/               # 已存在的工具模块
│   └── visualize/          # 新增可视化模块
│       ├── __init__.py
│       ├── base.py          # BaseVisualizer抽象基类
│       ├── labelme_visualizer.py
│       ├── yolo_visualizer.py
│       ├── coco_visualizer.py
│       └── utils.py         # 可视化工具函数
├── tests/
│   ├── __init__.py
│   └── visualize/
│       ├── __init__.py
│       ├── test_base.py
│       ├── test_labelme_visualizer.py
│       ├── test_yolo_visualizer.py
│       └── test_coco_visualizer.py
├── samples/
│   └── visualize/
│       ├── __init__.py
│       ├── labelme_demo.py
│       ├── yolo_demo.py
│       └── coco_demo.py
└── assets/
    └── test_data/          # 现有的测试数据
        ├── det
        └── seg
```

### 2.2 文件树结构说明

- **dataflow/visualize/**：核心可视化模块目录
- **tests/visualize/**：可视化模块测试目录
- **samples/visualize/**：可视化示例代码目录
- **assets/test_data/**：现有的测试数据（无需新增）

### 2.3 依赖关系

```
dataflow.visualize
├── dataflow.label (依赖，用于读取标注)
├── dataflow.util (依赖，用于文件操作和日志)
├── opencv-python (必需，图像处理和可视化)
├── pycocotools (可选，RLE掩码解码)
└── Python标准库
```

### 2.4 数据流

1. **输入**：
   - 标签文件目录/路径（根据格式不同）
   - 图像文件目录
   - 配置参数（is_show, is_save, output_dir等）

2. **处理流程**：
   - 通过对应的handler读取标签数据 → DatasetAnnotations
   - 加载对应的图像文件
   - 根据标注类型进行可视化绘制：
     - 目标检测：绘制边界框 + 类别标签
     - 实例分割：绘制多边形轮廓 + 半透明填充
     - RLE掩码：解码后绘制半透明掩码
   - 颜色管理：确保同一类别颜色一致

3. **输出**：
   - is_show模式：在OpenCV窗口中显示，支持键盘交互
   - is_save模式：保存为JPEG格式图像文件到指定目录

## 3. dataflow/visualize模块详细设计

### 3.1 基础组件

#### 3.1.1 BaseVisualizer抽象基类

所有可视化器的基类，定义统一的接口规范。

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import logging

import cv2

from dataflow.label.models import DatasetAnnotations, ImageAnnotation, ObjectAnnotation
from dataflow.util import FileOperations, LoggingOperations

@dataclass
class VisualizationResult:
    """可视化处理结果"""
    success: bool
    data: Optional[Any] = None
    message: str = ""
    errors: List[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []

class BaseVisualizer(ABC):
    """可视化器抽象基类"""

    def __init__(self,
                 label_dir: Union[str, Path],
                 image_dir: Union[str, Path],
                 output_dir: Optional[Union[str, Path]] = None,
                 is_show: bool = True,
                 is_save: bool = False,
                 strict_mode: bool = True,
                 logger: Optional[logging.Logger] = None):
        self.label_dir = Path(label_dir)
        self.image_dir = Path(image_dir)
        self.output_dir = Path(output_dir) if output_dir else None
        self.is_show = is_show
        self.is_save = is_save
        self.strict_mode = strict_mode
        self.logger = logger or logging.getLogger(__name__)
        self.file_ops = FileOperations(logger=self.logger)

        # 配置参数
        self.config = {
            'bbox_thickness': 2,        # 边界框线宽
            'seg_thickness': 1,         # 分割线宽
            'seg_alpha': 0.3,           # 分割掩码透明度
            'text_thickness': 1,        # 文本线宽
            'text_scale': 0.5,          # 文本大小
            'text_padding': 5,          # 文本内边距
            'font': cv2.FONT_HERSHEY_SIMPLEX  # 字体
        }

        # 颜色管理器
        self.color_manager = ColorManager()

    @abstractmethod
    def load_annotations(self) -> DatasetAnnotations:
        """加载标注数据（抽象方法）"""
        pass

    def visualize(self) -> VisualizationResult:
        """执行可视化流程"""
        result = VisualizationResult(success=False)

        try:
            # 1. 加载标注数据
            annotations = self.load_annotations()

            # 2. 验证输出目录（如果启用保存模式）
            if self.is_save:
                if not self.output_dir:
                    result.add_error("保存模式需要指定output_dir参数")
                    return result
                self.file_ops.ensure_dir(self.output_dir)

            # 3. 遍历所有图像进行可视化
            processed_count = 0
            for image_ann in annotations.images:
                success = self._visualize_single_image(image_ann)
                if success:
                    processed_count += 1
                elif self.strict_mode:
                    result.add_error(f"Failed to visualize image: {image_ann.image_id}")
                    return result

            result.success = True
            result.message = f"Successfully visualized {processed_count}/{len(annotations.images)} images"
            result.data = {"processed_count": processed_count}

        except Exception as e:
            result.add_error(f"Unexpected error during visualization: {e}")

        return result

    def _visualize_single_image(self, image_ann: ImageAnnotation) -> bool:
        """可视化单个图像"""
        try:
            # 1. 加载图像
            image_path = Path(image_ann.image_path)
            if not image_path.is_absolute():
                image_path = self.image_dir / image_ann.image_path

            if not image_path.exists():
                self._log_error(f"Image file not found: {image_path}")
                return False

            image = cv2.imread(str(image_path))
            if image is None:
                self._log_error(f"Failed to load image: {image_path}")
                return False

            # 2. 绘制所有对象
            for obj in image_ann.objects:
                self._draw_object(image, obj, image_ann.width, image_ann.height)

            # 3. 显示或保存
            if self.is_show:
                window_name = f"Visualization - {image_ann.image_id}"
                cv2.imshow(window_name, image)
                key = cv2.waitKey(0)
                cv2.destroyWindow(window_name)

                # 处理键盘输入
                if key == ord('q') or key == 27:  # q键或ESC键
                    return False  # 停止可视化
                # Enter键或空格键继续

            if self.is_save:
                output_file = self.output_dir / f"{image_ann.image_id}_visualized.jpg"
                cv2.imwrite(str(output_file), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
                self._log_info(f"Saved visualization to: {output_file}")

            return True

        except Exception as e:
            self._log_error(f"Error visualizing image {image_ann.image_id}: {e}")
            return False

    def _draw_object(self, image: np.ndarray, obj: ObjectAnnotation, img_width: int, img_height: int):
        """绘制单个对象标注"""
        # 获取类别颜色
        color = self.color_manager.get_color(obj.class_id)

        # 绘制边界框
        if obj.bbox is not None:
            self._draw_bbox(image, obj.bbox, color, obj.class_name, img_width, img_height)

        # 绘制分割
        if obj.segmentation is not None:
            if obj.segmentation.has_rle():
                # RLE格式（需要pycocotools）
                self._draw_rle_mask(image, obj.segmentation.rle, color, img_width, img_height)
            else:
                # 多边形格式
                self._draw_polygon(image, obj.segmentation, color, obj.class_name, img_width, img_height)

    def _draw_bbox(self, image: np.ndarray, bbox: BoundingBox, color: Tuple[int, int, int],
                   class_name: str, img_width: int, img_height: int):
        """绘制边界框"""
        x1, y1, x2, y2 = bbox.xyxy(img_width, img_height)

        # 绘制矩形
        cv2.rectangle(image, (x1, y1), (x2, y2), color, self.config['bbox_thickness'])

        # 绘制类别标签
        self._draw_text(image, class_name, (x1, y1 - self.config['text_padding']), color)

    def _draw_polygon(self, image: np.ndarray, segmentation: Segmentation, color: Tuple[int, int, int],
                      class_name: str, img_width: int, img_height: int):
        """绘制多边形分割"""
        points = segmentation.points_abs(img_width, img_height)

        if len(points) < 2:
            return

        # 转换为numpy数组
        points_np = np.array(points, dtype=np.int32)

        # 绘制多边形填充（半透明）
        overlay = image.copy()
        cv2.fillPoly(overlay, [points_np], color)
        cv2.addWeighted(overlay, self.config['seg_alpha'], image, 1 - self.config['seg_alpha'], 0, image)

        # 绘制多边形轮廓
        cv2.polylines(image, [points_np], True, color, self.config['seg_thickness'])

        # 绘制类别标签（使用第一个点）
        if points:
            self._draw_text(image, class_name, points[0], color)

    def _draw_rle_mask(self, image: np.ndarray, rle: Dict, color: Tuple[int, int, int],
                       img_width: int, img_height: int):
        """绘制RLE掩码"""
        # 需要pycocotools支持
        try:
            from pycocotools import mask as coco_mask
        except ImportError:
            self._log_error("pycocotools not installed, cannot draw RLE mask")
            return

        # 解码RLE为二值掩码
        binary_mask = coco_mask.decode(rle)

        # 将二值掩码转换为彩色掩码
        color_mask = np.zeros_like(image)
        for c in range(3):
            color_mask[:, :, c] = binary_mask * color[c]

        # 半透明叠加
        overlay = image.copy()
        overlay = cv2.addWeighted(overlay, 1 - self.config['seg_alpha'],
                                  color_mask, self.config['seg_alpha'], 0)

        # 将叠加结果复制回原图
        np.copyto(image, overlay, where=binary_mask[:, :, None].astype(bool))

    def _draw_text(self, image: np.ndarray, text: str, position: Tuple[int, int], color: Tuple[int, int, int]):
        """绘制文本标签"""
        # 计算文本大小
        (text_width, text_height), baseline = cv2.getTextSize(
            text, self.config['font'], self.config['text_scale'], self.config['text_thickness'])

        # 绘制背景矩形
        x, y = position
        bg_rect = (x, y - text_height - baseline),
                   (x + text_width, y + baseline)
        cv2.rectangle(image, bg_rect, (0, 0, 0), -1)  # 黑色背景

        # 绘制文本
        cv2.putText(image, text, (x, y - baseline), self.config['font'],
                    self.config['text_scale'], (255, 255, 255),  # 白色文本
                    self.config['text_thickness'], cv2.LINE_AA)

    def _log_info(self, message: str):
        """记录信息日志"""
        self.logger.info(message)

    def _log_error(self, message: str):
        """记录错误日志并抛出异常（严格模式）"""
        self.logger.error(message)
        if self.strict_mode:
            raise ValueError(message)

    def _log_warning(self, message: str):
        """记录警告日志"""
        self.logger.warning(message)
```

#### 3.1.2 ColorManager颜色管理器

```python
class ColorManager:
    """颜色管理器，确保同一类别使用相同颜色"""

    def __init__(self):
        # 预定义20个高对比度颜色（BGR格式）
        self.predefined_colors = [
            (0, 0, 255),      # 红色
            (0, 255, 0),      # 绿色
            (255, 0, 0),      # 蓝色
            (0, 255, 255),    # 黄色
            (255, 0, 255),    # 紫色
            (255, 255, 0),    # 青色
            (0, 128, 255),    # 橙色
            (128, 0, 255),    # 粉色
            (0, 255, 128),    # 浅绿
            (255, 128, 0),    # 天蓝
            (128, 255, 0),    # 酸橙绿
            (255, 0, 128),    # 玫瑰红
            (0, 128, 128),    # 橄榄绿
            (128, 0, 128),    # 深紫色
            (128, 128, 0),    # 深青色
            (192, 192, 192),  # 银色
            (128, 128, 128),  # 灰色
            (64, 64, 64),     # 深灰色
            (0, 64, 128),     # 深蓝色
            (128, 64, 0),     # 棕色
        ]
        self.color_cache: Dict[int, Tuple[int, int, int]] = {}

    def get_color(self, class_id: int) -> Tuple[int, int, int]:
        """获取类别颜色"""
        if class_id in self.color_cache:
            return self.color_cache[class_id]

        # 循环使用预定义颜色
        color_idx = class_id % len(self.predefined_colors)
        color = self.predefined_colors[color_idx]
        self.color_cache[class_id] = color

        return color
```

### 3.2 可视化器设计

#### 3.2.1 LabelMeVisualizer

处理LabelMe格式标注的可视化。

**输入参数**：
- `label_dir`: LabelMe标签文件夹目录（包含多个JSON文件）
- `image_dir`: 图像文件夹目录
- `class_file`: 类别文件路径（可选）
- `output_dir`: 输出目录（is_save=True时使用）
- `is_show`: 是否显示（默认True）
- `is_save`: 是否保存（默认False）

**API设计**：
```python
class LabelMeVisualizer(BaseVisualizer):
    """LabelMe格式可视化器"""

    def __init__(self,
                 label_dir: Union[str, Path],
                 image_dir: Union[str, Path],
                 class_file: Optional[Union[str, Path]] = None,
                 **kwargs):
        super().__init__(label_dir, image_dir, **kwargs)
        self.class_file = Path(class_file) if class_file else None
        self.handler = LabelMeAnnotationHandler(
            label_dir=str(label_dir),
            class_file=str(class_file) if class_file else None,
            strict_mode=self.strict_mode,
            logger=self.logger
        )

    def load_annotations(self) -> DatasetAnnotations:
        """加载LabelMe标注数据"""
        result = self.handler.read()
        if not result.success:
            raise ValueError(f"Failed to load LabelMe annotations: {result.message}")
        return result.data
```

**特性**：
1. 支持批量处理LabelMe JSON文件
2. 自动从标注文件中提取类别信息
3. 支持矩形（检测）和多边形（分割）标注的绘制
4. 严格验证JSON格式和必需字段

#### 3.2.2 YOLOVisualizer

处理YOLO格式标注的可视化。

**输入参数**：
- `label_dir`: YOLO标签文件夹目录（包含多个.txt文件）
- `image_dir`: 图像文件夹目录
- `class_file`: 类别文件路径（必需）
- `output_dir`: 输出目录（is_save=True时使用）
- `is_show`: 是否显示（默认True）
- `is_save`: 是否保存（默认False）

**API设计**：
```python
class YOLOVisualizer(BaseVisualizer):
    """YOLO格式可视化器"""

    def __init__(self,
                 label_dir: Union[str, Path],
                 image_dir: Union[str, Path],
                 class_file: Union[str, Path],
                 **kwargs):
        super().__init__(label_dir, image_dir, **kwargs)
        self.class_file = Path(class_file)
        self.handler = YOLOAnnotationHandler(
            label_dir=str(label_dir),
            class_file=str(class_file),
            image_dir=str(image_dir),
            strict_mode=self.strict_mode,
            logger=self.logger
        )

    def load_annotations(self) -> DatasetAnnotations:
        """加载YOLO标注数据"""
        result = self.handler.read()
        if not result.success:
            raise ValueError(f"Failed to load YOLO annotations: {result.message}")
        return result.data
```

**特性**：
1. 自动检测标注类型（目标检测 vs 实例分割）
2. 支持归一化坐标验证（0-1范围）
3. 自动匹配图片和标签文件
4. 严格的格式验证（每行5个值表示检测，多个值表示分割）

#### 3.2.3 COCOVisualizer

处理COCO格式标注的可视化。

**输入参数**：
- `annotation_file`: COCO标注文件路径（单个JSON文件）
- `image_dir`: 图像文件夹目录
- `output_dir`: 输出目录（is_save=True时使用）
- `is_show`: 是否显示（默认True）
- `is_save`: 是否保存（默认False）

**API设计**：
```python
class COCOVisualizer(BaseVisualizer):
    """COCO格式可视化器"""

    def __init__(self,
                 annotation_file: Union[str, Path],
                 image_dir: Union[str, Path],
                 **kwargs):
        super().__init__(annotation_file, image_dir, **kwargs)
        self.annotation_file = Path(annotation_file)
        self.handler = COCOAnnotationHandler(
            annotation_file=str(annotation_file),
            strict_mode=self.strict_mode,
            logger=self.logger
        )

    def load_annotations(self) -> DatasetAnnotations:
        """加载COCO标注数据"""
        result = self.handler.read()
        if not result.success:
            raise ValueError(f"Failed to load COCO annotations: {result.message}")
        return result.data
```

**特性**：
1. 支持标准COCO JSON格式
2. 自动检测RLE格式和点列表格式
3. 支持`is_crowd`标志处理
4. 完整的类别和图片信息提取
5. RLE掩码解码和绘制（如果pycocotools可用）

### 3.3 颜色管理机制

#### 3.3.1 设计要点

1. **确定性颜色分配**：同一类别的标注框使用同一个颜色，基于类别ID实现
2. **高对比度颜色**：预定义20-30个高对比度的颜色列表
3. **颜色循环使用**：如果要展示的类别列表超出长度，循环使用颜色
4. **颜色缓存**：避免重复计算，提高性能

#### 3.3.2 实现原理

```python
class_id = 0  # 类别ID
color_idx = class_id % len(predefined_colors)  # 循环取模
color = predefined_colors[color_idx]  # 获取颜色
```

### 3.4 用户交互机制

#### 3.4.1 is_show模式

```python
# 显示图像
window_name = f"Visualization - {image_id}"
cv2.imshow(window_name, image)

# 等待键盘输入
key = cv2.waitKey(0)

# 处理按键
if key == ord('q') or key == 27:  # q键或ESC键
    break  # 退出可视化
# Enter键或空格键：继续下一张
```

#### 3.4.2 is_save模式

```python
# 创建输出目录（如果不存在）
output_dir.mkdir(parents=True, exist_ok=True)

# 生成输出文件名
output_file = output_dir / f"{image_id}_visualized.jpg"

# 保存图像
cv2.imwrite(str(output_file), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
```

## 4. 技术实现要求

### 4.1 核心依赖

**必需依赖**：
- `opencv-python>=4.6.0.66`：图像处理和可视化

**可选依赖**：
- `pycocotools>=2.0.0`：RLE掩码解码（COCO专用）

**内部依赖**：
- `dataflow.label`：标签数据读取
- `dataflow.util`：文件操作和日志记录

**Python版本**：
- Python 3.8+（确保类型提示支持）

### 4.2 跨平台兼容性保证

1. **路径处理**：统一使用`pathlib.Path`，避免字符串拼接
2. **文件编码**：统一使用UTF-8编码
3. **行尾符**：使用Python的通用换行模式
4. **图像格式**：使用OpenCV支持的通用格式
5. **权限处理**：正确处理文件权限和异常

### 4.3 错误处理策略

1. **严格模式**：默认启用，遇到错误立即引发异常
2. **详细错误信息**：包含上下文信息的错误消息
3. **类型验证**：使用类型提示和运行时验证
4. **资源清理**：确保OpenCV窗口的正确关闭
5. **降级处理**：pycocotools未安装时的降级处理

### 4.4 性能优化

1. **批量处理**：支持文件夹级别的批量可视化
2. **颜色缓存**：避免重复计算类别颜色
3. **图像缓存**：可选的图像缓存机制（未来扩展）
4. **并行处理**：大规模数据集的可选并行处理（未来扩展）

### 4.5 代码质量要求

1. **类型提示**：所有公共API必须有完整的类型提示
2. **文档字符串**：所有类和方法必须有完整的docstring
3. **代码风格**：遵循PEP 8规范
4. **单元测试**：核心功能必须有单元测试覆盖

## 5. 测试策略

### 5.1 测试目录结构

```
tests/
├── __init__.py
├── conftest.py              # 测试配置和fixture
├── visualize/
│   ├── __init__.py
│   ├── test_base.py         # 基类测试
│   ├── test_labelme_visualizer.py
│   ├── test_yolo_visualizer.py
│   └── test_coco_visualizer.py
```

### 5.2 测试数据使用

使用现有的`assets/test_data/`目录下的测试数据：
- **目标检测数据**（det目录）：
  - LabelMe格式：`det/labelme/`
  - YOLO格式：`det/yolo/`
  - COCO格式：`det/coco/`
- **实例分割数据**（seg目录）：
  - LabelMe格式：`seg/labelme/`
  - YOLO格式：`seg/yolo/`
  - COCO格式：`seg/coco/`（包含annotations.json和annotations-rle.json）

### 5.3 测试编写规范

1. **测试类命名**：`Test{ClassName}`
2. **测试方法命名**：`test_{function_name}_{scenario}`
3. **测试覆盖**：正常流程、边界条件、异常情况
4. **测试隔离**：每个测试独立，不依赖执行顺序
5. **测试清理**：测试后清理创建的临时文件和OpenCV窗口

### 5.4 关键测试用例

1. **颜色管理测试**：
   - 测试相同类别获得相同颜色
   - 测试颜色循环使用
   - 测试颜色可区分度

2. **可视化绘制测试**：
   - 测试边界框绘制正确性
   - 测试多边形绘制正确性
   - 测试RLE掩码绘制（如果pycocotools可用）

3. **用户交互测试**：
   - 测试is_show模式的键盘交互
   - 测试is_save模式的图像保存

4. **错误处理测试**：
   - 测试无效输入的处理
   - 测试图像加载失败的处理
   - 测试严格模式和宽容模式

### 5.5 测试覆盖率目标

- 总体覆盖率：≥85%
- 核心模块覆盖率：≥90%
- 异常处理覆盖率：≥80%

## 6. 示例代码

### 6.1 示例目录结构

```
samples/
├── __init__.py
└── visualize/
    ├── __init__.py
    ├── labelme_demo.py    # LabelMe可视化示例
    ├── yolo_demo.py       # YOLO可视化示例
    └── coco_demo.py       # COCO可视化示例
```

### 6.2 示例代码编写规范

1. **完整性**：展示完整的使用流程
2. **简洁性**：聚焦核心功能，避免冗余代码
3. **注释**：详细的步骤说明和注意事项
4. **错误处理**：展示正确的错误处理方式
5. **可执行性**：确保示例可以直接运行

### 6.3 LabelMe可视化示例

```python
#!/usr/bin/env python3
"""
LabelMe标注可视化示例

展示如何使用LabelMeVisualizer可视化LabelMe格式标注。
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dataflow.visualize import LabelMeVisualizer
from dataflow.util import LoggingOperations

def main():
    """主函数"""
    # 配置日志
    log_ops = LoggingOperations()
    logger = log_ops.get_logger("labelme_visualize_demo", level="INFO")

    # 示例数据路径
    data_dir = project_root / "assets" / "test_data" / "det" / "labelme"
    image_dir = data_dir  # LabelMe格式中JSON文件和图片在同一目录
    class_file = data_dir / "classes.txt"

    if not data_dir.exists():
        logger.error(f"数据目录不存在: {data_dir}")
        return

    logger.info("=" * 50)
    logger.info("LabelMe标注可视化示例")
    logger.info("=" * 50)

    # 创建可视化器
    logger.info(f"创建LabelMe可视化器:")
    logger.info(f"  标签目录: {data_dir}")
    logger.info(f"  图片目录: {image_dir}")
    logger.info(f"  类别文件: {class_file}")

    visualizer = LabelMeVisualizer(
        label_dir=str(data_dir),
        image_dir=str(image_dir),
        class_file=str(class_file) if class_file.exists() else None,
        is_show=True,      # 显示窗口
        is_save=False,     # 不保存
        strict_mode=True,
        logger=logger
    )

    # 执行可视化
    logger.info("\n开始可视化（按Enter键下一张，按q键退出）...")
    result = visualizer.visualize()

    if result.success:
        logger.info(f"可视化完成: {result.message}")
    else:
        logger.error(f"可视化失败: {result.message}")
        if result.errors:
            for error in result.errors:
                logger.error(f"  - {error}")

if __name__ == "__main__":
    main()
```

### 6.4 YOLO可视化示例

```python
#!/usr/bin/env python3
"""
YOLO标注可视化示例

展示如何使用YOLOVisualizer可视化YOLO格式标注。
支持自动检测目标检测和实例分割格式。
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dataflow.visualize import YOLOVisualizer
from dataflow.util import LoggingOperations

def main():
    """主函数"""
    # 配置日志
    log_ops = LoggingOperations()
    logger = log_ops.get_logger("yolo_visualize_demo", level="INFO")

    # 示例数据路径
    data_dir = project_root / "assets" / "test_data" / "det" / "yolo"
    image_dir = data_dir / "images"
    label_dir = data_dir / "labels"
    class_file = data_dir / "classes.txt"

    if not data_dir.exists():
        logger.error(f"数据目录不存在: {data_dir}")
        return

    logger.info("=" * 50)
    logger.info("YOLO标注可视化示例")
    logger.info("=" * 50)

    # 创建可视化器
    logger.info(f"创建YOLO可视化器:")
    logger.info(f"  标签目录: {label_dir}")
    logger.info(f"  图片目录: {image_dir}")
    logger.info(f"  类别文件: {class_file}")

    visualizer = YOLOVisualizer(
        label_dir=str(label_dir),
        image_dir=str(image_dir),
        class_file=str(class_file),
        is_show=True,      # 显示窗口
        is_save=True,      # 同时保存
        output_dir=data_dir / "visualized_output",
        strict_mode=True,
        logger=logger
    )

    # 执行可视化
    logger.info("\n开始可视化（按Enter键下一张，按q键退出）...")
    result = visualizer.visualize()

    if result.success:
        logger.info(f"可视化完成: {result.message}")
        logger.info(f"结果保存到: {data_dir / 'visualized_output'}")
    else:
        logger.error(f"可视化失败: {result.message}")
        if result.errors:
            for error in result.errors:
                logger.error(f"  - {error}")

if __name__ == "__main__":
    main()
```

### 6.5 COCO可视化示例

```python
#!/usr/bin/env python3
"""
COCO标注可视化示例

展示如何使用COCOVisualizer可视化COCO格式标注。
支持RLE格式和多边形格式。
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dataflow.visualize import COCOVisualizer
from dataflow.util import LoggingOperations

def main():
    """主函数"""
    # 配置日志
    log_ops = LoggingOperations()
    logger = log_ops.get_logger("coco_visualize_demo", level="INFO")

    # 示例数据路径
    data_dir = project_root / "assets" / "test_data" / "det" / "coco"
    annotation_file = data_dir / "annotations.json"
    image_dir = data_dir / "images"

    if not annotation_file.exists():
        logger.error(f"标注文件不存在: {annotation_file}")
        return

    logger.info("=" * 50)
    logger.info("COCO标注可视化示例")
    logger.info("=" * 50)

    # 创建可视化器
    logger.info(f"创建COCO可视化器:")
    logger.info(f"  标注文件: {annotation_file}")
    logger.info(f"  图片目录: {image_dir}")

    visualizer = COCOVisualizer(
        annotation_file=str(annotation_file),
        image_dir=str(image_dir),
        is_show=True,      # 显示窗口
        is_save=False,     # 不保存
        strict_mode=True,
        logger=logger
    )

    # 执行可视化
    logger.info("\n开始可视化（按Enter键下一张，按q键退出）...")
    result = visualizer.visualize()

    if result.success:
        logger.info(f"可视化完成: {result.message}")
    else:
        logger.error(f"可视化失败: {result.message}")
        if result.errors:
            for error in result.errors:
                logger.error(f"  - {error}")

if __name__ == "__main__":
    main()
```

## 7. 开发计划

### 7.1 阶段划分（共5个阶段）

#### 阶段一：基础架构（预计2天）
**目标**：创建可视化模块基础架构，实现BaseVisualizer和ColorManager

**任务清单**：
1. 创建`dataflow/visualize/`目录结构
2. 创建`__init__.py`模块导出文件
3. 实现`base.py`中的`BaseVisualizer`基类
4. 实现`ColorManager`颜色管理器
5. 创建`utils.py`工具函数文件
6. 编写基础测试文件`tests/visualize/test_base.py`

**验收标准**：
- 基类功能完整，测试通过
- 颜色管理正确（相同类别相同颜色）
- 代码结构清晰，类型提示完整

#### 阶段二：具体可视化器实现（预计3天）
**目标**：实现三种格式的可视化器

**任务清单**：
1. 实现`labelme_visualizer.py`
2. 实现`yolo_visualizer.py`
3. 实现`coco_visualizer.py`
4. 完善绘图逻辑（边界框、多边形、RLE掩码）
5. 编写单元测试

**验收标准**：
- 三种格式的可视化功能完整
- 与现有label模块集成正常
- 单元测试覆盖率≥85%

#### 阶段三：用户交互和图像保存（预计2天）
**目标**：实现is_show和is_save功能

**任务清单**：
1. 实现is_show模式交互逻辑
2. 实现is_save模式保存逻辑
3. 完善错误处理和日志记录
4. 编写交互测试用例

**验收标准**：
- is_show模式交互正常（Enter键下一张，q键退出）
- is_save模式保存正常（JPEG格式）
- 错误处理完善，日志记录完整

#### 阶段四：测试和示例（预计2天）
**目标**：编写完整的测试和示例代码

**任务清单**：
1. 编写单元测试（使用现有测试数据）
2. 创建使用示例（三个格式的演示）
3. 集成测试（与现有模块的集成）
4. 文档编写（API文档、使用指南）

**验收标准**：
- 单元测试通过率100%
- 示例代码可正常运行
- 文档完整清晰

#### 阶段五：优化和发布（预计1天）
**目标**：代码优化和准备发布

**任务清单**：
1. 代码审查和优化
2. 依赖管理（更新requirements.txt）
3. 发布准备（更新CHANGELOG）
4. 整体测试和验证

**验收标准**：
- 代码质量检查通过（flake8, mypy）
- 依赖管理正确
- 跨平台兼容性验证通过

### 7.2 详细任务清单

#### 阶段一详细任务
- [ ] 创建`dataflow/visualize/`目录
- [ ] 编写`dataflow/visualize/__init__.py`
- [ ] 实现`BaseVisualizer`基类的所有方法
- [ ] 实现`ColorManager`颜色管理器
- [ ] 创建`VisualizationResult`数据类
- [ ] 编写`test_base.py`单元测试
- [ ] 测试颜色管理功能
- [ ] 验证基类接口设计合理性

#### 阶段二详细任务
- [ ] 实现`LabelMeVisualizer`类
- [ ] 实现`YOLOVisualizer`类
- [ ] 实现`COCOVisualizer`类
- [ ] 实现`_draw_bbox()`边界框绘制
- [ ] 实现`_draw_polygon()`多边形绘制
- [ ] 实现`_draw_rle_mask()`RLE掩码绘制
- [ ] 编写`test_labelme_visualizer.py`单元测试
- [ ] 编写`test_yolo_visualizer.py`单元测试
- [ ] 编写`test_coco_visualizer.py`单元测试
- [ ] 使用现有测试数据进行测试

#### 阶段三详细任务
- [ ] 实现is_show模式的键盘交互逻辑
- [ ] 实现is_save模式的图像保存逻辑
- [ ] 添加详细的日志记录点
- [ ] 实现严格模式和宽容模式切换
- [ ] 处理图像加载失败等异常情况
- [ ] 编写交互测试用例
- [ ] 测试OpenCV窗口管理

#### 阶段四详细任务
- [ ] 为每个可视化器编写全面的测试用例
- [ ] 使用`assets/test_data/`中的测试数据
- [ ] 创建`samples/visualize/`目录
- [ ] 编写`labelme_demo.py`示例
- [ ] 编写`yolo_demo.py`示例
- [ ] 编写`coco_demo.py`示例
- [ ] 编写模块API文档
- [ ] 编写使用指南
- [ ] 更新项目README

#### 阶段五详细任务
- [ ] 检查代码规范和风格
- [ ] 优化性能瓶颈
- [ ] 更新`requirements.txt`
- [ ] 更新`setup.py`安装配置
- [ ] 添加可选依赖标记
- [ ] 更新CHANGELOG
- [ ] 创建发布说明
- [ ] 整体集成测试

### 7.3 验收标准

#### 功能验收标准
- [ ] LabelMe格式可视化功能正常
- [ ] YOLO格式可视化功能正常（需要类别文件）
- [ ] COCO格式可视化功能正常（支持RLE掩码）
- [ ] is_show模式交互正常（Enter键下一张，q键退出）
- [ ] is_save模式保存正常（JPEG格式）
- [ ] 颜色管理正确（相同类别相同颜色，颜色循环使用）
- [ ] 边界框绘制正确
- [ ] 多边形分割绘制正确
- [ ] RLE掩码绘制正确（如果pycocotools可用）

#### 代码质量验收标准
- [ ] 单元测试通过率100%
- [ ] 代码覆盖率不低于85%
- [ ] 代码风格检查通过（flake8）
- [ ] 类型检查通过（mypy）
- [ ] 文档完整（API文档、使用示例、README）

#### 集成验收标准
- [ ] 与现有label模块集成正常
- [ ] 与现有util模块集成正常
- [ ] 依赖管理正确（requirements.txt、setup.py）
- [ ] 安装和导入正常
- [ ] 跨平台兼容性测试通过（Windows、Linux、macOS）
- [ ] Python版本兼容性测试通过（3.8-3.12）

### 7.4 风险评估和缓解措施

#### 风险1：OpenCV窗口交互问题
- **风险**：不同操作系统下的OpenCV窗口行为可能不同
- **缓解**：
  - 实现简单的键盘交互（仅支持基本键）
  - 提供详细的错误信息
  - 充分的跨平台测试

#### 风险2：RLE处理依赖问题
- **风险**：pycocotools在某些平台安装困难
- **缓解**：
  - 提供降级处理（pycocotools未安装时跳过RLE功能）
  - 详细的安装说明
  - 提供多边形格式作为备选

#### 风险3：内存使用问题
- **风险**：大规模数据集可视化可能内存占用高
- **缓解**：
  - 支持分批处理
  - 优化图像加载和绘制性能
  - 提供内存使用建议

#### 风险4：颜色可区分度问题
- **风险**：类别过多时颜色可能难以区分
- **缓解**：
  - 使用高对比度颜色方案
  - 提供自定义颜色映射选项
  - 在文档中说明最佳实践

## 8. 质量保证

### 8.1 代码规范检查

使用以下工具确保代码质量：

1. **black**：代码格式化
2. **isort**：导入排序
3. **flake8**：代码风格检查
4. **mypy**：类型检查
5. **pylint**：代码质量分析

配置预提交钩子（pre-commit hooks）自动检查。

### 8.2 测试覆盖率要求

使用`pytest-cov`进行测试覆盖率统计：

```bash
# 运行测试并生成覆盖率报告
pytest tests/visualize/ --cov=dataflow.visualize --cov-report=term --cov-report=html

# 最小覆盖率要求
pytest tests/visualize/ --cov=dataflow.visualize --cov-fail-under=85
```

### 8.3 发布前检查清单

发布前必须完成以下检查：

- [ ] 所有单元测试通过
- [ ] 测试覆盖率≥85%
- [ ] 类型检查通过（mypy）
- [ ] 代码风格检查通过（flake8）
- [ ] 示例代码可正常运行
- [ ] 跨平台兼容性验证通过
- [ ] 安全扫描通过

## 总结

本规范文档为DataFlow-CV可视化模块的开发提供了完整的蓝图，包括：

1. **详细的设计方案**：涵盖架构、接口、颜色管理、交互机制
2. **完整的实现指南**：每个可视化器的具体实现要求
3. **严格的测试策略**：确保代码质量和可靠性
4. **实用的示例代码**：帮助用户快速上手
5. **可行的开发计划**：分阶段实施，降低风险
6. **全面的质量保证**：从代码规范到持续集成

遵循本规范进行开发，可以确保DataFlow-CV可视化模块的高质量、可维护性和易用性，为用户提供直观、高效的计算机视觉标注可视化工具。

---

**文档版本**：1.0
**创建日期**：2026-03-22
**最后更新**：2026-03-22
**作者**：Claude Code