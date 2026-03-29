# DataFlow-CV 可视化模块开发文档 (visualize)

## 1. 模块概述

### 功能定位
`dataflow/visualize` 模块负责计算机视觉标注的可视化展示，支持 YOLO、LabelMe 和 COCO 三种格式的标注数据。它将标注信息（边界框、分割掩码）绘制到原始图像上，提供直观的视觉反馈，支持交互式浏览和批量导出。

### 核心特性
- **多格式支持**：YOLO、LabelMe、COCO 格式的可视化
- **双任务可视化**：目标检测（边界框）和实例分割（多边形/RLE掩码）
- **颜色管理系统**：自动为每个类别分配唯一且一致的颜色
- **交互模式**：键盘控制（Enter下一张，q退出，s保存）
- **批量处理**：支持整个数据集的可视化导出
- **RLE支持**：COCO RLE 格式掩码的可视化
- **详细日志**：集成详细日志模式，支持进度报告

### 设计原则
1. **用户友好**：直观的可视化效果，简单的交互控制
2. **一致性**：同一类别在不同图片中使用相同颜色
3. **性能优化**：支持批量处理，内存效率高
4. **扩展性**：易于添加新的可视化格式或效果
5. **错误容忍**：支持严格模式和非严格模式

## 2. 核心架构

### 类图/组件图
```
┌─────────────────────────────────────────┐
│         dataflow.visualize              │
├─────────────────────────────────────────┤
│  • BaseVisualizer (抽象基类)            │
│    - visualize() (模板方法)             │
│    - ColorManager (颜色管理)            │
│    - VisualizationResult (结果容器)     │
│                                         │
│  • 具体可视化器实现                     │
│    - YOLOVisualizer                     │
│    - LabelMeVisualizer                  │
│    - COCOVisualizer                     │
│                                         │
│  • 辅助组件                             │
│    - ColorManager (独立颜色管理)        │
│    - drawing_utils (绘图工具函数)       │
└─────────────────────────────────────────┘
```

### 数据流
1. **可视化流程**（模板方法模式）：
   ```
   初始化配置 → 加载标注 → 遍历图片 → 绘制标注 → 显示/保存
        ↳ 颜色管理        ↳ 进度报告
   ```

2. **颜色管理流程**：
   ```
   获取类别ID → 检查颜色缓存 → 返回分配的颜色
                    ↳ 若未分配则生成新颜色
   ```

3. **交互控制流程**：
   ```
   显示图片 → 等待键盘输入 → 处理命令 → 更新状态
                    ↳ Enter: 下一张
                    ↳ q: 退出
                    ↳ s: 保存当前
   ```

### 依赖关系
- **内部依赖**：`dataflow/label`（标注数据模型）, `dataflow/util`（日志和文件操作）
- **外部依赖**：`opencv-python`（图像处理和显示）, `numpy`（数组操作）
- **可选依赖**：`pycocotools`（COCO RLE 解码）
- **被依赖**：`dataflow/cli`（CLI可视化命令）

## 3. 主要API

### BaseVisualizer 抽象基类
**职责**：定义可视化流程的模板方法

**核心方法**：
- `visualize() -> VisualizationResult`：执行可视化流程（模板方法）
- `load_annotations() -> DatasetAnnotations`：加载标注数据（抽象方法）
- `draw_annotations(image, annotations) -> np.ndarray`：绘制标注到图像
- `display_image(image, window_name) -> str`：显示图像并处理交互

**模板方法流程**：
```python
def visualize(self):
    # 1. 初始化
    self._setup()

    # 2. 加载标注
    dataset = self.load_annotations()

    # 3. 遍历图片
    for image_anno in dataset.images:
        # 4. 加载图片
        image = self._load_image(image_anno)

        # 5. 绘制标注
        annotated = self.draw_annotations(image, image_anno.objects)

        # 6. 显示/保存
        if self.is_show:
            key = self.display_image(annotated, image_anno.image_id)
            # 处理键盘输入
        if self.is_save:
            self._save_image(annotated, image_anno.image_id)

    # 7. 返回结果
    return VisualizationResult(success=True, ...)
```

### ColorManager 类
**职责**：管理类别颜色分配，确保一致性

**核心方法**：
- `get_color(class_id: int) -> Tuple[int, int, int]`：获取类别对应的颜色（BGR格式）
- `reset()`：重置颜色缓存
- `get_color_map() -> Dict[int, Tuple[int, int, int]]`：获取当前颜色映射

**颜色生成算法**：
- 预生成1000个唯一颜色（HSV色彩空间均匀分布）
- 超出预生成范围时使用确定性算法生成
- 确保相邻类别颜色差异最大化

### 具体可视化器类

#### YOLOVisualizer 类
**职责**：可视化 YOLO 格式标注

**特定参数**：
- `label_dir: Path`：YOLO标签目录
- `image_dir: Path`：图片目录
- `class_file: Path`：类别文件

**特性**：
- 支持 YOLO 检测和分割格式
- 自动从 TXT 文件解析标注

#### LabelMeVisualizer 类
**职责**：可视化 LabelMe 格式标注

**特定参数**：
- `label_dir: Path`：LabelMe JSON 文件目录
- `image_dir: Path`：图片目录
- `class_file: Optional[Path]`：可选类别文件

**特性**：
- 支持矩形和多边形形状
- 从 JSON 文件解析形状和标签

#### COCOVisualizer 类
**职责**：可视化 COCO 格式标注

**特定参数**：
- `annotation_file: Path`：COCO JSON 标注文件
- `image_dir: Path`：图片目录

**特性**：
- 支持多边形和 RLE 分割格式
- 处理 COCO 的特殊字段（iscrowd, area等）
- 可选 RLE 解码（需 pycocotools）

### VisualizationResult 类
**职责**：封装可视化结果

**属性**：
- `success: bool`：是否成功
- `images_processed: int`：处理的图片数量
- `errors: List[str]`：错误信息列表
- `warnings: List[str]`：警告信息列表

### 配置参数

**通用参数**：
- `is_show: bool = True`：是否显示图像窗口
- `is_save: bool = False`：是否保存结果图像
- `output_dir: Optional[Path]`：保存目录（is_save=True时必需）
- `verbose: bool = False`：详细日志模式
- `strict_mode: bool = True`：严格错误处理模式

**绘图参数**：
- `bbox_thickness: int = 2`：边界框线宽
- `seg_thickness: int = 1`：分割轮廓线宽
- `seg_alpha: float = 0.3`：分割掩码透明度
- `text_thickness: int = 1`：文本线宽
- `text_scale: float = 0.5`：文本缩放
- `text_padding: int = 5`：文本内边距
- `font: int = cv2.FONT_HERSHEY_SIMPLEX`：字体类型

## 4. 使用指南

### 快速开始

```python
# 1. YOLO 格式可视化
from dataflow.visualize import YOLOVisualizer
from pathlib import Path

visualizer = YOLOVisualizer(
    label_dir=Path("yolo_labels"),
    image_dir=Path("images"),
    class_file=Path("classes.txt"),
    is_show=True,      # 显示窗口
    is_save=False,     # 不保存
    verbose=True       # 详细日志
)

result = visualizer.visualize()
print(f"Processed {result.images_processed} images")

# 2. COCO 格式可视化（批量保存）
from dataflow.visualize import COCOVisualizer

visualizer = COCOVisualizer(
    annotation_file=Path("annotations.json"),
    image_dir=Path("images"),
    is_show=False,     # 不显示（批量处理）
    is_save=True,      # 保存结果
    output_dir=Path("visualized"),  # 保存目录
    verbose=True
)

result = visualizer.visualize()

# 3. LabelMe 格式可视化（交互模式）
from dataflow.visualize import LabelMeVisualizer

visualizer = LabelMeVisualizer(
    label_dir=Path("labelme_json"),
    image_dir=Path("images"),
    is_show=True,
    is_save=False
)

# 交互控制：Enter下一张，q退出，s保存当前
result = visualizer.visualize()
```

### 常见场景

#### 场景1：批量导出可视化结果
```python
visualizer = YOLOVisualizer(
    label_dir=Path("dataset/labels"),
    image_dir=Path("dataset/images"),
    class_file=Path("dataset/classes.txt"),
    is_show=False,     # 禁用显示（批量处理）
    is_save=True,      # 启用保存
    output_dir=Path("output/visualized"),
    verbose=True       # 进度报告
)

result = visualizer.visualize()
print(f"Exported {result.images_processed} images to output/visualized/")
```

#### 场景2：交互式标注检查
```python
visualizer = COCOVisualizer(
    annotation_file=Path("annotations.json"),
    image_dir=Path("images"),
    is_show=True,      # 启用显示
    is_save=True,      # 可选保存
    output_dir=Path("checked"),
    verbose=False      # 简洁输出
)

# 交互操作：
# - Enter: 下一张图片
# - q: 退出程序
# - s: 保存当前图片到输出目录
# - 其他: 忽略，继续显示
result = visualizer.visualize()
```

#### 场景3：自定义绘图样式
```python
from dataflow.visualize import YOLOVisualizer

visualizer = YOLOVisualizer(
    label_dir=Path("labels"),
    image_dir=Path("images"),
    class_file=Path("classes.txt"),
    # 自定义绘图参数
    bbox_thickness=3,      # 更粗的边界框
    seg_alpha=0.4,         # 更不透明的分割掩码
    text_scale=0.7,        # 更大的文本
    text_padding=8,        # 更多的文本内边距
    is_show=True
)

result = visualizer.visualize()
```

#### 场景4：调试模式（详细日志）
```python
visualizer = LabelMeVisualizer(
    label_dir=Path("labelme"),
    image_dir=Path("images"),
    verbose=True,       # 启用详细日志
    strict_mode=False   # 非严格模式，继续处理错误
)

result = visualizer.visualize()

if result.errors:
    print(f"Encountered {len(result.errors)} errors:")
    for err in result.errors:
        print(f"  - {err}")

if result.warnings:
    print(f"Encountered {len(result.warnings)} warnings")
```

## 5. 开发步骤

### 阶段1：设计 BaseVisualizer 基类
1. **分析可视化需求**：确定通用流程（加载、绘制、显示/保存）
2. **设计模板方法模式**：
   ```python
   class BaseVisualizer(ABC):
       def visualize(self) -> VisualizationResult:  # 模板方法
           self._setup()
           dataset = self.load_annotations()  # 抽象方法
           for image_anno in dataset.images:
               image = self._load_image(image_anno)
               annotated = self.draw_annotations(image, image_anno.objects)
               # 显示/保存逻辑
           return self._create_result()

       @abstractmethod
       def load_annotations(self) -> DatasetAnnotations:
           pass
   ```
3. **设计配置系统**：绘图参数、显示/保存选项、错误处理模式
4. **设计结果封装**：`VisualizationResult` 类设计

### 阶段2：实现 ColorManager 颜色管理系统
1. **设计颜色分配策略**：
   - 预生成固定数量唯一颜色
   - 确定性算法确保一致性
2. **实现颜色生成算法**：
   ```python
   class ColorManager:
       def __init__(self):
           # 预生成1000个唯一颜色
           self._predefined_colors = self._generate_colors(1000)
           self._color_cache: Dict[int, Tuple[int, int, int]] = {}

       def _generate_colors(self, n):
           colors = []
           for i in range(n):
               hue = i / n  # HSV色相均匀分布
               rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
               bgr = (int(rgb[2]*255), int(rgb[1]*255), int(rgb[0]*255))
               colors.append(bgr)
           return colors
   ```
3. **实现颜色缓存机制**：
   - 相同 class_id 返回相同颜色
   - 支持缓存重置和查询

### 阶段3：实现 YOLOVisualizer
1. **实现标注加载**：
   ```python
   def load_annotations(self) -> DatasetAnnotations:
       # 使用 label 模块的 YoloAnnotationHandler
       from dataflow.label import YoloAnnotationHandler

       handler = YoloAnnotationHandler()
       result = handler.read(
           self.label_dir,
           image_dir=self.image_dir,
           class_file=self.class_file
       )

       if not result.success:
           raise VisualizationError(f"Failed to load YOLO annotations: {result.errors}")

       return result.data
   ```
2. **实现绘图逻辑**：
   - 边界框绘制：矩形 + 类别标签
   - 分割绘制：多边形填充 + 轮廓
   - 文本绘制：带背景的类别名称
3. **集成颜色管理**：使用 `ColorManager` 获取类别颜色

### 阶段4：实现 LabelMeVisualizer
1. **实现标注加载**：
   - 使用 `LabelMeAnnotationHandler` 加载数据
   - 处理可选类别文件情况
2. **实现绘图逻辑**：
   - 矩形形状：绘制边界框
   - 多边形形状：绘制填充多边形
   - 支持 LabelMe 特定的形状属性
3. **实现交互控制**：
   - 键盘事件处理（Enter, q, s）
   - 状态更新和反馈

### 阶段5：实现 COCOVisualizer
1. **实现标注加载**：
   - 使用 `CocoAnnotationHandler` 加载数据
   - 处理 RLE 格式支持
2. **实现 RLE 解码**：
   ```python
   def _decode_rle(self, rle_data):
       if not HAS_PYCOCOTOOLS:
           self._logger.warning("pycocotools not available, skipping RLE visualization")
           return None

       import pycocotools.mask as mask_util
       mask = mask_util.decode(rle_data)
       return mask
   ```
3. **实现掩码绘制**：
   - RLE 掩码解码和绘制
   - 多边形轮廓绘制
   - crowd 标注的特殊处理

### 阶段6：实现交互模式
1. **设计交互接口**：
   ```python
   def display_image(self, image, window_name):
       cv2.imshow(window_name, image)
       key = cv2.waitKey(0) & 0xFF

       if key == ord('q'):  # q: 退出
           return 'quit'
       elif key == 13:      # Enter: 下一张
           return 'next'
       elif key == ord('s'): # s: 保存当前
           self._save_current(image, window_name)
           return 'next'
       else:
           return 'next'  # 其他键也继续
   ```
2. **实现状态管理**：
   - 当前图片索引跟踪
   - 退出条件判断
   - 保存状态管理
3. **实现进度报告**：
   - 控制台进度条
   - 详细日志记录

### 阶段7：集成详细日志和错误处理
1. **集成日志系统**：
   ```python
   if self.verbose:
       self._logger = VerboseLoggingOperations().get_verbose_logger(
           "dataflow.visualize",
           verbose=True
       )
   else:
       self._logger = LoggingOperations().get_logger("dataflow.visualize")
   ```
2. **实现错误处理**：
   - 严格模式：错误时立即停止
   - 非严格模式：记录错误并继续
   - 结果中收集所有错误和警告
3. **实现性能优化**：
   - 图片懒加载
   - 批量保存优化
   - 内存使用监控

### 注意事项
1. **OpenCV 兼容性**：注意 BGR/RGB 颜色空间转换
2. **窗口管理**：多个窗口的创建和销毁
3. **内存管理**：大图片或批量处理时的内存使用
4. **用户交互**：响应式交互设计，避免界面卡顿

## 6. 开发要点

### 扩展指南
**添加新格式支持**：
1. 创建新的可视化器类继承 `BaseVisualizer`
2. 实现 `load_annotations()` 方法
3. 可选重写 `draw_annotations()` 实现自定义绘图
4. 在 CLI 模块中添加对应的命令

**自定义绘图效果**：
```python
class CustomVisualizer(BaseVisualizer):
    def draw_annotations(self, image, objects):
        # 调用父类方法绘制基础标注
        image = super().draw_annotations(image, objects)

        # 添加自定义效果
        for obj in objects:
            if obj.confidence:
                # 在边界框上显示置信度
                self._draw_confidence(image, obj)

        return image
```

**集成新的交互功能**：
1. 重写 `display_image()` 方法
2. 添加新的键盘快捷键处理
3. 更新交互状态管理

### 调试技巧
1. **启用详细日志**：`verbose=True` 查看处理详情
2. **单步调试模式**：设置 `is_show=True` 逐张检查
3. **颜色调试**：输出颜色分配信息验证一致性
4. **错误收集**：使用 `strict_mode=False` 收集所有问题再处理

### 性能优化
1. **批量处理优化**：
   - 禁用显示 (`is_show=False`) 提高速度
   - 使用多线程加载图片
   - 批量保存时优化文件IO
2. **内存优化**：
   - 及时释放不需要的图片数据
   - 使用生成器处理大型数据集
   - 调整 OpenCV 缓存设置
3. **绘图优化**：
   - 避免重复的颜色计算
   - 使用向量化操作
   - 优化文本渲染

## 7. 测试指南

### 测试策略
**单元测试目标**：
- 各个可视化器的标注加载功能
- 绘图逻辑的正确性
- 颜色管理的一致性
- 交互控制的响应
- 错误处理逻辑

**集成测试目标**：
- 与 label 模块的数据流集成
- 完整可视化流程
- 性能测试（批量处理）

### 测试数据准备
1. **小型可视化数据集**：
   ```
   test_visualize/
   ├── images/ (测试图片，3-5张即可)
   ├── yolo/ (YOLO标签)
   ├── labelme/ (LabelMe JSON)
   ├── coco/ (COCO JSON)
   └── classes.txt
   ```
2. **边界情况数据**：
   - 空标注图片
   - 密集标注图片
   - 大尺寸/小尺寸图片
   - 无效标注数据

### 示例测试用例

```python
# 测试 YOLO 可视化器
def test_yolo_visualizer_basic():
    visualizer = YOLOVisualizer(
        label_dir=TEST_YOLO_DIR,
        image_dir=TEST_IMAGE_DIR,
        class_file=TEST_CLASS_FILE,
        is_show=False,
        is_save=False
    )

    result = visualizer.visualize()

    assert result.success is True
    assert result.images_processed == len(list(TEST_IMAGE_DIR.glob("*.jpg")))
    assert len(result.errors) == 0

# 测试颜色管理一致性
def test_color_manager_consistency():
    color_mgr = ColorManager()

    # 相同 class_id 应返回相同颜色
    color1 = color_mgr.get_color(0)
    color2 = color_mgr.get_color(0)
    assert color1 == color2

    # 不同 class_id 应返回不同颜色
    color3 = color_mgr.get_color(1)
    assert color1 != color3

    # 重置后颜色可能变化（设计决定）
    color_mgr.reset()
    color4 = color_mgr.get_color(0)
    # 根据设计决定是否断言相等

# 测试交互控制
def test_display_image_interaction():
    visualizer = YOLOVisualizer(...)
    test_image = np.zeros((100, 100, 3), dtype=np.uint8)

    # 模拟不同键盘输入（需要模拟 cv2.waitKey）
    # 实际测试中可能需要模拟或跳过交互测试

# 测试错误处理
def test_visualizer_error_handling():
    visualizer = YOLOVisualizer(
        label_dir=Path("non_existent"),
        image_dir=TEST_IMAGE_DIR,
        class_file=TEST_CLASS_FILE,
        strict_mode=False  # 非严格模式
    )

    result = visualizer.visualize()

    assert result.success is False
    assert len(result.errors) > 0
    assert "not found" in result.errors[0].lower()

# 测试批量保存
def test_batch_save():
    output_dir = Path("test_output")
    if output_dir.exists():
        shutil.rmtree(output_dir)

    visualizer = YOLOVisualizer(
        label_dir=TEST_YOLO_DIR,
        image_dir=TEST_IMAGE_DIR,
        class_file=TEST_CLASS_FILE,
        is_show=False,
        is_save=True,
        output_dir=output_dir
    )

    result = visualizer.visualize()

    assert result.success is True
    saved_files = list(output_dir.glob("*.jpg")) + list(output_dir.glob("*.png"))
    assert len(saved_files) == result.images_processed

    # 清理
    shutil.rmtree(output_dir)
```

### 测试覆盖率目标
- 可视化器核心方法：100% 行覆盖
- 颜色管理逻辑：100% 分支覆盖
- 绘图功能：主要绘图路径覆盖
- 错误处理：所有错误路径测试
- 交互功能：基本交互流程测试

---

**文档版本**：1.0
**最后更新**：2026-03-29
**基于实现版本**：DataFlow-CV 当前实现