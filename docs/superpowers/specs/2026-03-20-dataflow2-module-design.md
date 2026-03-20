# dataflow2模块设计规范

**日期**: 2026-03-20
**状态**: 草案
**版本**: 1.0

## 概述

dataflow2模块是dataflow模块的完全重构版本，旨在提供更统一、健壮且功能完整的计算机视觉数据集处理库。本设计规范详细描述了dataflow2模块的架构、组件和实现细节。

## 设计目标

1. **统一API设计**: 所有格式处理器遵循相同的接口规范
2. **完整功能覆盖**: 支持目标检测和实例分割标注，包括RLE格式
3. **健壮错误处理**: 严格错误处理策略，遇到错误立即停止
4. **模块化设计**: 清晰的模块边界，便于维护和扩展
5. **逐步替代**: 最终替代现有dataflow模块的全部功能

## 总体架构

```
dataflow2/
├── __init__.py              # 包导出
├── label/                   # 标签处理模块
│   ├── __init__.py          # 导出所有处理器类
│   ├── base.py              # BaseAnnotationHandler基类
│   ├── yolo.py              # YOLO格式处理器
│   ├── coco.py              # COCO格式处理器
│   └── labelme.py           # LabelMe格式处理器
└── util/                    # 工具模块
    ├── __init__.py          # 导出工具类/函数
    ├── file_util.py         # 文件操作工具
    └── logging_util.py      # 日志配置工具
```

## util模块设计

### file_util.py - 文件操作工具

提供基础的文件读写和批量处理功能：

```python
class FileOperations:
    """文件操作工具类"""

    @staticmethod
    def read_file(file_path: str, encoding='utf-8') -> str:
        """读取文本文件"""

    @staticmethod
    def write_file(file_path: str, content: str, encoding='utf-8') -> bool:
        """写入文本文件"""

    @staticmethod
    def ensure_directory(dir_path: str) -> bool:
        """确保目录存在"""

    @staticmethod
    def list_files(directory: str, pattern: str = "*") -> List[str]:
        """列出匹配模式的文件"""

    @staticmethod
    def batch_read(files: List[str], on_error='strict') -> List[str]:
        """批量读取文件（严格模式：遇到错误立即停止）"""
```

### logging_util.py - 日志配置工具

提供统一的日志配置：

```python
class LoggingOperations:
    """日志配置工具类"""

    @staticmethod
    def setup_logger(name: str, verbose: bool = False) -> logging.Logger:
        """设置和配置日志记录器"""

    @staticmethod
    def configure_logging(level=logging.WARNING, format_str=None):
        """全局日志配置"""
```

## label模块设计

### BaseAnnotationHandler基类

所有格式处理器的统一基类：

```python
class BaseAnnotationHandler(ABC):
    """标注处理器基类（抽象类）"""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.logger = LoggingOperations.setup_logger(
            self.__class__.__name__, verbose
        )

    @abstractmethod
    def read(self, label_path: str, **kwargs) -> Dict:
        """读取单个标签文件

        Args:
            label_path: 标签文件路径
            **kwargs: 格式特定的参数（如图像路径、类别文件等）

        Returns:
            统一格式的图像标注数据字典

        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 文件格式错误
        """

    @abstractmethod
    def read_batch(self, **kwargs) -> List[Dict]:
        """批量读取标签文件

        Args:
            **kwargs: 格式特定的批量读取参数

        Returns:
            图像标注数据列表

        Raises:
            FileNotFoundError: 目录或文件不存在
            ValueError: 批量读取过程中发生错误
        """

    @abstractmethod
    def write(self, image_annotations: Dict, output_path: str, **kwargs) -> bool:
        """写入单个标签文件

        Args:
            image_annotations: 统一格式的图像标注数据
            output_path: 输出文件路径
            **kwargs: 格式特定的写入参数

        Returns:
            是否写入成功

        Raises:
            ValueError: 数据格式错误
        """

    @abstractmethod
    def write_batch(self, images_annotations: List[Dict], **kwargs) -> bool:
        """批量写入标签文件

        Args:
            images_annotations: 图像标注数据列表
            **kwargs: 格式特定的批量写入参数

        Returns:
            是否全部写入成功
        """

    def _validate_input_path(self, path: str) -> bool:
        """验证输入路径"""

    def _validate_output_path(self, path: str) -> bool:
        """验证输出路径"""
```

### 统一数据格式

所有处理器使用相同的数据格式：

```python
{
    "image_id": str,           # 图像ID（通常为文件名）
    "image_path": str,         # 图像文件路径
    "width": int,              # 图像宽度
    "height": int,             # 图像高度
    "annotations": [           # 标注列表
        {
            "category_id": int,            # 类别ID
            "category_name": str,          # 类别名称
            "bbox": [x_min, y_min, width, height],  # 边界框（可选）
            "segmentation": [[x1, y1, x2, y2, ...]],  # 分割多边形（可选）
            "_is_rle": bool                # 是否为RLE格式（内部使用）
        }
    ]
}
```

### YoloAnnotationHandler设计

```python
class YoloAnnotationHandler(BaseAnnotationHandler):
    """YOLO格式处理器"""

    def read(self, label_path: str, image_path: str, classes: List[str],
             image_size: Optional[Tuple[int, int]] = None,
             is_seg: bool = False) -> Dict:
        """读取YOLO标签文件

        自动检测逻辑：
        - 如果is_seg=True：强制按分割格式解析
        - 如果is_seg=False：
          - 每行5个值（class_id + 4个坐标）→ 检测格式
          - 每行>=7个值（class_id + 6+个坐标）→ 分割格式
          - 其他格式 → 错误
        """

    def read_batch(self, labels_dir: str, images_dir: str, classes_path: str,
                   label_ext: str = ".txt",
                   image_exts: Tuple[str] = (".jpg", ".png", ".jpeg"),
                   is_seg: bool = False) -> List[Dict]:
        """批量读取YOLO标签文件"""

    def write(self, image_annotations: Dict, output_path: str,
              classes: List[str]) -> bool:
        """写入YOLO标签文件"""

    def write_batch(self, images_annotations: List[Dict],
                    output_dir: str, classes_path: str) -> bool:
        """批量写入YOLO标签文件"""
```

### CocoAnnotationHandler设计

```python
class CocoAnnotationHandler(BaseAnnotationHandler):
    """COCO格式处理器"""

    def read(self, json_path: str, is_seg: bool = False) -> Dict:
        """读取COCO JSON文件

        支持格式：
        - 多边形点列表格式
        - RLE掩码格式（自动检测和解码）

        自动检测逻辑：
        - 如果is_seg=True：只接受包含原始分割数据的标注
        - 如果is_seg=False：接受所有标注（包括从边界框生成的多边形）
        """

    def read_batch(self, json_path: str, image_dir: str = "",
                   is_seg: bool = False) -> List[Dict]:
        """批量处理COCO文件（从单个JSON文件）"""

    def write(self, coco_data: Dict, output_path: str,
              rle: bool = False) -> bool:
        """写入COCO JSON文件

        Args:
            rle: 是否将分割数据编码为RLE格式
                 - True: 多边形→RLE编码（非简单矩形）
                 - False: 保持多边形格式
        """

    def write_batch(self, unified_data: List[Dict], output_path: str,
                    info: Optional[Dict] = None,
                    licenses: Optional[List[Dict]] = None,
                    rle: bool = False,
                    classes: Optional[List[str]] = None) -> bool:
        """批量写入COCO文件（生成单个JSON文件）"""
```

### LabelMeAnnotationHandler设计

```python
class LabelMeAnnotationHandler(BaseAnnotationHandler):
    """LabelMe格式处理器"""

    def read(self, json_path: str, classes: Optional[List[str]] = None,
             is_seg: bool = False) -> Dict:
        """读取LabelMe JSON文件

        自动检测逻辑：
        - 如果is_seg=True：只接受shape_type="polygon"的标注
        - 如果is_seg=False：接受所有shape_type（"polygon"和"rectangle"）
        """

    def read_batch(self, labels_dir: str, classes_path: str,
                   label_ext: str = ".json",
                   is_seg: bool = False) -> List[Dict]:
        """批量读取LabelMe标签文件"""

    def write(self, image_annotations: Dict, output_path: str,
              classes: Optional[List[str]] = None) -> bool:
        """写入LabelMe JSON文件"""

    def write_batch(self, images_annotations: List[Dict],
                    output_dir: str, classes_path: str) -> bool:
        """批量写入LabelMe标签文件"""
```

## 错误处理策略

### 严格错误处理
- **单个文件操作**: 遇到任何错误立即引发异常
- **批量操作**: 遇到第一个错误立即停止并引发异常
- **错误类型**:
  - `FileNotFoundError`: 文件或目录不存在
  - `ValueError`: 文件格式错误、数据无效、参数错误
  - `RuntimeError`: 处理过程中的运行时错误

### 错误恢复
- 不提供自动错误恢复机制
- 调用者负责错误处理和重试
- 日志记录所有错误详情便于调试

## RLE支持实现

### COCO格式的RLE支持
1. **读取时自动检测**:
   - 检测RLE格式（字典包含'counts'和'size'键）
   - 自动解码RLE为多边形格式
   - 标记`_is_rle=True`供内部使用

2. **写入时可选编码**:
   - `rle=False`: 保持多边形格式
   - `rle=True`: 将多边形编码为RLE格式（非简单矩形）
   - 简单矩形多边形（从边界框生成）保持多边形格式

3. **依赖管理**:
   - 需要`pycocotools>=2.0.0`
   - 优雅降级：如果pycocotools不可用，RLE功能受限
   - 清晰的错误消息指导安装

## 测试策略

### 单元测试
- **文件**: `tests/dataflow2/test_label_*.py`, `tests/dataflow2/test_util_*.py`
- **覆盖范围**:
  - 各个处理器的读写功能
  - 错误处理（文件不存在、格式错误等）
  - 自动检测逻辑（检测vs分割）
  - RLE编码/解码（如果pycocotools可用）

### 集成测试
- **文件**: `tests/dataflow2/test_integration_*.py`
- **测试场景**:
  - 格式转换循环（A→B→A应保持一致性）
  - 批量处理完整性
  - 错误传播（批量操作中的错误处理）

### 测试数据
- 使用`assets/`目录中的样本数据
- 创建最小化的测试数据集
- 包含边界情况测试（空文件、无效格式等）

## 实现优先级

1. **第一阶段** (核心功能):
   - util模块实现（file_util.py, logging_util.py）
   - BaseAnnotationHandler基类
   - YoloAnnotationHandler基础功能

2. **第二阶段** (完整标签支持):
   - CocoAnnotationHandler（含RLE支持）
   - LabelMeAnnotationHandler

3. **第三阶段** (增强功能):
   - 完整的错误处理
   - 性能优化和批量处理改进
   - 测试覆盖

4. **第四阶段** (替代dataflow):
   - 实现与dataflow相同的CLI接口
   - 逐步替换现有dataflow模块

## 向后兼容性

**明确不兼容dataflow模块**：
- API完全重新设计
- 不提供自动迁移工具
- 用户需要更新代码以适应新API

## 性能考虑

- 批量操作使用迭代器而非一次性加载所有数据
- 大文件处理时使用流式读取
- 内存使用优化，避免不必要的数据复制
- 清晰的性能边界文档

---

**批准状态**: [ ] 设计已批准
**下一步**: 创建详细实现计划