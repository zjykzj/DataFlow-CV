# DataFlow-CV CLI 模块开发规范

## 1. 项目概述

### 1.1 项目背景和目标

DataFlow-CV已经实现了三个核心模块：dataflow/label（标签格式读写）、dataflow/convert（格式转换）、dataflow/visualize（可视化）。这些模块提供了强大的API接口，但在实际使用中，用户需要一个简单易用的命令行界面来快速执行常见任务。

本模块的核心目标是实现一个标准化、用户友好的命令行界面（CLI），支持：
- 统一的命令行入口 `dataflow-cv`
- `<主任务> <子任务>` 的命令结构模式
- `visualize` 主任务：支持yolo、coco、labelme三种标签格式的可视化
- `convert` 主任务：支持六种格式转换（yolo2coco、yolo2labelme、coco2yolo、coco2labelme、labelme2yolo、labelme2coco）
- 薄包装层设计：CLI仅调用现有API，不重复实现业务逻辑
- 完整的帮助系统和错误提示
- 与现有模块一致的日志记录和错误处理机制

### 1.2 核心功能特性

1. **统一命令行入口**：单个命令 `dataflow-cv` 作为所有功能的入口点
2. **结构化命令模式**：`<主任务> <子任务>` 模式，清晰易记
3. **可视化功能**：支持三种格式的标签可视化，支持交互显示和保存图像
4. **格式转换功能**：支持六种格式间的双向转换，保持与现有API的完全兼容
5. **薄包装层设计**：CLI层仅负责参数解析和API调用，业务逻辑完全复用现有模块
6. **完整的帮助系统**：每个命令和子命令都有详细的帮助信息
7. **错误友好提示**：统一的错误处理，提供清晰的问题诊断信息
8. **日志记录集成**：支持现有日志系统，可配置详细程度
9. **跨平台兼容**：确保在Windows、Linux、macOS上的完全兼容

### 1.3 设计原则

1. **薄包装原则**：CLI仅作为现有API的包装层，不重复实现任何业务逻辑
2. **一致性原则**：保持与现有模块相同的错误处理、日志记录和配置方式
3. **用户友好原则**：清晰的帮助信息、合理的默认值、有意义的错误提示
4. **可测试性原则**：每个命令都可以独立测试，支持单元测试和集成测试
5. **可扩展性原则**：易于添加新的主任务和子任务支持
6. **平台兼容原则**：使用 `pathlib.Path` 进行路径操作，确保跨平台兼容性

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
│   ├── convert/                  # 现有转换模块
│   │   ├── base.py
│   │   ├── labelme_and_yolo.py
│   │   ├── yolo_and_coco.py
│   │   ├── coco_and_labelme.py
│   │   ├── utils.py
│   │   └── rle_converter.py
│   ├── visualize/                # 现有可视化模块
│   │   ├── base.py
│   │   ├── labelme_visualizer.py
│   │   ├── yolo_visualizer.py
│   │   ├── coco_visualizer.py
│   │   └── utils.py
│   ├── cli/                      # 新CLI模块
│   │   ├── __init__.py
│   │   ├── main.py               # CLI入口点和命令组定义
│   │   ├── commands/             # 命令实现目录
│   │   │   ├── __init__.py
│   │   │   ├── visualize.py      # visualize命令实现
│   │   │   ├── convert.py        # convert命令实现
│   │   │   └── utils.py          # CLI工具函数
│   │   └── models.py             # CLI参数模型和配置类
│   └── util/                     # 现有工具模块
│       ├── file_util.py
│       └── logging_util.py
├── tests/
│   ├── __init__.py
│   ├── cli/                      # CLI模块测试
│   │   ├── __init__.py
│   │   ├── test_main.py
│   │   ├── test_visualize.py
│   │   ├── test_convert.py
│   │   └── test_utils.py
├── samples/
│   ├── cli/                      # CLI模块示例
│   │   ├── __init__.py
│   │   ├── visualize_demo.py
│   │   ├── convert_demo.py
│   │   └── full_cli_demo.py
└── assets
    └── test_data/                # 测试数据
```

### 2.2 文件树结构说明

- **dataflow/cli/**：CLI核心模块目录
  - `main.py`：CLI入口点，Click命令组定义
  - `commands/`：具体命令实现目录
    - `visualize.py`：visualize命令及其子命令实现
    - `convert.py`：convert命令及其子命令实现
    - `utils.py`：CLI工具函数（参数验证、路径处理等）
  - `models.py`：CLI参数模型和配置数据类
- **tests/cli/**：CLI模块测试目录
- **samples/cli/**：CLI模块示例代码目录
- **assets/test_data/**：测试数据资源目录

### 2.3 依赖关系

```
dataflow.cli
├── click>=7.0.0 (CLI框架)
├── dataflow.label (依赖)
├── dataflow.convert (依赖)
├── dataflow.visualize (依赖)
└── dataflow.util (依赖)
```

### 2.4 CLI在项目中的位置

CLI模块位于整个DataFlow-CV项目的最上层，作为用户与底层API之间的桥梁：

```
用户
  ↓
dataflow-cv CLI (本模块)
  ↓
dataflow.visualize ──┬──> dataflow.label
  ↓                   ↓
dataflow.convert ────┘
  ↓
底层API (OpenCV, numpy等)
```

## 3. dataflow/cli模块详细设计

### 3.1 命令结构设计

CLI采用两层命令结构：`<主任务> <子任务>` 模式，与用户需求完全对应：

```
dataflow-cv <主任务> <子任务> [参数...]
```

#### 3.1.1 主任务：visualize (可视化)

```
dataflow-cv visualize <format> <input_path> [选项...]
```

支持的子任务（format）：
- `yolo`：可视化YOLO格式标签
- `coco`：可视化COCO格式标签
- `labelme`：可视化LabelMe格式标签

#### 3.1.2 主任务：convert (格式转换)

```
dataflow-cv convert <direction> <input_path> <output_path> [选项...]
```

支持的子任务（direction）：
- `yolo2coco`：YOLO格式转COCO格式
- `yolo2labelme`：YOLO格式转LabelMe格式
- `coco2yolo`：COCO格式转YOLO格式
- `coco2labelme`：COCO格式转LabelMe格式
- `labelme2yolo`：LabelMe格式转YOLO格式
- `labelme2coco`：LabelMe格式转COCO格式

### 3.2 参数设计

#### 3.2.1 通用参数

所有命令都支持的参数：

| 参数 | 短选项 | 类型 | 默认值 | 描述 |
|------|--------|------|--------|------|
| `--verbose` | `-v` | 布尔 | `False` | 启用详细日志输出 |
| `--log-dir` | 无 | 路径 | `./logs` | 日志文件保存目录 |
| `--strict` | 无 | 布尔 | `True` | 严格模式（遇错停止） |
| `--help` | `-h` | 布尔 | 无 | 显示帮助信息 |

#### 3.2.2 visualize命令参数

| 参数 | 短选项 | 类型 | 默认值 | 描述 |
|------|--------|------|--------|------|
| `--image-dir` | `-i` | 路径 | 无 | 图像文件目录（如与标签文件分离） |
| `--output-dir` | `-o` | 路径 | 无 | 可视化结果保存目录 |
| `--display` | `-d` | 布尔 | `False` | 交互式显示结果（OpenCV窗口） |
| `--save` | `-s` | 布尔 | `True` | 保存可视化结果到文件 |
| `--color-scheme` | `-c` | 字符串 | `random` | 颜色方案：random/category/consistent |
| `--thickness` | 无 | 整数 | `2` | 边界框/多边形线宽 |

#### 3.2.3 convert命令参数

| 参数 | 短选项 | 类型 | 默认值 | 描述 |
|------|--------|------|--------|------|
| `--image-dir` | `-i` | 路径 | 无 | 图像文件目录（用于获取图像尺寸） |
| `--class-file` | `-c` | 路径 | 无 | 类别文件路径（YOLO格式需要） |
| `--do-rle` | 无 | 布尔 | `False` | 对COCO格式使用RLE编码（需要pycocotools） |
| `--category-mapping` | 无 | 路径 | 无 | 自定义类别映射文件（JSON格式） |
| `--skip-errors` | 无 | 布尔 | `False` | 跳过错误继续处理（宽松模式） |

### 3.3 帮助系统设计

每个命令和子命令都提供完整的帮助信息：

1. **顶层帮助**：`dataflow-cv --help`
   - 显示所有可用的主任务
   - 简要描述每个主任务的功能

2. **主任务帮助**：`dataflow-cv <主任务> --help`
   - 显示该主任务下所有可用的子任务
   - 描述子任务的用途和基本用法

3. **子任务帮助**：`dataflow-cv <主任务> <子任务> --help`
   - 显示该子任务的详细参数说明
   - 提供使用示例

### 3.4 错误处理设计

CLI错误处理分为三个层次：

1. **参数验证错误**：在命令执行前检查参数有效性
   - 路径不存在、格式不支持、参数冲突等
   - 提供清晰的错误信息和建议的解决方法

2. **运行时错误**：在API调用过程中出现的错误
   - 文件读写错误、格式解析错误、图像加载失败等
   - 根据`--strict`参数决定是否继续执行

3. **系统错误**：底层系统错误
   - 内存不足、权限错误等
   - 提供简洁的错误描述和退出码

### 3.5 退出码设计

| 退出码 | 含义 | 描述 |
|--------|------|------|
| `0` | 成功 | 所有操作成功完成 |
| `1` | 参数错误 | 命令行参数无效或缺失 |
| `2` | 输入错误 | 输入文件不存在或格式错误 |
| `3` | 输出错误 | 输出目录无法创建或写入 |
| `4` | 运行时错误 | API调用过程中出现错误 |
| `5` | 系统错误 | 内存不足、权限错误等系统问题 |

## 4. dataflow/cli/commands模块详细设计

### 4.1 核心模块设计

#### 4.1.1 main.py - CLI入口点

```python
# dataflow/cli/main.py
import click
from pathlib import Path
from dataflow.util.logging_util import LoggingOperations

@click.group()
@click.version_option()
@click.option('--verbose', '-v', is_flag=True, help='启用详细日志输出')
@click.option('--log-dir', type=click.Path(path_type=Path), default='./logs',
              help='日志文件保存目录')
@click.pass_context
def cli(ctx, verbose, log_dir):
    """DataFlow-CV命令行工具 - 计算机视觉数据处理工具集"""
    # 初始化上下文对象
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose
    ctx.obj['log_dir'] = log_dir

    # 配置日志
    if verbose:
        from dataflow.util.logging_util import VerboseLoggingOperations
        logger = VerboseLoggingOperations.get_verbose_logger(
            name='dataflow.cli',
            verbose=True,
            log_dir=log_dir
        )
    else:
        logger = LoggingOperations.get_logger('dataflow.cli')

    ctx.obj['logger'] = logger

# 注册子命令组
from .commands import visualize, convert
cli.add_command(visualize.visualize_group)
cli.add_command(convert.convert_group)

if __name__ == '__main__':
    cli()
```

#### 4.1.2 commands/visualize.py - 可视化命令实现

```python
# dataflow/cli/commands/visualize.py
import click
from pathlib import Path
from typing import Optional

@click.group()
def visualize_group():
    """可视化命令组 - 支持多种标签格式的可视化"""
    pass

@visualize_group.command()
@click.argument('input_path', type=click.Path(exists=True, path_type=Path))
@click.option('--image-dir', '-i', type=click.Path(path_type=Path),
              help='图像文件目录（如与标签文件分离）')
@click.option('--output-dir', '-o', type=click.Path(path_type=Path),
              help='可视化结果保存目录')
@click.option('--display', '-d', is_flag=True, help='交互式显示结果')
@click.option('--save', '-s', is_flag=True, default=True,
              help='保存可视化结果到文件')
@click.pass_context
def yolo(ctx, input_path, image_dir, output_dir, display, save):
    """可视化YOLO格式标签"""
    from dataflow.visualize.yolo_visualizer import YOLOVisualizer
    from dataflow.cli.utils import validate_visualize_params

    logger = ctx.obj['logger']
    logger.info(f"开始可视化YOLO标签: {input_path}")

    # 参数验证
    validate_visualize_params(input_path, image_dir, output_dir)

    # 调用现有API
    visualizer = YOLOVisualizer()
    result = visualizer.visualize(
        annotation_path=input_path,
        image_dir=image_dir,
        output_dir=output_dir,
        display=display,
        save=save,
        verbose=ctx.obj['verbose']
    )

    logger.info(f"可视化完成: {result.summary}")

# 类似地实现coco和labelme子命令
```

#### 4.1.3 commands/convert.py - 转换命令实现

```python
# dataflow/cli/commands/convert.py
import click
from pathlib import Path

@click.group()
def convert_group():
    """格式转换命令组 - 支持多种标签格式间的转换"""
    pass

@convert_group.command()
@click.argument('input_path', type=click.Path(exists=True, path_type=Path))
@click.argument('output_path', type=click.Path(path_type=Path))
@click.option('--image-dir', '-i', type=click.Path(path_type=Path),
              help='图像文件目录（用于获取图像尺寸）')
@click.option('--class-file', '-c', type=click.Path(path_type=Path),
              help='类别文件路径（YOLO格式需要）')
@click.option('--do-rle', is_flag=True, help='对COCO格式使用RLE编码')
@click.pass_context
def yolo2coco(ctx, input_path, output_path, image_dir, class_file, do_rle):
    """YOLO格式转COCO格式"""
    from dataflow.convert.yolo_and_coco import YoloAndCocoConverter
    from dataflow.cli.utils import validate_convert_params

    logger = ctx.obj['logger']
    logger.info(f"开始转换YOLO到COCO: {input_path} -> {output_path}")

    # 参数验证
    validate_convert_params('yolo', 'coco', input_path, output_path,
                           image_dir, class_file)

    # 调用现有API
    converter = YoloAndCocoConverter()
    result = converter.yolo_to_coco(
        yolo_dir=input_path,
        output_path=output_path,
        image_dir=image_dir,
        class_file=class_file,
        do_rle=do_rle,
        strict=not ctx.obj.get('skip_errors', False)
    )

    logger.info(f"转换完成: {result.summary}")

# 类似地实现其他5种转换子命令
```

#### 4.1.4 models.py - CLI参数模型

```python
# dataflow/cli/models.py
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any

@dataclass
class CLIConfig:
    """CLI配置数据类"""
    verbose: bool = False
    log_dir: Path = Path('./logs')
    strict: bool = True
    skip_errors: bool = False

@dataclass
class VisualizeParams:
    """可视化参数数据类"""
    format: str  # yolo/coco/labelme
    input_path: Path
    image_dir: Optional[Path] = None
    output_dir: Optional[Path] = None
    display: bool = False
    save: bool = True
    color_scheme: str = 'random'
    thickness: int = 2

@dataclass
class ConvertParams:
    """转换参数数据类"""
    direction: str  # yolo2coco等
    input_path: Path
    output_path: Path
    image_dir: Optional[Path] = None
    class_file: Optional[Path] = None
    do_rle: bool = False
    category_mapping: Optional[Dict[str, Any]] = None
```

### 4.2 与现有API的集成策略

CLI模块作为薄包装层，需要与现有三个模块紧密集成：

#### 4.2.1 与label模块集成

- **数据读取**：使用`BaseAnnotationHandler`子类读取标签文件
- **参数传递**：将CLI参数映射到handler的构造函数参数
- **错误处理**：捕获handler抛出的异常，转换为用户友好的错误信息

#### 4.2.2 与convert模块集成

- **转换器选择**：根据`<direction>`参数选择对应的转换器
- **参数映射**：将CLI参数映射到converter的方法参数
- **结果处理**：处理`ConversionResult`，生成用户友好的输出

#### 4.2.3 与visualize模块集成

- **可视化器选择**：根据`<format>`参数选择对应的visualizer
- **显示控制**：处理`--display`和`--save`参数
- **颜色管理**：传递`--color-scheme`参数到`ColorManager`

#### 4.2.4 与util模块集成

- **日志记录**：使用`LoggingOperations`和`VerboseLoggingOperations`
- **文件操作**：使用`FileOperations`进行安全的文件操作
- **路径处理**：使用`pathlib.Path`确保跨平台兼容性

### 4.3 工具函数设计

#### 4.3.1 utils.py - CLI工具函数

```python
# dataflow/cli/utils.py
from pathlib import Path
from typing import Optional, Tuple
import click

def validate_path_exists(path: Path, name: str = "路径"):
    """验证路径是否存在"""
    if not path.exists():
        raise click.BadParameter(f"{name}不存在: {path}")
    return path

def validate_visualize_params(
    input_path: Path,
    image_dir: Optional[Path],
    output_dir: Optional[Path]
) -> Tuple[Path, Optional[Path], Optional[Path]]:
    """验证可视化参数"""
    input_path = validate_path_exists(input_path, "输入路径")

    if image_dir:
        image_dir = validate_path_exists(image_dir, "图像目录")

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    return input_path, image_dir, output_dir

def validate_convert_params(
    source_format: str,
    target_format: str,
    input_path: Path,
    output_path: Path,
    image_dir: Optional[Path],
    class_file: Optional[Path]
) -> Tuple[Path, Path, Optional[Path], Optional[Path]]:
    """验证转换参数"""
    input_path = validate_path_exists(input_path, "输入路径")

    # 确保输出目录存在
    if output_path.suffix:  # 是文件
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:  # 是目录
        output_path.mkdir(parents=True, exist_ok=True)

    if image_dir:
        image_dir = validate_path_exists(image_dir, "图像目录")

    if class_file:
        class_file = validate_path_exists(class_file, "类别文件")

    return input_path, output_path, image_dir, class_file
```

## 5. 技术实现要求

### 5.1 Click框架最佳实践

1. **命令组织**：使用`@click.group()`创建命令组，实现层次化命令结构
2. **参数装饰器**：使用`@click.option()`和`@click.argument()`定义参数
3. **上下文传递**：使用`@click.pass_context`和`ctx.obj`传递共享数据
4. **类型转换**：使用`type=click.Path(path_type=Path)`确保路径类型安全
5. **帮助文本**：为每个命令和参数提供详细的中文帮助文本

### 5.2 错误处理最佳实践

1. **参数验证**：在命令执行前验证所有参数的有效性
2. **异常转换**：将底层异常转换为用户友好的错误信息
3. **退出码**：使用标准退出码表示不同的错误类型
4. **错误恢复**：根据`--strict`参数决定错误处理策略
5. **日志记录**：所有错误都记录到日志文件，便于调试

### 5.3 日志记录集成

1. **双模式日志**：普通模式和详细模式，通过`--verbose`切换
2. **文件日志**：详细模式下自动生成带时间戳的日志文件
3. **结构化日志**：使用现有`LoggingOperations`的格式化输出
4. **日志轮转**：考虑未来添加日志轮转功能

### 5.4 跨平台兼容性

1. **路径处理**：使用`pathlib.Path`替代字符串路径操作
2. **编码处理**：强制使用UTF-8编码读写文件
3. **路径分隔符**：使用`/`作为路径分隔符，`pathlib`会自动转换
4. **权限处理**：正确处理不同平台的文件权限问题

## 6. 测试策略

### 6.1 测试目标

- **功能正确性**：确保每个CLI命令正确调用底层API
- **参数验证**：测试参数验证逻辑的正确性
- **错误处理**：测试各种错误场景的处理
- **帮助系统**：测试帮助信息的完整性和准确性
- **集成测试**：测试CLI与现有模块的集成

### 6.2 测试类型

#### 6.2.1 单元测试

测试CLI模块内部的各个组件：

```python
# tests/cli/test_utils.py
def test_validate_path_exists():
    """测试路径验证函数"""
    with tempfile.TemporaryDirectory() as tmpdir:
        existing_path = Path(tmpdir) / "test.txt"
        existing_path.touch()

        # 测试存在的路径
        result = validate_path_exists(existing_path)
        assert result == existing_path

        # 测试不存在的路径
        non_existent = Path(tmpdir) / "nonexistent.txt"
        with pytest.raises(click.BadParameter):
            validate_path_exists(non_existent)
```

#### 6.2.2 命令测试

使用Click的`CliRunner`测试命令执行：

```python
# tests/cli/test_visualize.py
from click.testing import CliRunner

def test_visualize_yolo_command():
    """测试visualize yolo命令"""
    runner = CliRunner()
    result = runner.invoke(
        visualize_group,
        ['yolo', 'test_path', '--help']
    )
    assert result.exit_code == 0
    assert '可视化YOLO格式标签' in result.output
```

#### 6.2.3 集成测试

测试CLI与现有API的集成：

```python
# tests/cli/test_integration.py
def test_yolo2coco_integration():
    """测试yolo2coco命令与底层API的集成"""
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmpdir:
        # 准备测试数据
        input_dir = prepare_test_yolo_data(tmpdir)
        output_file = Path(tmpdir) / "output.json"

        # 执行命令
        result = runner.invoke(
            convert_group,
            ['yolo2coco', str(input_dir), str(output_file)]
        )

        assert result.exit_code == 0
        assert output_file.exists()
```

### 6.3 测试数据

使用现有`assets/test_data/`目录中的测试数据：

- `assets/test_data/det/`：目标检测测试数据
- `assets/test_data/seg/`：实例分割测试数据
- 每种格式（yolo/coco/labelme）都有对应的测试数据

### 6.4 测试覆盖率目标

- **行覆盖率**：≥90%
- **分支覆盖率**：≥85%
- **命令覆盖率**：100%（所有命令都有测试）
- **参数组合覆盖率**：覆盖主要参数组合

## 7. 示例代码

### 7.1 基本使用示例

#### 7.1.1 可视化示例

```bash
# 可视化YOLO标签（交互式显示）
dataflow-cv visualize yolo /path/to/yolo/labels --image-dir /path/to/images --display

# 可视化COCO标签（保存到文件）
dataflow-cv visualize coco /path/to/coco/annotations.json --output-dir /path/to/output --save

# 可视化LabelMe标签（详细日志）
dataflow-cv visualize labelme /path/to/labelme/files -v --log-dir ./my_logs
```

#### 7.1.2 格式转换示例

```bash
# YOLO转COCO（使用RLE编码）
dataflow-cv convert yolo2coco /path/to/yolo /path/to/output.json --do-rle

# COCO转YOLO（指定类别文件）
dataflow-cv convert coco2yolo /path/to/coco.json /path/to/yolo --class-file classes.txt

# LabelMe转COCO（宽松模式，跳过错误）
dataflow-cv convert labelme2coco /path/to/labelme /path/to/output.json --skip-errors
```

### 7.2 进阶使用示例

#### 7.2.1 批量处理示例

```bash
#!/bin/bash
# 批量转换多个YOLO数据集到COCO格式
for dataset in dataset1 dataset2 dataset3; do
    echo "处理数据集: $dataset"
    dataflow-cv convert yolo2coco \
        "data/$dataset/yolo" \
        "data/$dataset/annotations.json" \
        --image-dir "data/$dataset/images" \
        --class-file "data/$dataset/classes.txt" \
        --verbose
done
```

#### 7.2.2 管道使用示例

```bash
# 将YOLO转COCO，然后可视化结果
dataflow-cv convert yolo2coco input/yolo output/annotations.json && \
dataflow-cv visualize coco output/annotations.json --image-dir input/images
```

### 7.3 配置示例

#### 7.3.1 自定义日志配置

```bash
# 使用自定义日志目录和详细输出
dataflow-cv --verbose --log-dir ./custom_logs visualize yolo /path/to/labels
```

#### 7.3.2 类别映射文件示例

```json
// category_mapping.json
{
  "person": "human",
  "car": "vehicle",
  "dog": "animal",
  "cat": "animal"
}
```

```bash
# 使用自定义类别映射
dataflow-cv convert yolo2coco input/yolo output.json --category-mapping category_mapping.json
```

## 8. 开发计划

### 8.1 开发阶段划分（共5个阶段）

#### 阶段一：基础框架搭建（预计2-3天）
**目标**：建立CLI模块基础架构，实现Click框架集成和日志系统

**任务清单**：
1. 创建CLI模块目录结构（`dataflow/cli/`, `dataflow/cli/commands/`）
2. 实现Click基础框架（顶层`cli()`命令组）
3. 添加基本参数处理（--verbose, --log-dir, --strict等）
4. 集成现有日志系统（`LoggingOperations`和`VerboseLoggingOperations`）
5. 实现上下文传递机制（`ctx.obj`）
6. 创建参数模型类（`models.py`）

**验收标准**：
- `dataflow-cv --help`正常显示基础帮助信息
- `dataflow-cv --version`正常显示版本信息
- 通用参数（--verbose, --log-dir）解析正确
- 日志系统集成正常，支持双模式日志输出
- 代码结构清晰，符合项目规范

#### 阶段二：可视化命令实现（预计3-4天）
**目标**：实现`visualize`命令组及三个子命令

**任务清单**：
1. 创建`commands/visualize.py`文件
2. 定义`visualize_group`命令组
3. 实现`yolo`子命令：参数解析、验证、调用`YOLOVisualizer`
4. 实现`coco`子命令：参数解析、验证、调用`COCOVisualizer`
5. 实现`labelme`子命令：参数解析、验证、调用`LabelMeVisualizer`
6. 添加参数验证工具函数（`validate_visualize_params`等）
7. 实现可视化特定参数（--image-dir, --output-dir, --display, --save等）

**验收标准**：
- `dataflow-cv visualize --help`显示完整帮助信息
- 三个子命令（yolo, coco, labelme）都能正常执行
- 参数验证功能正常，提供清晰错误提示
- 正确调用底层visualize API，输出结果一致
- 支持交互显示和文件保存两种模式

#### 阶段三：转换命令实现（预计4-5天）
**目标**：实现`convert`命令组及六种转换子命令

**任务清单**：
1. 创建`commands/convert.py`文件
2. 定义`convert_group`命令组
3. 实现6种转换子命令：
   - `yolo2coco`：调用`YoloAndCocoConverter.yolo_to_coco()`
   - `yolo2labelme`：调用`LabelMeAndYoloConverter.yolo_to_labelme()`
   - `coco2yolo`：调用`YoloAndCocoConverter.coco_to_yolo()`
   - `coco2labelme`：调用`CocoAndLabelMeConverter.coco_to_labelme()`
   - `labelme2yolo`：调用`LabelMeAndYoloConverter.labelme_to_yolo()`
   - `labelme2coco`：调用`CocoAndLabelMeConverter.labelme_to_coco()`
4. 处理格式特定参数（--class-file, --do-rle, --category-mapping等）
5. 实现参数验证工具函数（`validate_convert_params`等）
6. 实现类别映射功能（JSON文件支持）

**验收标准**：
- `dataflow-cv convert --help`显示完整帮助信息
- 六种子命令都能正常执行，参数解析正确
- 格式特定参数处理正常（如YOLO需要--class-file，COCO支持--do-rle）
- 正确调用底层convert API，转换结果准确
- 类别映射功能正常，支持自定义JSON映射文件

#### 阶段四：测试和完善（预计3-4天）
**目标**：编写全面的测试套件，完善错误处理和帮助系统

**任务清单**：
1. 创建`tests/cli/`测试目录结构
2. 编写单元测试：
   - `test_main.py`：测试CLI基础框架
   - `test_visualize.py`：测试可视化命令
   - `test_convert.py`：测试转换命令
   - `test_utils.py`：测试工具函数
3. 编写集成测试：
   - `test_integration.py`：测试CLI与底层API的集成
   - 测试错误处理场景和参数组合
4. 完善帮助系统：
   - 为所有命令和参数添加详细的中文帮助文本
   - 提供使用示例和常见问题解答
5. 优化错误处理：
   - 统一错误处理模式，提供清晰的错误信息
   - 实现退出码系统（0-5）
6. 测试跨平台兼容性（Windows、Linux、macOS）

**验收标准**：
- 单元测试覆盖率≥90%，集成测试通过率100%
- 所有命令都有完整的测试用例
- 帮助信息完整准确，包含使用示例
- 错误处理完善，提供有用的诊断信息
- 跨平台兼容性验证通过

#### 阶段五：示例和文档（预计2-3天）
**目标**：创建示例脚本，完善文档，完成项目集成

**任务清单**：
1. 创建`samples/cli/`示例目录
2. 编写示例脚本：
   - `visualize_demo.py`：可视化功能示例
   - `convert_demo.py`：格式转换示例
   - `full_cli_demo.py`：完整CLI使用示例
3. 更新项目文档：
   - 更新`README.md`添加CLI使用说明
   - 编写用户指南（快速入门、高级用法）
4. 项目集成：
   - 修改`setup.py`添加`entry_points`配置
   - 测试命令行安装和`dataflow-cv`命令可用性
5. 最终代码审查和质量检查

**验收标准**：
- 示例脚本可正常运行，展示核心功能
- 项目文档完整，包含CLI模块详细说明
- `dataflow-cv`命令可通过pip安装后直接使用
- 代码质量符合项目规范，通过所有检查
- 用户可按照文档快速上手使用CLI

### 8.2 详细任务清单（按阶段分解）

#### 阶段一详细任务
- [ ] 创建目录：`dataflow/cli/`, `dataflow/cli/commands/`, `tests/cli/`, `samples/cli/`
- [ ] 编写`dataflow/cli/__init__.py`导出必要接口
- [ ] 实现`main.py`：顶层`cli()`命令组，通用参数，上下文传递
- [ ] 实现`models.py`：`CLIConfig`, `VisualizeParams`, `ConvertParams`数据类
- [ ] 集成日志系统：支持`--verbose`双模式日志
- [ ] 编写`test_main.py`基础测试
- [ ] 验证`dataflow-cv --help`和`--version`正常显示

#### 阶段二详细任务
- [ ] 创建`commands/visualize.py`文件
- [ ] 定义`visualize_group`命令组和三个子命令装饰器
- [ ] 实现`yolo`子命令：参数解析、验证、调用`YOLOVisualizer.visualize()`
- [ ] 实现`coco`子命令：参数解析、验证、调用`COCOVisualizer.visualize()`
- [ ] 实现`labelme`子命令：参数解析、验证、调用`LabelMeVisualizer.visualize()`
- [ ] 创建`commands/utils.py`：实现`validate_visualize_params()`等工具函数
- [ ] 编写`test_visualize.py`单元测试
- [ ] 验证`dataflow-cv visualize --help`和各子命令功能

#### 阶段三详细任务
- [ ] 创建`commands/convert.py`文件
- [ ] 定义`convert_group`命令组和六个子命令装饰器
- [ ] 实现`yolo2coco`子命令：参数解析、验证、调用`YoloAndCocoConverter.yolo_to_coco()`
- [ ] 实现`yolo2labelme`子命令：参数解析、验证、调用`LabelMeAndYoloConverter.yolo_to_labelme()`
- [ ] 实现`coco2yolo`子命令：参数解析、验证、调用`YoloAndCocoConverter.coco_to_yolo()`
- [ ] 实现`coco2labelme`子命令：参数解析、验证、调用`CocoAndLabelMeConverter.coco_to_labelme()`
- [ ] 实现`labelme2yolo`子命令：参数解析、验证、调用`LabelMeAndYoloConverter.labelme_to_yolo()`
- [ ] 实现`labelme2coco`子命令：参数解析、验证、调用`CocoAndLabelMeConverter.labelme_to_coco()`
- [ ] 扩展`commands/utils.py`：实现`validate_convert_params()`等工具函数
- [ ] 实现类别映射功能：支持`--category-mapping`参数和JSON文件
- [ ] 编写`test_convert.py`单元测试
- [ ] 验证`dataflow-cv convert --help`和各子命令功能

#### 阶段四详细任务
- [ ] 完善`tests/cli/`测试套件：
  - [ ] `test_main.py`：覆盖所有基础功能
  - [ ] `test_visualize.py`：覆盖所有可视化场景
  - [ ] `test_convert.py`：覆盖所有转换场景
  - [ ] `test_utils.py`：覆盖所有工具函数
  - [ ] `test_integration.py`：集成测试CLI与底层API
- [ ] 完善帮助系统：
  - [ ] 提供使用示例和常见问题解答
- [ ] 优化错误处理：
  - [ ] 统一错误处理模式，使用装饰器或中间件
  - [ ] 实现退出码系统（0-5），提供清晰错误信息
- [ ] 测试跨平台兼容性：
  - [ ] Linux测试
- [ ] 确保测试覆盖率≥90%

#### 阶段五详细任务
- [ ] 创建示例脚本：
  - [ ] `samples/cli/visualize_demo.py`：展示可视化功能
  - [ ] `samples/cli/convert_demo.py`：展示格式转换功能
  - [ ] `samples/cli/full_cli_demo.py`：展示完整CLI使用流程
- [ ] 更新项目文档：
  - [ ] 更新`README.md`添加CLI模块介绍和使用说明
- [ ] 项目集成：
  - [ ] 修改`setup.py`添加`entry_points`配置
  - [ ] 测试`pip install -e .`后`dataflow-cv`命令可用性

### 8.3 关键检查点（里程碑）

1. **检查点1**：CLI基础框架完成（阶段一结束）
   - [ ] `dataflow-cv --help`正常显示基础帮助
   - [ ] `dataflow-cv --version`正常显示版本
   - [ ] 通用参数（--verbose, --log-dir）解析正确
   - [ ] 日志系统集成正常，支持双模式输出

2. **检查点2**：可视化命令完成（阶段二结束）
   - [ ] `dataflow-cv visualize --help`显示完整帮助
   - [ ] 三个子命令（yolo, coco, labelme）都能正常执行
   - [ ] 参数验证功能正常，错误提示清晰
   - [ ] 与现有visualize API集成正常，输出一致

3. **检查点3**：转换命令完成（阶段三结束）
   - [ ] `dataflow-cv convert --help`显示完整帮助
   - [ ] 六个子命令都能正常执行，参数解析正确
   - [ ] 格式特定参数处理正常（--class-file, --do-rle等）
   - [ ] 与现有convert API集成正常，转换结果准确

4. **检查点4**：测试覆盖达标（阶段四结束）
   - [ ] 所有命令都有完整的测试用例

5. **检查点5**：示例和文档完成（阶段五结束）
   - [ ] 示例脚本能正常运行，展示核心功能
   - [ ] `dataflow-cv`命令可通过pip安装使用
   - [ ] 用户指南清晰易懂，方便快速上手

### 8.4 风险评估与应对措施

#### 风险1：Click框架兼容性问题
- **风险描述**：Click版本兼容性问题或平台特定行为差异
- **影响程度**：中
- **发生概率**：低
- **应对措施**：
  - 指定Click>=7.0.0版本要求
  - 使用Click标准API，避免使用实验性功能
  - 充分测试跨平台兼容性（Windows、Linux、macOS）
  - 提供清晰的安装说明和依赖管理

#### 风险2：与现有API集成问题
- **风险描述**：CLI参数与底层API参数映射错误或行为不一致
- **影响程度**：高
- **发生概率**：中
- **应对措施**：
  - 仔细设计参数映射关系，确保一一对应
  - 编写充分的集成测试，验证CLI与API的一致性
  - 使用类型注解和dataclass确保参数类型安全
  - 实现参数验证和转换逻辑，处理边界情况

#### 风险3：错误处理复杂度过高
- **风险描述**：多层错误处理逻辑复杂，难以维护和测试
- **影响程度**：中
- **发生概率**：中
- **应对措施**：
  - 统一错误处理模式，使用装饰器或中间件简化
  - 实现清晰的错误层次结构，区分参数错误、运行时错误、系统错误
  - 提供详细的错误信息和解决建议
  - 编写全面的错误处理测试用例

#### 风险4：用户学习成本过高
- **风险描述**：命令结构复杂，参数众多，用户难以掌握
- **影响程度**：中
- **发生概率**：低
- **应对措施**：
  - 设计直观的命令结构（`<主任务> <子任务>`模式）
  - 提供清晰的帮助信息和使用示例
  - 设置合理的默认参数，减少必需参数数量
  - 编写详细的用户指南和快速入门文档

#### 风险5：性能问题
- **风险描述**：CLI启动慢或命令执行效率低
- **影响程度**：低
- **发生概率**：低
- **应对措施**：
  - 优化导入延迟，使用惰性导入
  - 避免不必要的初始化操作
  - 进行性能测试，确保响应时间可接受
  - 提供进度指示和状态反馈

#### 风险6：跨平台兼容性问题
- **风险描述**：在不同操作系统上行为不一致
- **影响程度**：高
- **发生概率**：中
- **应对措施**：
  - 统一使用`pathlib.Path`进行路径操作
  - 强制使用UTF-8编码读写文件
  - 避免使用平台特定的系统调用
  - 进行全面跨平台测试

### 8.5 成功标准（验收条件）

1. **功能完整性**：
   - [ ] 所有规划的命令和功能都实现（2个主任务，9个子命令）
   - [ ] 参数系统完整，支持所有必需和可选参数
   - [ ] 与现有三大模块（label、convert、visualize）完全集成

2. **正确性**：
   - [ ] CLI正确调用底层API，输出结果与直接使用API一致
   - [ ] 参数验证准确，错误处理正确
   - [ ] 格式转换准确，可视化显示正确

3. **可用性**：
   - [ ] 命令行界面直观易用，符合用户习惯
   - [ ] 帮助信息完整准确，包含使用示例
   - [ ] 错误信息清晰，提供有用的诊断信息
   - [ ] 响应时间可接受，用户体验良好

4. **稳定性**：
   - [ ] 在各种输入和环境条件下都能稳定运行
   - [ ] 无内存泄漏或资源泄露
   - [ ] 错误恢复机制完善，不会因单个错误导致整体失败

5. **可维护性**：
   - [ ] 代码结构清晰，模块化设计
   - [ ] 符合项目代码规范，通过所有质量检查
   - [ ] 测试覆盖率高，便于后续维护和扩展
   - [ ] 文档完整，便于其他开发者理解

6. **文档完整性**：
   - [ ] CLI模块有完整的开发文档
   - [ ] 用户文档详细，包含快速入门和高级用法
   - [ ] 示例代码可运行，展示核心功能
   - [ ] API文档完整，便于二次开发

7. **性能指标**：
   - [ ] CLI启动时间<1秒
   - [ ] 命令响应时间可接受（可视化<5秒，转换<30秒）
   - [ ] 内存使用合理，无内存泄漏
   - [ ] 支持大文件处理，性能衰减可接受

## 9. 质量保证

### 9.1 代码质量标准

1. **代码规范**：
   - 遵循PEP 8编码规范
   - 使用Black进行代码格式化
   - 使用isort进行导入排序
   - 使用flake8进行代码风格检查

2. **类型安全**：
   - 所有函数和方法都有类型注解
   - 使用mypy进行类型检查，确保类型安全
   - 复杂数据使用dataclass进行封装

3. **文档要求**：
   - 所有公开API都有docstring
   - 使用Google风格docstring格式
   - CLI命令帮助文本完整准确

### 9.2 测试质量标准

1. **测试覆盖率**：
   - 单元测试覆盖率≥90%
   - 分支覆盖率≥85%
   - 所有公共API都有测试用例

2. **测试类型**：
   - 单元测试：测试单个函数或类的功能
   - 集成测试：测试CLI与底层API的集成
   - 端到端测试：测试完整命令行工作流

3. **测试数据**：
   - 使用标准测试数据集
   - 测试数据涵盖各种边界情况
   - 测试数据与生产数据隔离

### 9.3 安全与兼容性

1. **安全性**：
   - 所有路径操作使用pathlib.Path，防止路径注入
   - 文件读写使用UTF-8编码，防止编码问题
   - 输入参数进行严格验证

2. **兼容性**：
   - 支持Python 3.8+版本
   - 支持Windows、Linux、macOS三大平台
   - 依赖版本明确指定，避免隐式依赖

3. **性能要求**：
   - CLI启动时间<1秒
   - 命令响应时间可接受
   - 内存使用合理

### 9.4 维护性要求

1. **代码结构**：
   - 模块化设计，高内聚低耦合
   - 清晰的目录结构和文件命名
   - 遵循单一职责原则

2. **可扩展性**：
   - 易于添加新的命令和子命令
   - 参数系统灵活可扩展
   - 与现有模块松耦合

3. **文档维护**：
   - 代码变更同步更新文档
   - 保持示例代码的时效性
   - 定期更新用户指南

## 10. 总结

本规范详细设计了DataFlow-CV CLI模块的开发方案。CLI作为现有API的薄包装层，采用`<主任务> <子任务>`的命令结构，提供`visualize`和`convert`两大功能，支持三种标签格式的可视化和六种格式转换。

开发过程将遵循薄包装原则、一致性原则和用户友好原则，确保CLI模块与现有代码库无缝集成。通过详细的开发计划和测试策略，确保项目按时高质量完成。

CLI模块的完成将使DataFlow-CV项目更加完整，为用户提供更加便捷的使用体验，同时为项目的进一步发展和推广奠定坚实基础。