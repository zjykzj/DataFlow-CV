# DataFlow-CV 命令行接口模块开发文档 (cli)

## 1. 模块概述

### 功能定位
`dataflow/cli` 模块是 DataFlow-CV 项目的命令行接口，为用户提供直观、高效的命令行操作界面。它基于 Click 框架构建，将 Python API 的功能封装成命令行工具，支持格式转换和可视化两大核心功能。

### 核心特性
- **完整的命令体系**：主命令 `dataflow-cv` 包含 `convert` 和 `visualize` 两个子命令组
- **6种双向转换命令**：`yolo2coco`、`yolo2labelme`、`coco2yolo`、`coco2labelme`、`labelme2yolo`、`labelme2coco`
- **3种可视化命令**：`yolo`、`coco`、`labelme` 标注可视化
- **统一的错误处理**：自定义异常类体系，分类型退出码（1-5）
- **详细的帮助系统**：格式化的参数显示，完善的帮助文档
- **日志集成**：支持 `--verbose` 标志，详细日志写入 `logs/` 目录
- **参数验证**：路径存在性检查，格式依赖验证

### 设计原则
1. **用户友好**：直观的命令结构，清晰的帮助信息
2. **一致性**：统一的参数命名，一致的错误处理
3. **API 封装**：薄封装层，重用现有 Python API 逻辑
4. **错误隔离**：明确的错误类型和退出码
5. **可扩展性**：易于添加新命令和子命令

## 2. 核心架构

### 类图/组件图
```
┌─────────────────────────────────────────┐
│           dataflow.cli                  │
├─────────────────────────────────────────┤
│  • 入口层                               │
│    - main.py (cli 主命令定义)           │
│      - 版本信息 (--version)             │
│      - 帮助系统 (--help)                │
│                                         │
│  • 命令层                               │
│    - commands/convert.py (转换命令组)   │
│      - 6个双向转换子命令                │
│    - commands/visualize.py (可视化组)   │
│      - 3个可视化子命令                  │
│                                         │
│  • 工具层                               │
│    - commands/utils.py (工具函数)       │
│      - 参数验证、装饰器、格式化命令类   │
│    - exceptions.py (异常类体系)         │
│      - CLIError、ParameterError 等      │
└─────────────────────────────────────────┘
```

### 数据流
1. **命令执行流程**：
   ```
   用户输入 → Click 解析 → 参数验证 → API调用 → 结果处理 → 输出反馈
                   ↳ 错误处理(退出码)      ↳ 日志记录
   ```

2. **日志配置流程**：
   ```
   主命令初始化 → 默认日志配置 → 子命令处理 → verbose检查 → 详细日志配置
                                        ↳ 文件日志记录
   ```

3. **错误处理流程**：
   ```
   异常发生 → 异常类型识别 → 转换为CLIError → 输出错误信息 → 返回退出码
              ↳ 参数错误      ↳ 输入错误      ↳ 运行时错误
   ```

### 依赖关系
- **内部依赖**：`dataflow/convert`、`dataflow/visualize`、`dataflow/util`
- **外部依赖**：`click>=7.0.0`（命令行框架）
- **被依赖**：无（顶层用户接口）

## 3. 主要API

### cli 主命令 (`main.py`)
**职责**：定义顶级命令组和全局选项

**核心配置**：
- `@click.group()`：定义命令组
- `--version` / `-v`：版本信息显示
- `--help` / `-h`：帮助信息显示

**上下文管理**：
```python
@click.pass_context
def cli(ctx):
    # 初始化上下文对象
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = False
    ctx.obj["log_dir"] = Path("./logs")
    ctx.obj["strict"] = True
    # 配置默认日志
    logger = LoggingOperations().get_logger("dataflow.cli")
    ctx.obj["logger"] = logger
```

### convert 命令组 (`commands/convert.py`)
**职责**：格式转换相关命令

**子命令结构**：
- `yolo2coco`：YOLO → COCO 转换
- `yolo2labelme`：YOLO → LabelMe 转换
- `coco2yolo`：COCO → YOLO 转换
- `coco2labelme`：COCO → LabelMe 转换
- `labelme2yolo`：LabelMe → YOLO 转换
- `labelme2coco`：LabelMe → COCO 转换

**参数模式**：
- 位置参数：`IMAGE_DIR`、`LABEL_DIR`、`CLASS_FILE`、`OUTPUT_FILE/DIR`
- 选项参数：`--do-rle`（RLE编码）、`--verbose`（详细日志）

### visualize 命令组 (`commands/visualize.py`)
**职责**：标注可视化相关命令

**子命令结构**：
- `yolo`：YOLO 格式可视化
- `coco`：COCO 格式可视化
- `labelme`：LabelMe 格式可视化

**参数模式**：
- 位置参数：`IMAGE_DIR`、`LABEL_DIR`/`COCO_FILE`
- 选项参数：`--save` / `-s`（保存目录）、`--verbose`（详细日志）

### 工具类 (`commands/utils.py`)
**职责**：提供 CLI 常用工具函数

**核心组件**：
- `add_common_options`：公共选项装饰器（`--verbose`）
- `add_visualize_options`：可视化专用选项装饰器
- `validate_convert_params`：转换参数验证
- `validate_visualize_params`：可视化参数验证
- `FormattedCommand`：格式化命令类（改善帮助显示）

### 异常体系 (`exceptions.py`)
**职责**：统一的错误处理和退出码管理

**异常类层次**：
- `CLIError`：基类（继承 `click.ClickException`）
- `ParameterError`：参数错误（退出码 1）
- `InputError`：输入文件错误（退出码 2）
- `OutputError`：输出文件错误（退出码 3）
- `RuntimeCLIError`：运行时错误（退出码 4）
- `SystemError`：系统错误（退出码 5）

## 4. 使用指南

### 快速开始

```bash
# 1. 查看帮助信息
dataflow-cv --help
dataflow-cv convert --help
dataflow-cv visualize --help

# 2. 查看版本信息
dataflow-cv --version

# 3. YOLO → COCO 转换
dataflow-cv convert yolo2coco images/ yolo_labels/ classes.txt coco_annotations.json

# 4. COCO 可视化（交互式）
dataflow-cv visualize coco images/ annotations.json

# 5. 批量可视化并保存结果
dataflow-cv visualize yolo images/ yolo_labels/ classes.txt --save visualized/
```

### 常见场景

#### 场景1：完整的转换工作流
```bash
# 工作流：YOLO → COCO → LabelMe → YOLO（无损验证）
# 步骤1：YOLO → COCO
dataflow-cv convert yolo2coco images/ yolo_labels/ classes.txt coco.json --verbose

# 步骤2：COCO → LabelMe
dataflow-cv convert coco2labelme coco.json labelme_output/

# 步骤3：LabelMe → YOLO（验证无损）
dataflow-cv convert labelme2yolo labelme_output/ classes.txt restored_yolo/
```

#### 场景2：批量处理和数据检查
```bash
# 启用详细日志，记录处理过程
dataflow-cv convert yolo2labelme images/ labels/ classes.txt output/ --verbose

# 检查日志文件（logs/ 目录）
ls -la logs/
cat logs/log_*.log | head -20

# 可视化检查标注质量
dataflow-cv visualize yolo images/ labels/ classes.txt --save quality_check/
```

#### 场景3：调试模式（详细日志）
```bash
# 转换时启用详细日志
dataflow-cv convert yolo2coco --verbose images/ labels/ classes.txt output.json

# 查看生成的日志文件
# logs/log_YYYYMMDD_HHMMSS.log 包含详细调试信息

# 可视化时启用详细日志
dataflow-cv visualize coco --verbose images/ annotations.json --save debug_output/
```

#### 场景4：自动化脚本集成
```bash
#!/bin/bash
# 自动化数据处理脚本

set -e  # 错误时退出

# 转换数据集
dataflow-cv convert yolo2coco images/ labels/ classes.txt coco.json

# 检查转换结果
if [ $? -eq 0 ]; then
    echo "转换成功，开始可视化检查"
    dataflow-cv visualize coco images/ coco.json --save check/
else
    echo "转换失败，退出码: $?"
    exit 1
fi

# 批量处理多个数据集
for dataset in datasets/*; do
    echo "处理数据集: $dataset"
    dataflow-cv convert yolo2coco "$dataset/images" "$dataset/labels" \
        "$dataset/classes.txt" "$dataset/coco.json" --verbose
done
```

## 5. 开发步骤

### 阶段1：设计CLI整体架构
1. **选择命令行框架**：
   - 评估 `click`、`argparse`、`typer` 等选项
   - 选择 `click`：功能丰富，装饰器语法清晰，社区支持好

2. **设计命令结构**：
   ```python
   dataflow-cv
   ├── --version / -v      # 版本信息
   ├── --help / -h        # 帮助信息
   ├── convert            # 转换命令组
   │   ├── yolo2coco      # YOLO→COCO
   │   ├── yolo2labelme   # YOLO→LabelMe
   │   ├── coco2yolo      # COCO→YOLO
   │   ├── coco2labelme   # COCO→LabelMe
   │   ├── labelme2yolo   # LabelMe→YOLO
   │   └── labelme2coco   # LabelMe→COCO
   └── visualize          # 可视化命令组
       ├── yolo           # YOLO可视化
       ├── coco           # COCO可视化
       └── labelme        # LabelMe可视化
   ```

3. **设计上下文管理**：
   - Click 上下文对象传递配置
   - 共享：verbose 标志、日志目录、strict 模式、logger 实例

### 阶段2：实现主命令入口
1. **实现版本信息回调**：
   ```python
   def print_version(ctx, param, value):
       """Callback function: display version information"""
       if not value or ctx.resilient_parsing:
           return
       from dataflow import __version__
       click.echo(f"dataflow-cv version {__version__}")
       ctx.exit()
   ```

2. **实现主命令装饰器**：
   ```python
   @click.group(context_settings={
       "help_option_names": ["-h", "--help"],
       "max_content_width": 100,
       "show_default": True,
   })
   @click.option(
       "--version",
       "-v",
       is_flag=True,
       is_eager=True,
       expose_value=False,
       callback=print_version,
       help="Display version information",
   )
   @click.pass_context
   def cli(ctx):
       """DataFlow-CV command line tool - Computer vision dataset processing toolkit"""
       # 初始化上下文
       ctx.ensure_object(dict)
       ctx.obj["verbose"] = False
       ctx.obj["log_dir"] = Path("./logs")
       ctx.obj["strict"] = True
       # 配置默认日志
       logger = LoggingOperations().get_logger("dataflow.cli")
       ctx.obj["logger"] = logger
   ```

3. **注册子命令组**：
   ```python
   from .commands import visualize, convert
   cli.add_command(visualize.visualize_group)
   cli.add_command(convert.convert_group)
   ```

### 阶段3：实现convert命令组
1. **设计转换命令组结构**：
   ```python
   @click.group(name="convert")
   def convert_group():
       """Format conversion command group - supports conversion between multiple label formats"""
       pass
   ```

2. **实现单个转换命令**（以 `yolo2coco` 为例）：
   ```python
   @convert_group.command(cls=FormattedCommand)
   @add_common_options
   @click.argument("image_dir", type=click.Path(exists=True, path_type=Path))
   @click.argument("label_dir", type=click.Path(exists=True, path_type=Path))
   @click.argument("class_file", type=click.Path(exists=True, path_type=Path))
   @click.argument("output_file", type=click.Path(path_type=Path))
   @click.option("--do-rle", is_flag=True, help="Use RLE encoding for COCO format")
   def yolo2coco(ctx, image_dir, label_dir, class_file, output_file, do_rle):
       """Convert YOLO format to COCO format"""
       # 1. 参数验证
       validate_convert_params("yolo", "coco", label_dir, output_file, image_dir, class_file)

       # 2. 调用转换器 API
       converter = YoloAndCocoConverter(source_to_target=True, verbose=verbose, strict_mode=strict)
       result = converter.convert(...)

       # 3. 处理结果
       if result.success:
           logger.info(f"Conversion completed: {result.get_summary()}")
       else:
           raise RuntimeCLIError(f"Conversion failed: {result.errors[0] if result.errors else 'Unknown error'}")
   ```

3. **实现其他5个转换命令**：
   - `yolo2labelme`：使用 `LabelMeAndYoloConverter`
   - `coco2yolo`：使用 `YoloAndCocoConverter`
   - `coco2labelme`：使用 `CocoAndLabelMeConverter`
   - `labelme2yolo`：使用 `LabelMeAndYoloConverter`
   - `labelme2coco`：使用 `CocoAndLabelMeConverter`

### 阶段4：实现visualize命令组
1. **设计可视化命令组结构**：
   ```python
   @click.group(name="visualize")
   def visualize_group():
       """Visualization command group - supports visualization of multiple label formats"""
       pass
   ```

2. **实现可视化选项装饰器**：
   ```python
   def add_visualize_options(func):
       @click.option("--verbose", is_flag=True, help="Enable verbose log output")
       @click.pass_context
       @wraps(func)
       def wrapper(ctx, verbose, *args, **kwargs):
           # 更新上下文中的 verbose 标志
           ctx.obj["verbose"] = verbose
           # 重新配置日志
           if verbose:
               logger = VerboseLoggingOperations().get_verbose_logger(...)
           else:
               logger = LoggingOperations().get_logger(...)
           ctx.obj["logger"] = logger
           return func(ctx, *args, **kwargs)
       return wrapper
   ```

3. **实现单个可视化命令**（以 `yolo` 为例）：
   ```python
   @visualize_group.command()
   @add_visualize_options
   @click.argument("image_dir", type=click.Path(exists=True, path_type=Path))
   @click.argument("label_dir", type=click.Path(exists=True, path_type=Path))
   @click.argument("class_file", type=click.Path(exists=True, path_type=Path))
   @click.option("--save", "-s", type=click.Path(path_type=Path), help="Directory to save results")
   def yolo(ctx, image_dir, label_dir, class_file, save):
       """Visualize YOLO format labels"""
       # 参数验证
       validate_visualize_params(label_dir, image_dir, save)

       # 调用可视化器 API
       visualizer = YOLOVisualizer(
           label_dir=label_dir,
           image_dir=image_dir,
           class_file=class_file,
           output_dir=save,
           is_show=True,
           is_save=save is not None,
           verbose=verbose,
           logger=logger
       )
       result = visualizer.visualize()

       # 处理结果
       if not result.success:
           raise RuntimeCLIError(f"Visualization failed: {result.message}")
   ```

4. **实现其他2个可视化命令**：
   - `coco`：使用 `COCOVisualizer`
   - `labelme`：使用 `LabelMeVisualizer`

### 阶段5：实现参数验证和错误处理
1. **设计异常类体系**：
   ```python
   class CLIError(click.ClickException):
       def __init__(self, message, exit_code=1):
           super().__init__(message)
           self.exit_code = exit_code

   class ParameterError(CLIError): ...  # 退出码 1
   class InputError(CLIError): ...      # 退出码 2
   class OutputError(CLIError): ...     # 退出码 3
   class RuntimeCLIError(CLIError): ... # 退出码 4
   class SystemError(CLIError): ...     # 退出码 5
   ```

2. **实现参数验证函数**：
   ```python
   def validate_convert_params(source_format, target_format, input_path, output_path, image_dir, class_file):
       # 检查输入路径存在
       validate_path_exists(input_path, "input path")

       # 确保输出目录存在
       if output_path.suffix:  # 是文件
           output_path.parent.mkdir(parents=True, exist_ok=True)
       else:  # 是目录
           output_path.mkdir(parents=True, exist_ok=True)

       # 检查格式特定的必需参数
       if source_format == "yolo" and target_format == "coco":
           if not image_dir:
               raise InputError("--image-dir is required for YOLO→COCO conversion")
           if not class_file:
               raise InputError("--class-file is required for YOLO→COCO conversion")
       # ... 其他格式组合检查
   ```

3. **实现路径验证函数**：
   ```python
   def validate_path_exists(path, name="path"):
       if not path.exists():
           raise InputError(f"{name} does not exist: {path}")
       return path
   ```

### 阶段6：集成日志系统
1. **实现公共选项装饰器**：
   ```python
   def add_common_options(func):
       @click.option("--verbose", is_flag=True, help="Enable verbose log output")
       @click.pass_context
       @wraps(func)
       def wrapper(ctx, verbose, *args, **kwargs):
           ctx.obj["verbose"] = verbose
           # 重新配置日志
           if verbose:
               logger = VerboseLoggingOperations().get_verbose_logger(
                   name=ctx.command.name,
                   verbose=True,
                   log_dir=Path("./logs"),
               )
           else:
               logger = LoggingOperations().get_logger(ctx.command.name)
           ctx.obj["logger"] = logger
           return func(ctx, *args, **kwargs)
       return wrapper
   ```

2. **集成到转换命令**：
   - 所有转换命令使用 `@add_common_options` 装饰器
   - 从上下文获取 logger：`logger = ctx.obj["logger"]`
   - 传递 verbose 标志给底层 API

3. **集成到可视化命令**：
   - 可视化命令使用 `@add_visualize_options` 装饰器
   - 处理逻辑类似，但选项更简单

### 阶段7：编写帮助文档和示例
1. **实现格式化命令类**：
   ```python
   class FormattedCommand(click.Command):
       """自定义Command类，提供格式化的Arguments显示"""

       def format_help(self, ctx, formatter):
           # 重写帮助输出格式
           self.format_usage(ctx, formatter)
           if self.help:
               formatter.write_paragraph()
               with formatter.indentation():
                   formatter.write_text(self.help)
           self._format_arguments(ctx, formatter)  # 自定义Arguments格式
           self.format_options(ctx, formatter)
   ```

2. **添加参数帮助文本**：
   ```python
   def _get_argument_help(self, param_name):
       help_map = {
           "image_dir": "Image file directory (for obtaining image dimensions)",
           "label_dir": "YOLO label directory",
           "class_file": "Class file path",
           "output_file": "Output COCO JSON file path",
           # ... 其他参数
       }
       return help_map.get(param_name, "")
   ```

3. **编写使用示例**：
   - 在 `__init__.py` 中添加模块级文档字符串
   - 在命令的 `help` 参数中提供清晰描述
   - 创建 `samples/cli/` 目录存放完整示例

### 注意事项
1. **错误处理一致性**：所有命令使用统一的异常处理模式
2. **日志配置**：正确处理 verbose 标志，确保日志文件生成
3. **路径处理**：使用 `pathlib.Path` 确保跨平台兼容性
4. **API 集成**：保持 CLI 作为薄封装层，避免业务逻辑重复
5. **用户体验**：清晰的错误信息，有意义的退出码

## 6. 开发要点

### 扩展指南
**添加新转换方向**：
1. 在 `convert.py` 中添加新命令函数
2. 使用 `@convert_group.command(cls=FormattedCommand)` 装饰器
3. 实现参数验证和 API 调用
4. 更新 `validate_convert_params` 函数支持新格式

**添加新可视化格式**：
1. 在 `visualize.py` 中添加新命令函数
2. 使用 `@visualize_group.command()` 装饰器
3. 使用 `@add_visualize_options` 添加选项
4. 实现参数验证和可视化器调用

**自定义参数验证**：
```python
def custom_validation(ctx, param, value):
    """自定义参数验证函数"""
    if not value.endswith('.json'):
        raise click.BadParameter('Output file must have .json extension')
    return value

@click.argument("output", callback=custom_validation)
```

### 调试技巧
1. **启用详细日志**：使用 `--verbose` 标志查看详细处理过程
2. **检查上下文对象**：在命令中打印 `ctx.obj` 查看配置状态
3. **测试错误路径**：故意提供错误参数测试异常处理
4. **查看退出码**：使用 `echo $?`（Linux/Mac）检查命令退出码

### 性能优化
1. **懒加载模块**：在命令函数内导入大模块，减少启动时间
2. **缓存配置**：重复使用的配置（如类别映射）进行缓存
3. **批量处理优化**：底层 API 已优化，CLI 主要关注参数处理效率

## 7. 测试指南

### 测试策略
**单元测试目标**：
- 参数验证函数的正确性
- 异常类的退出码设置
- 装饰器的功能完整性
- 帮助系统的格式正确性

**集成测试目标**：
- 完整命令执行流程
- 与底层 API 的集成
- 错误处理和工作流程
- 跨平台兼容性

**端到端测试目标**：
- 真实数据集的处理
- 完整工作流的执行
- 用户体验测试

### 测试数据准备
1. **小型测试数据集**：
   ```
   test_cli/
   ├── images/ (测试图片，3-5张)
   ├── yolo_labels/ (YOLO标签)
   ├── labelme_json/ (LabelMe JSON)
   ├── coco.json (COCO标注)
   └── classes.txt (类别文件)
   ```

2. **边界情况数据**：
   - 空目录/文件
   - 无效路径
   - 格式错误数据
   - 权限问题目录

### 示例测试用例

```python
# 测试转换命令参数验证
def test_convert_param_validation():
    # 测试必需参数缺失
    runner = CliRunner()
    result = runner.invoke(cli, ["convert", "yolo2coco", "images/", "labels/"])
    assert result.exit_code == 2  # Click 参数错误退出码
    assert "Missing argument" in result.output

# 测试路径验证函数
def test_validate_path_exists():
    with tempfile.TemporaryDirectory() as tmpdir:
        valid_path = Path(tmpdir) / "test.txt"
        valid_path.write_text("test")

        # 有效路径应正常返回
        result = validate_path_exists(valid_path)
        assert result == valid_path

        # 无效路径应抛出异常
        invalid_path = Path(tmpdir) / "nonexistent.txt"
        with pytest.raises(InputError):
            validate_path_exists(invalid_path)

# 测试完整命令执行
def test_yolo2coco_command():
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmpdir:
        # 准备测试数据
        images_dir = Path(tmpdir) / "images"
        labels_dir = Path(tmpdir) / "labels"
        classes_file = Path(tmpdir) / "classes.txt"
        output_file = Path(tmpdir) / "coco.json"

        # 创建必要文件和目录
        images_dir.mkdir()
        labels_dir.mkdir()
        classes_file.write_text("person\ncar")

        # 运行命令
        result = runner.invoke(cli, [
            "convert", "yolo2coco",
            str(images_dir), str(labels_dir),
            str(classes_file), str(output_file)
        ])

        assert result.exit_code == 0
        assert output_file.exists()

# 测试错误处理
def test_error_handling():
    runner = CliRunner()
    # 测试不存在的路径
    result = runner.invoke(cli, [
        "convert", "yolo2coco",
        "nonexistent/images", "nonexistent/labels",
        "nonexistent/classes.txt", "output.json"
    ])
    assert result.exit_code == 2  # InputError 退出码
    assert "does not exist" in result.output

# 测试帮助系统
def test_help_output():
    runner = CliRunner()

    # 测试主命令帮助
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "DataFlow-CV command line tool" in result.output

    # 测试子命令组帮助
    result = runner.invoke(cli, ["convert", "--help"])
    assert result.exit_code == 0
    assert "yolo2coco" in result.output

    # 测试具体命令帮助
    result = runner.invoke(cli, ["convert", "yolo2coco", "--help"])
    assert result.exit_code == 0
    assert "Convert YOLO format to COCO format" in result.output
```

### 测试覆盖率目标
- 参数验证函数：100% 行覆盖
- 异常类：所有异常类型测试
- 命令函数：主要执行路径测试
- 错误处理：所有错误路径测试
- 集成测试：关键工作流测试

---

**文档版本**：1.0
**最后更新**：2026-03-29
**基于实现版本**：DataFlow-CV 当前实现