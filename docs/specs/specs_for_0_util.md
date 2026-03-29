# DataFlow-CV 工具模块开发文档 (util)

## 1. 模块概述

### 功能定位
`dataflow/util` 模块是 DataFlow-CV 项目的基础工具层，为其他模块提供统一的文件操作和日志记录功能。作为项目的基础设施，它封装了跨平台的文件处理、标准化的日志配置以及详细的调试日志支持。

### 核心特性
- **文件操作工具** (`FileOperations`)：提供跨平台的目录创建、文件复制、删除、查找等操作
- **标准日志配置** (`LoggingOperations`)：统一的日志格式、级别控制和处理器配置
- **详细日志扩展** (`VerboseLoggingOperations`)：支持详细模式、文件轮转和进度日志
- **错误处理集成**：所有操作都包含适当的异常处理和错误日志记录
- **性能优化**：支持 logger 缓存、批量操作和惰性初始化

### 设计原则
1. **一致性**：统一的 API 设计和错误处理模式
2. **可配置性**：支持参数化配置和自定义扩展
3. **跨平台兼容**：使用 `pathlib` 处理路径，确保跨平台兼容性
4. **松耦合**：工具类之间相互独立，可单独使用

## 2. 核心架构

### 类图/组件图
```
┌─────────────────────────────────────────┐
│            dataflow.util                │
├─────────────────────────────────────────┤
│  • FileOperations                       │
│    - ensure_dir()                       │
│    - safe_remove()                      │
│    - copy_files()                       │
│    - find_files()                       │
│                                         │
│  • LoggingOperations (基类)             │
│    - get_logger()                       │
│    - setup_root_logger()                │
│    - create_log_file()                  │
│                                         │
│  • VerboseLoggingOperations (派生类)    │
│    - get_verbose_logger()               │
│    - create_progress_logger()           │
│    - log_summary()                      │
└─────────────────────────────────────────┘
```

### 数据流
1. **文件操作流程**：
   ```
   用户调用 → 参数验证 → 路径处理 → 执行操作 → 结果返回
           ↳ 错误处理 ↳ 日志记录
   ```

2. **日志配置流程**：
   ```
   获取 logger → 检查缓存 → 配置处理器 → 返回 logger
         ↳ 文件输出(如启用) ↳ 控制台输出
   ```

### 依赖关系
- **内部依赖**：无（基础模块）
- **外部依赖**：Python 标准库 (`pathlib`, `logging`, `shutil`, `tempfile` 等)
- **被依赖**：`dataflow/label`, `dataflow/visualize`, `dataflow/convert`, `dataflow/cli` 等所有上层模块

## 3. 主要API

### FileOperations 类
**职责**：提供安全的文件系统操作

**核心方法**：
- `ensure_dir(dir_path: Path) -> bool`：确保目录存在，不存在则创建
- `safe_remove(file_path: Path) -> bool`：安全删除文件或目录（如果存在）
- `copy_files(src_pattern: str, dst_dir: Path, overwrite: bool = False) -> List[Path]`：批量复制匹配模式的文件
- `find_files(directory: Path, pattern: str = "*", recursive: bool = True) -> List[Path]`：查找匹配模式的文件
- `create_temp_dir(prefix: str = "dataflow_") -> Path`：创建临时目录
- `get_file_size(file_path: Path) -> int`：获取文件大小（字节）
- `read_lines(file_path: Path, encoding: str = "utf-8") -> List[str]`：读取文件所有行
- `write_lines(file_path: Path, lines: List[str], encoding: str = "utf-8") -> bool`：写入多行到文件

**配置参数**：
- `logger`：可选的自定义 logger 实例
- 所有方法都支持 `pathlib.Path` 对象

### LoggingOperations 类
**职责**：提供标准日志配置

**核心方法**：
- `get_logger(name: str = "dataflow", level: int = logging.INFO, log_file: Optional[Path] = None, console: bool = True) -> logging.Logger`：获取配置好的 logger
- `setup_root_logger(level: int = logging.INFO, log_file: Optional[Path] = None) -> None`：配置根 logger
- `create_log_file(base_name: str = "dataflow", directory: Path = Path("./logs")) -> Path`：创建带时间戳的日志文件路径
- `set_log_level(logger_name: str, level: int) -> None`：动态设置日志级别

**日志格式**：`%(asctime)s - %(name)s - %(levelname)s - %(message)s`

### VerboseLoggingOperations 类
**职责**：提供详细日志配置（继承自 `LoggingOperations`）

**核心方法**：
- `get_verbose_logger(name: str = "dataflow", verbose: bool = False, log_dir: Path = Path("./logs")) -> logging.Logger`：获取详细模式 logger
- `create_progress_logger(name: str = "dataflow.progress") -> logging.Logger`：创建进度报告专用 logger
- `log_summary(logger: logging.Logger, title: str, data: Dict[str, Any]) -> None`：记录格式化的摘要信息

**详细日志特性**：
- 文件日志包含文件名和行号：`%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s`
- 使用 `RotatingFileHandler`（10MB限制，保留5个备份）
- 进度 logger 使用简单格式：`%(message)s`

## 4. 使用指南

### 快速开始

```python
# 基础文件操作
from dataflow.util import FileOperations
from pathlib import Path

file_ops = FileOperations()
file_ops.ensure_dir(Path("data/output"))
file_ops.copy_files("data/*.json", Path("backup/"))

# 标准日志配置
from dataflow.util import LoggingOperations

log_ops = LoggingOperations()
logger = log_ops.get_logger("my_app", log_file="app.log")
logger.info("Application started")

# 详细日志配置
from dataflow.util import VerboseLoggingOperations

verbose_log_ops = VerboseLoggingOperations()
verbose_logger = verbose_log_ops.get_verbose_logger("my_app", verbose=True)
verbose_logger.debug("Detailed debug information")
```

### 常见场景

#### 场景1：批量处理文件
```python
file_ops = FileOperations()
# 查找所有 JSON 文件
json_files = file_ops.find_files(Path("data"), "*.json")
# 复制到备份目录
copied = file_ops.copy_files("data/*.json", Path("backup"))
# 处理完成后清理临时文件
file_ops.safe_remove(Path("temp/data.tmp"))
```

#### 场景2：配置项目日志
```python
# 项目启动时配置根 logger
LoggingOperations().setup_root_logger(
    level=logging.INFO,
    log_file=Path("logs/project.log")
)

# 各个模块使用自己的 logger
module_logger = LoggingOperations().get_logger("dataflow.convert")
module_logger.info("Starting conversion...")

# 详细调试时启用详细日志
debug_logger = VerboseLoggingOperations().get_verbose_logger(
    "dataflow.debug",
    verbose=True,
    log_dir=Path("logs/debug")
)
```

#### 场景3：进度报告
```python
progress_logger = VerboseLoggingOperations().create_progress_logger()
for i in range(100):
    # 处理任务...
    if i % 10 == 0:
        progress_logger.info(f"Progress: {i}%")
```

## 5. 开发步骤

### 阶段1：设计基础类接口
1. **分析需求**：确定文件操作和日志记录的核心功能需求
2. **设计 `FileOperations` 类接口**：
   - 定义文件操作的核心方法签名
   - 设计错误处理策略（返回布尔值 vs 异常）
   - 确定 `pathlib.Path` 作为标准路径类型
3. **设计 `LoggingOperations` 类接口**：
   - 定义标准日志配置方法
   - 设计 logger 缓存机制
   - 确定默认日志格式和级别

### 阶段2：实现文件操作工具
1. **实现 `FileOperations` 核心方法**：
   ```python
   def ensure_dir(self, dir_path: Path) -> bool:
       """确保目录存在，不存在则创建"""
       try:
           dir_path.mkdir(parents=True, exist_ok=True)
           self._log_success(f"Directory ensured: {dir_path}")
           return True
       except Exception as e:
           self._log_error(f"Failed to ensure directory {dir_path}: {e}")
           return False
   ```
2. **实现批量操作方法**：
   - `copy_files()`：支持通配符模式匹配
   - `find_files()`：支持递归查找
3. **实现临时文件管理**：
   - `create_temp_dir()`：使用 `tempfile` 模块
   - 自动清理机制设计

### 阶段3：实现标准日志配置
1. **实现 logger 缓存机制**：
   ```python
   _logger_cache: Dict[str, logging.Logger] = {}

   def get_logger(self, name: str, ...) -> logging.Logger:
       if name in self._logger_cache:
           return self._logger_cache[name]
       # 创建和配置新 logger
       logger = logging.getLogger(name)
       # ... 配置处理器
       self._logger_cache[name] = logger
       return logger
   ```
2. **配置控制台和文件处理器**：
   - 控制台使用 `StreamHandler`
   - 文件使用 `FileHandler`（可配置）
3. **实现动态日志级别调整**：
   - `set_log_level()` 方法实现

### 阶段4：实现详细日志扩展
1. **继承设计**：创建 `VerboseLoggingOperations` 继承 `LoggingOperations`
2. **实现详细日志格式**：
   - 添加文件名和行号信息
   - 配置更详细的格式字符串
3. **实现文件轮转机制**：
   ```python
   file_handler = RotatingFileHandler(
       log_file_path,
       maxBytes=10 * 1024 * 1024,  # 10MB
       backupCount=5
   )
   ```
4. **实现进度 logger**：
   - 简化格式，只输出消息内容
   - 独立配置，避免干扰主日志

### 阶段5：集成错误处理和缓存机制
1. **完善错误处理**：
   - 所有方法都包含 try-catch 块
   - 适当的错误日志记录
   - 合理的返回值设计（布尔值或结果列表）
2. **优化缓存性能**：
   - 检查缓存有效性
   - 支持缓存清理
3. **编写单元测试**：
   - 测试文件操作的各种边界情况
   - 测试日志配置和级别调整
   - 测试错误处理逻辑

### 注意事项
1. **路径处理**：始终使用 `pathlib.Path` 确保跨平台兼容性
2. **资源管理**：文件操作后及时释放资源，临时目录自动清理
3. **日志性能**：避免过多的日志调用影响性能，特别是在循环中
4. **线程安全**：如果项目需要多线程支持，考虑 logger 的线程安全性

## 6. 开发要点

### 扩展指南
**添加新的文件操作功能**：
1. 在 `FileOperations` 类中添加新方法
2. 遵循现有的错误处理和日志记录模式
3. 添加对应的单元测试

**自定义日志格式**：
```python
class CustomLoggingOperations(LoggingOperations):
    def get_logger(self, name: str, ...):
        # 自定义格式
        formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s in %(module)s: %(message)s'
        )
        # 应用自定义配置...
```

**集成第三方日志服务**：
1. 创建新的处理器类继承 `logging.Handler`
2. 在 `get_logger()` 方法中添加该处理器
3. 配置适当的格式化器和过滤器

### 调试技巧
1. **启用详细日志**：使用 `VerboseLoggingOperations().get_verbose_logger(verbose=True)`
2. **检查文件权限**：文件操作失败时检查目录权限和路径有效性
3. **日志级别调试**：动态调整日志级别查找问题
   ```python
   LoggingOperations().set_log_level("dataflow", logging.DEBUG)
   ```

### 性能优化
1. **批量操作**：使用 `copy_files()` 批量处理而非单个文件循环
2. **缓存利用**：重复获取同一 logger 时利用缓存
3. **惰性初始化**：处理器在首次使用时才创建
4. **日志级别控制**：生产环境使用 `INFO` 或 `WARNING` 级别减少日志量

## 7. 测试指南

### 测试策略
**单元测试目标**：
- 文件操作的各种场景（成功、失败、边界条件）
- 日志配置的正确性（格式、级别、处理器）
- 缓存机制的有效性
- 错误处理的完整性

**集成测试目标**：
- 与其他模块的集成使用
- 实际工作流中的表现

### 测试数据准备
1. **临时目录结构**：
   ```
   test_data/
   ├── input/
   │   ├── file1.json
   │   └── file2.json
   ├── output/ (空目录)
   └── temp/ (测试用临时目录)
   ```
2. **日志测试文件**：指定测试专用的日志文件路径

### 示例测试用例

```python
# 测试文件操作
def test_ensure_dir_creates_missing_directory():
    file_ops = FileOperations()
    test_dir = Path("test_temp/new_dir")

    # 确保目录不存在
    if test_dir.exists():
        shutil.rmtree(test_dir)

    # 测试创建目录
    result = file_ops.ensure_dir(test_dir)
    assert result is True
    assert test_dir.exists()

    # 清理
    shutil.rmtree(test_dir)

# 测试日志配置
def test_get_logger_creates_correct_logger():
    log_ops = LoggingOperations()
    logger = log_ops.get_logger("test_logger")

    assert logger.name == "test_logger"
    assert logger.level == logging.INFO
    assert len(logger.handlers) > 0

# 测试详细日志
def test_verbose_logger_creates_file():
    verbose_log_ops = VerboseLoggingOperations()
    logger = verbose_log_ops.get_verbose_logger(
        "test_verbose",
        verbose=True,
        log_dir=Path("test_logs")
    )

    # 检查文件处理器是否存在
    has_file_handler = any(
        isinstance(h, RotatingFileHandler)
        for h in logger.handlers
    )
    assert has_file_handler is True

    # 测试日志记录
    logger.debug("Test debug message")
```

### 测试覆盖率目标
- 文件操作方法：100% 行覆盖
- 日志配置方法：100% 行覆盖
- 错误处理分支：90% 以上覆盖
- 边界条件测试：所有边界情况都有测试用例

---

**文档版本**：1.0
**最后更新**：2026-03-29
**基于实现版本**：DataFlow-CV 当前实现