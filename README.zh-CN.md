# DataFlow-CV

> 🌊 **模型以外的事,我们来做。** 分析、转换、可视化、评估 —— 一个 CLI 搞定所有 CV 数据处理。

<p align="center">
  <a href="https://pypi.org/project/dataflow-cv/"><img src="https://img.shields.io/pypi/v/dataflow-cv.svg" alt="PyPI"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.8+-blue.svg" alt="Python 3.8+"></a>
  <a href="https://github.com/zjykzj/DataFlow-CV/actions/workflows/python-publish.yml"><img src="https://github.com/zjykzj/DataFlow-CV/actions/workflows/python-publish.yml/badge.svg" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="License"></a>
  <a href="https://deepwiki.com/zjykzj/DataFlow-CV"><img src="https://deepwiki.com/badge.svg" alt="Ask DeepWiki"></a>
  <br>
  <img src="https://img.shields.io/badge/Linux-Supported-fcc624?logo=linux" alt="Linux">
  <img src="https://img.shields.io/badge/Windows-Supported-00a2e8?logo=windows" alt="Windows">
  <img src="https://img.shields.io/badge/macOS-Supported-999999?logo=apple" alt="macOS">
  <img src="https://img.shields.io/badge/YOLO-.txt-00a86b?style=flat-square" alt="YOLO">
  <img src="https://img.shields.io/badge/LabelMe-.json-f39c12?style=flat-square" alt="LabelMe">
  <img src="https://img.shields.io/badge/COCO-.json-e74c3c?style=flat-square" alt="COCO">
</p>

DataFlow-CV 是一个计算机视觉数据集处理库 —— 支持在 YOLO、LabelMe 和 COCO 三种格式之间进行标注数据的分析、转换、可视化和评估。

| | | |
|:---|:---|:---|
| 🔍 **分析** | 统计、训练/验证集划分、类别过滤、N 路分区与文件抽样 —— 格式自动检测 | `dataflow-cv analyse stats ...` |
| 🔄 **转换** | 6 个方向:YOLO ↔ LabelMe ↔ COCO,外加模型预测文件 | `dataflow-cv convert yolo2coco ...` |
| 🎨 **可视化** | OpenCV 渲染、按类别着色,支持显示与保存模式 | `dataflow-cv visualize yolo ...` |
| 📊 **评估** | 基于 pycocotools 的 COCO mAP、按类别的单阈值 P/R/F1 | `dataflow-cv evaluate detection ...` |
| 💻 **CLI + API** | 基于 Click 的 CLI,带丰富的 `--help`;Python API 用于构建流水线 | `from dataflow.convert import ...` |

---

## 📦 安装

```bash
pip install dataflow-cv               # 从 PyPI 安装
pip install dataflow-cv[coco]         # 可选:COCO RLE 编码 + 评估
```

或者从源码安装:

```bash
git clone https://github.com/zjykzj/DataFlow-CV.git
cd DataFlow-CV && pip install .
```

---

## 🚀 快速开始

### 命令行界面

所有必需参数(图像目录、标签目录、类别文件、输出路径)均为位置参数,使用更便捷。在任意子命令后使用 `--help` 查看详细用法。

#### 🔍 数据集分析

```bash
# 数据集统计(自动检测 YOLO / LabelMe / COCO)
dataflow-cv analyse stats yolo_labels/ --image-dir images/ --class-file classes.txt
dataflow-cv analyse stats labelme_json/
dataflow-cv analyse stats coco_annotations.json

# 训练/测试集划分(仅 YOLO / LabelMe —— 标签 / 图像 / 两者模式)
dataflow-cv analyse split -l yolo_labels/ outputs/ --ratio 0.8 --seed 42 -c classes.txt
dataflow-cv analyse split -i images/ outputs/ --ratio 0.8
dataflow-cv analyse split -l yolo_labels/ -i images/ outputs/ --ratio 0.8

# 类别过滤(保留类别子集,按新的 classes.txt 重映射 ID)
dataflow-cv analyse filter yolo_labels/ classes.txt classes_new.txt filtered/
dataflow-cv analyse filter coco_annotations.json classes.txt classes_new.txt filtered/

# N 路分区 —— 仅 YOLO / LabelMe(标签主导,图像按文件名主干跟随)
dataflow-cv analyse partition -n 4 --label-dir yolo_labels/ --image-dir images/ parts/
dataflow-cv analyse partition -n 4 --image-dir images/ --shuffle parts/

# 文件抽样 —— 收集 N 个文件(随机或顺序,标签 / 图像 / 两者模式)
dataflow-cv analyse sample -l yolo_labels/ output/ -n 10
dataflow-cv analyse sample -i images/ output/ -n 10 --no-shuffle
dataflow-cv analyse sample -l yolo_labels/ -i images/ output/ -n 5 --seed 42

# 按数量降序排序(默认:类别 ID 升序)
dataflow-cv analyse stats --sort-by count --descending yolo_labels/

# 详细日志
dataflow-cv analyse stats --verbose yolo_labels/ --class-file classes.txt
```

#### 🔄 格式转换

```bash
# YOLO → COCO
dataflow-cv convert yolo2coco images/ yolo_labels/ classes.txt output.json

# YOLO → COCO(RLE 编码)
dataflow-cv convert yolo2coco images/ yolo_labels/ classes.txt output.json --do-rle

# YOLO → LabelMe
dataflow-cv convert yolo2labelme images/ yolo_labels/ classes.txt labelme_json/

# LabelMe → YOLO
dataflow-cv convert labelme2yolo labelme_json/ classes.txt yolo_labels/

# LabelMe → COCO
dataflow-cv convert labelme2coco labelme_json/ classes.txt output.json

# COCO → YOLO
dataflow-cv convert coco2yolo input.json yolo_labels/

# COCO → LabelMe
dataflow-cv convert coco2labelme input.json labelme_json/

# YOLO 预测文件 → COCO(输出:纯 JSON 列表 —— 预测格式)
dataflow-cv convert yolo2coco --prediction images/ yolo_preds/ classes.txt pred.json

# 选项
dataflow-cv convert yolo2coco --verbose images/ labels/ classes.txt output.json
dataflow-cv convert yolo2coco --no-strict images/ labels/ classes.txt output.json
```

#### 🎨 可视化

```bash
# 可视化 YOLO 标注
dataflow-cv visualize yolo images/ yolo_labels/ classes.txt --save visualized/

# 可视化 LabelMe 标注
dataflow-cv visualize labelme images/ labelme_json/ --save visualized/

# 可视化 COCO 标注
dataflow-cv visualize coco images/ coco_annotations.json --save visualized/

# 详细日志 + 无界面模式
dataflow-cv visualize yolo --verbose --no-display images/ yolo_labels/ classes.txt --save visualized/
```

<p align="center">
  <img src="assets/showcase/seg_demo_1.jpg" width="45%" alt="Segmentation visualization demo 1">
  <img src="assets/showcase/seg_demo_2.jpg" width="45%" alt="Segmentation visualization demo 2">
</p>

#### 📊 评估

使用 COCO 标准指标评估目标检测和实例分割模型。需要两个 COCO 格式的 JSON 文件:

| 文件 | 角色 | 格式 | 如何生成 |
|------|------|--------|---------------|
| **`anno.json`** | 真值 (GT) | 完整 COCO 字典(`images`、`annotations`、`categories`) | `yolo2coco`(标注模式) |
| **`pred.json`** | 检测结果 (DT) | 纯 JSON 列表(含 `score`) | `yolo2coco --prediction` |

##### ① 准备数据

```bash
# GT:YOLO 标注 → COCO
dataflow-cv convert yolo2coco images/ yolo_labels/ classes.txt anno.json

# DT:YOLO 预测 → COCO(模型输出需加 --prediction)
dataflow-cv convert yolo2coco --prediction images/ yolo_preds/ classes.txt pred.json
```

> ⚠️ YOLO 预测文件必须使用 `--prediction` —— 它们的每一行多一个 `confidence` token。该选项输出**纯 JSON 列表**(而非完整 COCO 字典),这是 `loadRes()` 的标准 DT 格式。只有 `yolo2coco` 支持 `--prediction`;`labelme2coco` 不需要它(LabelMe 没有标注与预测之分)。

##### ② 运行评估

```bash
# 目标检测(bbox IoU)
dataflow-cv evaluate detection anno.json pred.json
dataflow-cv evaluate detection --verbose anno.json pred.json           # 按类别明细
dataflow-cv evaluate detection --prf1 anno.json pred.json              # 仅 P/R/F1(跳过 mAP)
dataflow-cv evaluate detection --prf1 --prf1-iou 0.75 --prf1-method micro anno.json pred.json

# 实例分割(mask IoU)
dataflow-cv evaluate segmentation anno.json pred.json
dataflow-cv evaluate segmentation --verbose anno.json pred.json

# 结果保存为 JSON
dataflow-cv evaluate detection --output results.json anno.json pred.json

# 自定义日志目录
dataflow-cv evaluate detection --verbose --log-dir logs/eval/ anno.json pred.json
```

##### ③ 检测 vs 分割

两种评估模式,区别在于重叠程度的度量方式:

- **目标检测** —— 边界框 IoU。GT 和 DT 需要 `bbox`;DT 还需要 `score`。
- **实例分割** —— 掩码 IoU。GT 和 DT 需要 `bbox`、`segmentation`(多边形或 RLE)以及 `area`;DT 还需要 `score`。

`yolo2coco`(标注模式)和 `yolo2coco --prediction`(预测模式)会自动填充两种模式所需的全部字段 —— 无需手动编辑。

### 🐍 Python API

```python
from dataflow.util.logging import LogConfig
from dataflow.analyse import StatsAnalyser, SplitAnalyser, FilterAnalyser, PartitionAnalyser, SampleAnalyser
from dataflow.convert import YoloAndCocoConverter
from dataflow.visualize import YOLOVisualizer
from dataflow.evaluate import DetectionEvaluator, compute_pr_f1

# ── 分析 ─────────────────────────────────────────
log_cfg = LogConfig(name="analyse", verbose=True)

# 数据集统计
analyser = StatsAnalyser(log_config=log_cfg)
result = analyser.analyse("yolo_labels/", class_file="classes.txt")
print(f"{result.data.total_files} images, {result.data.total_annotations} objects")

# 训练/测试集划分(YOLO / LabelMe)
splitter = SplitAnalyser(log_config=log_cfg)
result = splitter.analyse(
    output_dir="output/", ratio=0.8, seed=42,
    label_dir="yolo_labels/", class_file="classes.txt",
)
print(f"Train: {result.data.train_count}, Val: {result.data.val_count}")

# 带图像的划分(两者模式 —— 标签主导,图像按文件名主干跟随)
result = splitter.analyse(
    output_dir="output/", ratio=0.8, seed=42,
    label_dir="yolo_labels/", image_dir="images/",
    class_file="classes.txt",
)

# 类别过滤(按新的 classes.txt 保留 / 重映射类别)
filterer = FilterAnalyser(log_config=log_cfg)
result = filterer.analyse(
    "yolo_labels/", original_class_file="classes.txt",
    new_class_file="classes_new.txt", output_dir="filtered/",
)

# N 路分区(YOLO / LabelMe 标签;图像按文件名主干跟随)
partitioner = PartitionAnalyser(log_config=log_cfg)
result = partitioner.analyse(
    output_dir="parts/", num=4,
    label_dir="yolo_labels/", image_dir="images/",
)

# 文件抽样(标签、图像或两者 —— 随机或顺序)
sampler = SampleAnalyser(log_config=log_cfg)
result = sampler.analyse(
    output_dir="sampled/", count=10,
    label_dir="yolo_labels/", shuffle=True, seed=42,
)

# ── 转换 ──────────────────────────────────────────
# YOLO 标注 → COCO(标注模式)
log_cfg = LogConfig(name="convert", verbose=True)
converter = YoloAndCocoConverter(source_to_target=True, log_config=log_cfg, strict_mode=True)
result = converter.convert(
    source_path="yolo_labels/", target_path="anno.json",
    class_file="classes.txt", image_dir="images/",
)

# YOLO 预测 → COCO(预测模式)
converter = YoloAndCocoConverter(source_to_target=True, prediction=True)
result = converter.convert(
    source_path="yolo_preds/", target_path="pred.json",
    class_file="classes.txt", image_dir="images/",
)

# ── 可视化 ────────────────────────────────────────
visualizer = YOLOVisualizer(
    label_dir="yolo_labels/", image_dir="images/",
    class_file="classes.txt", is_show=True, is_save=True,
    output_dir="visualized/", log_config=log_cfg,
)
result = visualizer.visualize()

# ── 评估 ─────────────────────────────────────────
evaluator = DetectionEvaluator(log_config=LogConfig(name="eval", verbose=True))
result = evaluator.evaluate("anno.json", "pred.json")
print(f"AP: {result.metrics.ap:.3f}, AP50: {result.metrics.ap50:.3f}")

# 快速 P/R/F1(IoU=0.5,默认:macro 平均、bbox IoU)
prf1 = compute_pr_f1("anno.json", "pred.json", iou_threshold=0.5)
print(f"Macro F1: {prf1.overall.f1_score:.3f}")

# Micro 平均 P/R/F1(样本等权)
prf1 = compute_pr_f1("anno.json", "pred.json", method="micro")
print(f"Micro F1: {prf1.overall.f1_score:.3f}")

# 分割 P/R/F1(mask IoU)
prf1 = compute_pr_f1("anno_segm.json", "pred_segm.json", iou_type="segm")
print(f"Segm F1: {prf1.overall.f1_score:.3f}")
```

> 📂 完整示例见 `samples/` 目录:`samples/analyse/`(统计与划分)、`samples/convert/`(6 个转换方向)、`samples/visualize/`(YOLO、LabelMe、COCO)、`samples/evaluate/`(检测与分割)、`samples/cli/`(CLI 工作流)。

---

## 📖 文档

| 资源 | 说明 |
|----------|-------------|
| **[CLAUDE.md](CLAUDE.md)** | 架构概述、开发指南与已知坑点 |
| **[CHANGELOG.md](CHANGELOG.md)** | 版本历史与破坏性变更 |
| **[specs/evaluate/](specs/evaluate/)** | 评估指标契约 —— IoU、匹配、AP/mAP/AR |
| **[specs/formats/](specs/formats/)** | 外部格式契约 —— YOLO、LabelMe、COCO、转换规则 |
| **[specs/modules/](specs/modules/)** | 内部模块架构、接口契约、依赖约束 |

### 💡 关键概念

- **格式原生坐标**:YOLO 使用归一化 [0,1] 中心点坐标;LabelMe 和 COCO 使用绝对像素左上角坐标。内部没有隐藏的归一化处理 —— 请查看 `DatasetAnnotations.format` 以理解坐标语义。
- **严格模式**(默认):验证错误立即抛出异常。通过 `--no-strict`(CLI)或 `strict_mode=False`(API)关闭后,将跳过无效标注并继续处理。
- **详细日志**:`--verbose` 通过 `LogManager` 启用各模块的文件日志 —— 控制台显示 INFO 级进度,日志文件记录 DEBUG 级细节。所有日志均由模块产生;CLI 使用 `click.echo()` 输出终端信息。
- **无界面支持**:服务器/Docker 环境使用 `--no-display` —— 搭配 `--save` 即可在无 GUI 窗口的情况下渲染可视化图像。
- **键盘快捷键**(可视化):`←` / `↑` 上一张,`→` / `↓` / `Enter` / `Space` 下一张,`s` 保存快照,`h` 显示帮助,`q` / `ESC` 退出。首尾边界操作为空操作。标题栏显示 `[N/T]` 位置。
- **评估**:`--prf1` 仅计算 P/R/F1(单阈值、按类别 TP/FP/FN)—— 跳过完整的 COCOeval mAP 流程以获得更快的速度。支持 macro/micro 平均以及 bbox/mask IoU。不加 `--prf1` 即计算标准 COCO mAP。如需两种指标,运行两次。
- **预测文件**:YOLO 预测文件检测任务每行 6 个 token、分割任务为偶数个 token,而标注文件为 5 个/奇数个。对 `yolo2coco` 使用 `--prediction` —— 输出纯 JSON 列表格式的标注字典,与 pycocotools `loadRes()` 兼容。

---

## 🔧 开发

详细的开发者指南(包括高级测试命令、调试和架构概述)见 [CLAUDE.md](CLAUDE.md)。常见任务的可选 Claude Code 技能(`/maestro:spec`、`/maestro:commit`、`/maestro:release`、`/maestro:claude`)可通过 [maestro 插件](https://github.com/zjykzj/claude-skills)获取——未安装时项目也能正常开发。

### 🤖 AI 助手技能

处理 dataflow-cv 的可选 Claude Code 技能 —— `/dataflow:dataflow-cv`(CLI 与 Python API 参考、权威示例、已知坑点)—— 可通过 claude-skills 市场的 [dataflow 插件](https://github.com/zjykzj/claude-skills)获取:

```bash
claude plugin install dataflow@claude-skills
```

该技能让 AI 助手能正确操作 dataflow-cv 的 CLI 与 Python API。未安装时项目也能正常开发。

### 🧪 测试

**561 个测试,80% 代码覆盖率(5462 条语句)。**

```bash
pytest                                    # 全部测试
pytest --cov=dataflow --cov-report=term   # 带覆盖率报告
pytest tests/convert/test_yolo_and_coco.py  # 单个模块
pytest tests/evaluate/test_evaluator.py     # 单个模块
```

<details>
<summary><b>📊 各模块覆盖率</b></summary>

| 模块 | 覆盖率 | 亮点 |
|--------|:--------:|------------|
| `dataflow/label/` | 71% | models (84%), base (82%), utils (78%), coco_handler (74%), labelme_handler (71%), yolo_handler (61%) |
| `dataflow/analyse/` | 84% | base (99%), log_templates (92%), sample (87%), utils (85%), split (85%), stats (83%), filter (76%), partition (74%) |
| `dataflow/convert/` | 85% | labelme_and_yolo (93%), yolo_and_coco (89%), utils (87%), coco_and_labelme (86%), base (80%), rle (80%) |
| `dataflow/visualize/` | 81% | yolo_vis (100%), labelme_vis (100%), coco_vis (93%), base (76%) |
| `dataflow/evaluate/` | 87% | evaluator (100%), result (99%), metrics (93%), base (90%), utils (67%) |
| `dataflow/cli/` | 74% | main (96%), visualize cmd (87%), utils (87%), evaluate cmd (83%), analyse cmd (65%), convert cmd (52%) |
| `dataflow/util/` | 100% | logging (100%) |

</details>

### 🎨 代码质量

```bash
pip install -e .[dev]        # 安装开发依赖
black dataflow tests samples  # 代码格式化
isort dataflow tests samples  # 导入排序
mypy dataflow                 # 类型检查
flake8 dataflow tests samples # 代码检查
```

### 🔗 Pre-commit 钩子(可选)

```bash
pip install pre-commit
pre-commit install            # 安装 git 钩子(只需运行一次)

# 之后每次 `git commit` 会自动运行:
#   black → isort → flake8 → 空白字符检查

pre-commit run --all-files    # 手动对全部文件运行
```

### 📁 项目结构

```
dataflow/
├── label/           # 标注处理器 + 数据模型
├── analyse/         # 数据集统计、训练/验证集划分、类别过滤、N 路分区、文件抽样
├── convert/         # 格式转换器、RLE 工具、日志模板
├── visualize/       # 基于 OpenCV 的渲染、日志模板
├── evaluate/        # 基于 pycocotools 的指标计算、日志模板
├── util/            # 统一日志(LogManager + 格式化辅助)
└── cli/             # CLI 入口、命令、校验
tests/               # 单元与集成测试(561 个测试、conftest 夹具)
samples/             # Python API 使用示例(analyse、convert、visualize、evaluate、cli)
assets/              # 测试数据(按格式分类的 det/seg)
specs/               # 权威规范(evaluate/ + formats/ + modules/)
```

---

## 🤝 贡献

欢迎贡献!请在贡献前阅读 [CLAUDE.md](CLAUDE.md),了解架构和开发模式。

1. 🍴 Fork 本仓库
2. 🌿 创建功能分支
3. ✏️ 进行修改 —— **规范优先**:如果修改影响了 [specs/](specs/) 中的契约,请先更新规范再改代码(SDD,见 `/maestro:spec` 技能)
4. 🧪 按需添加或更新测试
5. ✅ 确保代码通过格式化和代码检查
6. 📬 提交 Pull Request

---

## 📄 许可证

本项目采用 MIT 许可证 —— 详见 [LICENSE](LICENSE)。

---

## 🙏 致谢

- 感谢 YOLO、LabelMe 和 COCO 格式的创建者建立了这些标注标准
- 基于 [OpenCV](https://opencv.org/)、[NumPy](https://numpy.org/)、[Click](https://click.palletsprojects.com/) 和 [pycocotools](https://github.com/cocodataset/cocoapi) 构建
- 灵感来源于多工具 CV 流水线中对无缝格式转换的需求
