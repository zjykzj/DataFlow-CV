# Python API 参考

> dataflow-cv v3.6 —— Python API 与 CLI 能力一一对应。示例假设已安装
> (`pip install dataflow-cv`)。

## 概览

| 模块 | 关键类 / 函数 | 用途 |
|------|---------------|------|
| `dataflow.label` | `YoloAnnotationHandler`、`LabelMeAnnotationHandler`、`CocoAnnotationHandler`、`DatasetAnnotations` | 标注的读 / 写 / 校验 |
| `dataflow.analyse` | `StatsAnalyser`、`SplitAnalyser`、`FilterAnalyser`、`PartitionAnalyser`、`SampleAnalyser` | 数据集分析与准备 |
| `dataflow.convert` | `YoloAndCocoConverter`、`LabelMeAndYoloConverter`、`CocoAndLabelMeConverter` | 格式转换 |
| `dataflow.visualize` | `YOLOVisualizer`、`LabelMeVisualizer`、`COCOVisualizer` | OpenCV 渲染 |
| `dataflow.evaluate` | `DetectionEvaluator`、`SegmentationEvaluator`、`KeypointEvaluator`、`compute_pr_f1` | COCO 指标 |
| `dataflow.util.logging` | `LogConfig`、`LogManager` | 统一日志 |

## 通用模式

### 日志

所有模块都接受 `log_config: Optional[LogConfig] = None`。`verbose=True`
会为每次运行写日志文件;`log_dir` 控制位置(默认 `./logs/`)。

```python
from dataflow.util.logging import LogConfig

log_cfg = LogConfig(name="my-pipeline", verbose=True, log_dir="./logs/")
```

### 结果对象

所有操作都返回结果容器;使用 `result.data` 前先检查 `result.success`。

| 结果 | 字段 | 由谁返回 |
|------|------|---------|
| `AnalysisResult` | `success`、`data`、`errors`、`warnings`、`log_path` | `*.analyse()` |
| `ConversionResult` | `success`、`data`、`errors`、`warnings`、`log_path` | `converter.convert()` |
| `EvaluationResult` | `success`、`metrics`(12 项 COCO 指标 + `ar_max_20`)、`per_class`(verbose 模式)、`errors` | `evaluator.evaluate()` |
| `PRF1Result` | `success`、`overall`(`precision`/`recall`/`f1_score`)、`per_class`、`thresholds` | `compute_pr_f1()` |

## Label Handlers

按各格式原生表示读写标注。handlers 以格式原生坐标存储数据 —— 解读坐标前
务必查看 `DatasetAnnotations.format`(见[数据模型](#数据模型))。

```python
from dataflow.label import CocoAnnotationHandler, LabelMeAnnotationHandler, YoloAnnotationHandler

# YOLO(检测 / 分割 / 姿态通过 pose=True 开启)
handler = YoloAnnotationHandler(
    label_dir="yolo_labels/",      # 每图一个 .txt
    class_file="classes.txt",      # 类别名(行号即类别 id)
    image_dir="images/",
    pose=True,                     # YOLO pose 行(5+3k token)必须显式开启
    strict_mode=True,
)
result = handler.read()            # -> DatasetAnnotations(全部数据载入内存)
dataset = result.data
print(dataset.format, len(dataset.images), dataset.num_objects)

# LabelMe(每图一个 .json;kpts_config 提供关键点排序)
handler = LabelMeAnnotationHandler(label_dir="labelme_json/", kpts_config="kpts.yaml")

# COCO(单个 .json;prediction=True 时读/写纯列表预测格式)
handler = CocoAnnotationHandler(annotation_file="anno.json", do_rle=False)
```

| Handler | 构造函数 | 说明 |
|---------|----------|------|
| `YoloAnnotationHandler` | `(label_dir, class_file, image_dir=None, pose=False, strict_mode=True, **kwargs)` | `prediction=True`(经 kwargs)解析预测行(多一个 confidence token) |
| `LabelMeAnnotationHandler` | `(label_dir, class_file=None, kpts_config=None, strict_mode=True, **kwargs)` | 姿态约定 = `point` shapes + 共用 `group_id` 的 `rectangle` |
| `CocoAnnotationHandler` | `(annotation_file, do_rle=False, prediction=False, strict_mode=True, **kwargs)` | `prediction=True` → `write()` 输出纯 JSON 列表(Variant B) |

每个 handler 还提供 `write()`(批量)、`write_one()`(单图;COCO 抛
`NotImplementedError`)、`validate()` 与 `iter_images()`(流式)。

## Analyse

所有 analyser 的构造函数都接受 `log_config`,并以 `strict_mode=False` 读取
标注(只读、宽松 —— 错误累积在 `result.errors` 中)。

```python
from dataflow.analyse import FilterAnalyser, PartitionAnalyser, SampleAnalyser, SplitAnalyser, StatsAnalyser

# 统计(按路径自动检测 YOLO / LabelMe / COCO)
stats = StatsAnalyser(log_config=log_cfg).analyse(
    "yolo_labels/", image_dir="images/", class_file="classes.txt",
    sort_by="count", descending=True, recursive=False,
    kpts_config="kpts.yaml",   # 启用关键点统计(对 YOLO 源同时启用 pose 行解析)
)

# 训练/验证集划分(仅 YOLO / LabelMe —— 标签 / 图像 / 两者模式)
split = SplitAnalyser().analyse(
    output_dir="split/", ratio=0.8, seed=42,
    label_dir="yolo_labels/", image_dir="images/", class_file="classes.txt",
    move=False,                # False 复制 / True 移动
)

# 类别过滤(保留子集;按新的 classes.txt 重映射 ID)
filtered = FilterAnalyser().analyse(
    "yolo_labels/", original_class_file="classes.txt",
    new_class_file="classes_new.txt", output_dir="filtered/",
)

# N 路分区(仅 YOLO / LabelMe)
partition = PartitionAnalyser().analyse(
    output_dir="parts/", num=4,
    label_dir="yolo_labels/", image_dir="images/", shuffle=True, seed=42,
)

# 文件抽样 —— 收集 N 个文件(标签 / 图像 / 两者;随机或顺序)
sample = SampleAnalyser().analyse(
    output_dir="sampled/", count=10,
    label_dir="yolo_labels/", shuffle=True, seed=42, move=False,
)
```

## Convert

`convert()` 自动分发:目标为 COCO → 批处理管线,目标为 YOLO/LabelMe →
流式管线。`source_to_target=True` 表示按类名顺序"源名 → 目标名"
(见下表)。

```python
from dataflow.convert import CocoAndLabelMeConverter, LabelMeAndYoloConverter, YoloAndCocoConverter

result = YoloAndCocoConverter(source_to_target=True).convert(
    source_path="yolo_labels/", target_path="anno.json",
    class_file="classes.txt", image_dir="images/",
    do_rle=False,               # YOLO→COCO:RLE 分割输出
    pose=True,                  # 姿态转换(六个方向)
    kpts_config="kpts.yaml",    # 目标侧关键点名称/骨架
)
```

| Converter | `source_to_target=True` | `source_to_target=False` | 额外参数 |
|-----------|-------------------------|--------------------------|---------|
| `YoloAndCocoConverter` | YOLO → COCO | COCO → YOLO | `prediction=True` 用于模型输出(仅 `yolo2coco --prediction`);`do_rle`;`save_kpts_config`(COCO→YOLO) |
| `LabelMeAndYoloConverter` | LabelMe → YOLO | YOLO → LabelMe | `kpts_config` |
| `CocoAndLabelMeConverter` | LabelMe → COCO | COCO → LabelMe | `save_crowd`;`save_kpts_config`(COCO→LabelMe) |

所有 converter 在适用方向上均接受 `pose`、`kpts_config`、
`save_kpts_config`。COCO→YOLO / COCO→LabelMe 默认跳过 `iscrowd=1`
标注(`save_crowd=True` 保留)。

## Visualize

```python
from dataflow.visualize import COCOVisualizer, LabelMeVisualizer, YOLOVisualizer

visualizer = YOLOVisualizer(
    label_dir="yolo_labels/",
    image_dir="images/",
    class_file="classes.txt",
    pose=True,                  # 渲染关键点 + 骨架(YOLO)
    kpts_config="kpts.yaml",    # 骨架来源(YOLO / LabelMe)
    is_show=True,               # 交互窗口(键盘导航)
    is_save=True,               # 同时保存每一帧
    output_dir="visualized/",
    log_config=log_cfg,
)
result = visualizer.visualize()
```

| Visualizer | 构造函数 | 说明 |
|-----------|----------|------|
| `YOLOVisualizer` | `(label_dir, image_dir, class_file, pose=False, kpts_config=None, **kwargs)` | 姿态渲染需要 `pose=True` |
| `LabelMeVisualizer` | `(label_dir, image_dir, class_file=None, kpts_config=None, **kwargs)` | 存在关键点时自动渲染 |
| `COCOVisualizer` | `(annotation_file, image_dir, **kwargs)` | 骨架来自 COCO categories |

`BaseVisualizer` 通用 kwargs:`output_dir`、`is_show=True`、
`is_save=False`、`log_config`。无界面模式:`is_show=False` +
`is_save=True`。

## Evaluate

输入可以是文件路径、内存中的 dict/list 或 `DatasetAnnotations`。GT 必须是
完整 COCO 字典;DT 可以是完整字典或纯 JSON 列表(Variant B,需含
`score`)。

```python
from dataflow.evaluate import DetectionEvaluator, KeypointEvaluator, SegmentationEvaluator, compute_pr_f1

evaluator = DetectionEvaluator(log_config=LogConfig(name="eval", verbose=True))
result = evaluator.evaluate("anno.json", "pred.json")
print(f"AP: {result.metrics.ap:.3f}, AP50: {result.metrics.ap50:.3f}")
# SegmentationEvaluator:接口相同,mask IoU(iouType='segm')
# KeypointEvaluator(sigmas=None):OKS;sigmas 依次取显式列表、
#   COCO 17 关键点官方值、均匀 0.05 + WARNING

# 快速单阈值 P/R/F1(跳过 COCOeval mAP 管线)
prf1 = compute_pr_f1(
    "anno.json", "pred.json",
    iou_threshold=0.5,          # IoU 阈值(姿态任务为 OKS 阈值)
    confidence_threshold=0.0,
    iou_type="bbox",            # "bbox" | "segm" | "keypoints"
    method="macro",             # "macro"(逐类均值)| "micro"(汇总)
    kpt_sigmas=None,            # iou_type="keypoints" 时的逐关键点 sigmas
)
print(f"Macro F1: {prf1.overall.f1_score:.3f}")
```

> `KeypointEvaluator` 产生 10 项 COCO 统计(无 small 尺度、`maxDets=20`)
> —— `metrics.ar_max_20` 承载 AR;`ap_small`/`ar_small`/`ar_max_1/10/100`
> 保持 `-1.0`。

## 数据模型

```python
from dataflow.label.models import DatasetAnnotations
```

| 模型 | 关键字段 | 说明 |
|------|---------|------|
| `DatasetAnnotations` | `format`、`images`、`categories`、`num_objects`、`dataset_info` | `format` 定义坐标语义:`YOLO` = 中心点 + 归一化 [0,1];`LABELME` / `COCO` = 左上角 + 绝对像素 |
| `ImageAnnotation` | `width`、`height`、`objects`、`image_path`、`image_id` | 每图一个 |
| `ObjectAnnotation` | `class_id`、`class_name`、`bbox`、`segmentation`、`keypoints`、`confidence`、`is_crowd` | bbox / segmentation / keypoints 至少其一 |
| `BoundingBox` | `x`、`y`、`width`、`height` | **格式原生**(见上) |
| `Segmentation` | `points` | 原生坐标多边形 |
| `Keypoints` | `points`、`visibility`、`count_labeled()` | `v ∈ {0,1,2}`:0 = 未标注(存储为 0,0),1 = 遮挡,2 = 可见 |

## 参见

- [samples/](../../samples/) —— 每个模块的可运行示例
- CLI 参考:`dataflow-cv --help` 及各子命令的 `--help`
- [README](../../README.zh-CN.md) —— 快速上手与端到端工作流
