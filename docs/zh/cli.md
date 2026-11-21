# CLI 参考

> dataflow-cv v3.6 —— 完整命令清单。[README](../../README.zh-CN.md) 展示最常见
> 命令;本页记录每个子命令与全部选项。

## 命令树

```
dataflow-cv
├── analyse  stats | split | filter | partition | sample
├── convert  yolo2coco | yolo2labelme | labelme2yolo | labelme2coco | coco2yolo | coco2labelme
├── visualize  yolo | labelme | coco
└── evaluate  detection | segmentation | keypoints
```

所有必需参数(图像目录、标签目录、类别文件、输出路径)均为**位置参数**。在任意
子命令后使用 `--help` 查看权威选项列表。

## 通用选项

| 选项 | 适用范围 | 含义 |
|------|---------|------|
| `--verbose` | 全部命令 | 启用各模块文件日志(DEBUG 级日志文件写入 `--log-dir`,默认 `./logs/`) |
| `--log-dir PATH` | 全部命令 | 日志输出目录 |
| `--no-strict` | convert / visualize | 跳过无效标注并警告,而非中止(analyse 始终为 non-strict) |

## 🔍 分析

格式自动检测:指向你的标签 —— 单个 `.json` → COCO;只有 `.txt` 的目录 → YOLO;
`.json` 目录(先查 `shapes` → LabelMe,`images` → COCO)。混合扩展名或空目录会报错。

### `analyse stats`

```bash
dataflow-cv analyse stats yolo_labels/ --image-dir images/ --class-file classes.txt
dataflow-cv analyse stats labelme_json/
dataflow-cv analyse stats coco_annotations.json

# 多路径合并统计(需同格式); -R 递归子目录
dataflow-cv analyse stats yolo_train/ yolo_val/ --recursive

# 按数量降序排序(默认:类别 ID 升序)
dataflow-cv analyse stats --sort-by count --descending yolo_labels/

# Pose:关键点统计(逐关键点标注率;YOLO pose 源必需 —— 该选项同时启用 pose 行解析)
dataflow-cv analyse stats --kpts-config kpts.yaml yolo_pose_labels/

# 详细日志
dataflow-cv analyse stats --verbose yolo_labels/ --class-file classes.txt
```

### `analyse split`(仅 YOLO / LabelMe)

```bash
# 标签 / 图像 / 两者模式 → outputs/train/ + outputs/val/
dataflow-cv analyse split -l yolo_labels/ outputs/ --ratio 0.8 --seed 42 -c classes.txt
dataflow-cv analyse split -i images/ outputs/ --ratio 0.8
dataflow-cv analyse split -l yolo_labels/ -i images/ outputs/ --ratio 0.8
```

选项:`-l/--label-dir`、`-i/--image-dir`、`-c/--class-file`、`--ratio`
(默认 0.8)、`--seed`(默认 42)、`--move`(移动而非复制)。

### `analyse filter`(全部格式)

```bash
# 保留类别子集,按新的 classes.txt 重映射 ID
#   → filtered/ 接收过滤后的标签 + 新的 classes.txt
dataflow-cv analyse filter yolo_labels/ classes.txt classes_new.txt filtered/
dataflow-cv analyse filter coco_annotations.json classes.txt classes_new.txt filtered/
```

### `analyse partition`(仅 YOLO / LabelMe)

```bash
# N 路分区 —— 标签主导,图像按文件名主干跟随
#   → parts/part_1/ … parts/part_N/(两者模式下各自含 labels/ + images/)
dataflow-cv analyse partition -n 4 --label-dir yolo_labels/ --image-dir images/ parts/
dataflow-cv analyse partition -n 4 --image-dir images/ --shuffle parts/
```

选项:`-n/--num`(≥ 2)、`--label-dir`、`--image-dir`、`--shuffle`、
`--seed`、`--class-file`、`--move`。

### `analyse sample`(仅 YOLO / LabelMe)

```bash
# 收集 N 个文件 —— 平铺文件至 output/(两者模式为 output/labels/ + output/images/)
dataflow-cv analyse sample -l yolo_labels/ output/ -n 10
dataflow-cv analyse sample -i images/ output/ -n 10 --no-shuffle
dataflow-cv analyse sample -l yolo_labels/ -i images/ output/ -n 5 --seed 42
```

选项:`-n/--num`、`--label-dir`、`--image-dir`、`--shuffle`(默认开启)、
`--seed`、`--class-file`、`--move`。

## 🔄 转换

六个方向 —— 输出为单个 COCO 文件或每图一个文件:

```bash
# YOLO → COCO                      → output.json:单个 COCO 文件
dataflow-cv convert yolo2coco images/ yolo_labels/ classes.txt output.json

# YOLO → COCO(RLE 编码)
dataflow-cv convert yolo2coco images/ yolo_labels/ classes.txt output.json --do-rle

# YOLO → LabelMe                   → labelme_json/:每图一个 .json
dataflow-cv convert yolo2labelme images/ yolo_labels/ classes.txt labelme_json/

# LabelMe → YOLO                   → yolo_labels/:每图一个 .txt + classes.txt
dataflow-cv convert labelme2yolo labelme_json/ classes.txt yolo_labels/

# LabelMe → COCO                   → output.json:单个 COCO 文件
dataflow-cv convert labelme2coco labelme_json/ classes.txt output.json

# COCO → YOLO                      → yolo_labels/labels/*.txt + classes.txt
dataflow-cv convert coco2yolo input.json yolo_labels/

# COCO → LabelMe                   → labelme_json/:每图一个 .json
dataflow-cv convert coco2labelme input.json labelme_json/

# iscrowd=1(crowd)标注在两个方向默认跳过;如需写成普通行/shape 需显式开启:
dataflow-cv convert coco2yolo --save-crowd input.json yolo_labels/

# YOLO 预测文件 → COCO(输出:纯 JSON 列表 —— 预测格式)
dataflow-cv convert yolo2coco --prediction images/ yolo_preds/ classes.txt pred.json
```

### 选项

| 选项 | 方向 | 含义 |
|------|------|------|
| `--do-rle` | `yolo2coco` | 分割输出 RLE 编码 |
| `--prediction` | `yolo2coco` | 读取 YOLO 预测文件(多一个 confidence token);输出纯 JSON 列表(Variant B) |
| `--pose` | 六个方向 | 姿态转换 —— 见下 |
| `--kpts-config PATH` | 六个方向 | 关键点定义 YAML(名称 + 骨架) |
| `--save-kpts-config PATH` | `coco2yolo` / `coco2labelme` | 从 COCO categories 导出定义 YAML |
| `--save-crowd` | `coco2yolo` / `coco2labelme` | 将 `iscrowd=1` 标注写成普通行/shape |

### 姿态转换

姿态处理是**显式**的 —— 仅凭 token 数无法区分 YOLO pose 行与分割行(见
[README](../../README.zh-CN.md) 任务类型章节)。

```bash
dataflow-cv convert coco2yolo --pose coco_pose.json yolo_pose_labels/
dataflow-cv convert yolo2coco --pose --kpts-config kpts.yaml images/ yolo_pose_labels/ classes.txt pose.json
dataflow-cv convert coco2yolo --pose --save-kpts-config kpts.yaml coco_pose.json yolo_pose_labels/   # 导出定义 YAML

# Pose 预测文件(yolo2coco --prediction --pose → keypoints + score)
dataflow-cv convert yolo2coco --prediction --pose images/ yolo_pose_preds/ classes.txt pred.json
```

| 行为 | 非姿态模式(默认) | 姿态模式(`--pose`) |
|------|------------------|--------------------|
| **含**关键点的对象 | 丢弃关键点(每张图一条警告);bbox/分割照常转换 | 转换关键点;YOLO 目标丢弃分割(pose 行无法携带) |
| **不含**关键点的对象 | 照常转换 | 跳过(每张图一条警告) |

> 📂 **COCO→YOLO / COCO→LabelMe 输出命名**:每个 `.txt` / `.json` 以图片文件名
> 主干命名 —— 保留前导零(`000001.jpg` → `000001.txt`)。转换器会生成 `labels/`
> (或 `.json` 文件)+ `classes.txt`;`images/` 目录会被创建但保持**为空** ——
> 图片不会被拷贝,请自行放入。

## 🎨 可视化

```bash
# 可视化 YOLO 标注
dataflow-cv visualize yolo images/ yolo_labels/ classes.txt --save visualized/

# 可视化 LabelMe 标注
dataflow-cv visualize labelme images/ labelme_json/ --save visualized/

# 可视化 COCO 标注
dataflow-cv visualize coco images/ coco_annotations.json --save visualized/

# Pose:关键点 + 骨架渲染(v=2 实心绿点,v=1 空心橙点)
dataflow-cv visualize yolo --pose --kpts-config kpts.yaml images/ yolo_pose_labels/ classes.txt --save visualized/
dataflow-cv visualize coco images/ coco_pose.json --save visualized/   # 骨架来自 COCO categories

# 详细日志 + 无界面模式
dataflow-cv visualize yolo --verbose --no-display images/ yolo_labels/ classes.txt --save visualized/
```

选项:`--pose`(YOLO)、`--kpts-config`(YOLO / LabelMe —— 补充骨架线;COCO 从其
categories 读取)、`--display/--no-display`、`-s/--save PATH`、`--verbose`。

**键盘控制**(交互窗口):`←`/`↑` 上一张,`→`/`↓`/`Enter`/`Space` 下一张,`s`
快照(保存至 `./visualized_snapshots/`),`h` 帮助,`q`/`ESC` 退出。首尾边界操作
为空操作;标题栏显示 `[N/T]` 位置。窗口管理器关闭按钮可能不可靠 —— 请始终使用
键盘。

## 📊 评估

评估始终消费 **COCO 格式**的 GT/DT JSON。需要两个文件:

| 文件 | 角色 | 格式 | 如何生成 |
|------|------|------|---------------|
| **`anno.json`** | 真值 (GT) | 完整 COCO 字典(`images`、`annotations`、`categories`) | `yolo2coco`(标注模式) |
| **`pred.json`** | 检测结果 (DT) | 纯 JSON 列表(含 `score`) | `yolo2coco --prediction` |

### ① 准备数据

```bash
# GT:YOLO 标注 → COCO
dataflow-cv convert yolo2coco images/ yolo_labels/ classes.txt anno.json

# DT:YOLO 预测 → COCO(模型输出需加 --prediction)
dataflow-cv convert yolo2coco --prediction images/ yolo_preds/ classes.txt pred.json
```

> ⚠️ YOLO 预测文件必须使用 `--prediction` —— 它们的每一行多一个 `confidence`
> token。该选项输出**纯 JSON 列表**(而非完整 COCO 字典),这是 `loadRes()` 的
> 标准 DT 格式。只有 `yolo2coco` 支持 `--prediction`;`labelme2coco` 不需要它
> (LabelMe 没有标注与预测之分)。

### ② 运行评估

```bash
# 目标检测(bbox IoU)
dataflow-cv evaluate detection anno.json pred.json
dataflow-cv evaluate detection --verbose anno.json pred.json           # 按类别明细
dataflow-cv evaluate detection --prf1 anno.json pred.json              # 仅 P/R/F1(跳过 mAP)
dataflow-cv evaluate detection --prf1 --prf1-iou 0.75 --prf1-method micro anno.json pred.json

# 实例分割(mask IoU)
dataflow-cv evaluate segmentation anno.json pred.json
dataflow-cv evaluate segmentation --verbose anno.json pred.json

# 姿态估计(OKS)—— sigmas 取自定义 YAML;缺省时使用 COCO 17 关键点官方值
#(其他 k 退化为均匀 0.05 + 警告)
dataflow-cv evaluate keypoints --kpts-config kpts.yaml anno_pose.json pred_pose.json
dataflow-cv evaluate keypoints --prf1 anno_pose.json pred_pose.json   # 基于 OKS 的 P/R/F1

# 结果保存为 JSON
dataflow-cv evaluate detection --output results.json anno.json pred.json

# P/R/F1 结果保存为 JSON(阈值、聚合方式、整体与逐类指标)
dataflow-cv evaluate detection --prf1 --output prf1_results.json anno.json pred.json

# 自定义日志目录
dataflow-cv evaluate detection --verbose --log-dir logs/eval/ anno.json pred.json
```

选项:`--prf1`(单阈值 P/R/F1,跳过 COCOeval mAP 管线;两者都要则运行两次)、
`--prf1-iou`(默认 0.5)、`--prf1-conf`(默认 0.0)、`--prf1-method macro|micro`
(默认 macro)、`-o/--output PATH`、`--kpts-config`(关键点 sigmas)、`--verbose`、
`--log-dir`。

### ③ 检测 vs 分割 vs 姿态

三种评估模式,区别在于重叠程度的度量方式:

- **目标检测** —— 边界框 IoU。GT 和 DT 需要 `bbox`;DT 还需要 `score`。
- **实例分割** —— 掩码 IoU。GT 和 DT 需要 `bbox`、`segmentation`(多边形或 RLE)以及 `area`;DT 还需要 `score`。
- **姿态估计** —— OKS(对象关键点相似度)。GT 和 DT 需要 `keypoints`(扁平 `[x, y, v, ...]` 三元组)和 `area`;DT 还需要 `score`。逐关键点 sigmas 依次来自 `--kpts-config`、COCO 17 关键点官方值、均匀回退值。

`yolo2coco`(标注模式)和 `yolo2coco --prediction`(预测模式)会自动填充三种
模式所需的全部字段 —— 无需手动编辑。
