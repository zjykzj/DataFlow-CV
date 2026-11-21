# FAQ / 故障排查

使用 dataflow-cv 时的常见错误与解决方法。

## "无法自动检测标签格式"

格式自动检测会检查你传入的路径:单个 `.json` → COCO;只有 `.txt` 的目录 →
YOLO;`.json` 目录(先查 `shapes` → LabelMe,`images` → COCO)。

- 混合扩展名或空目录无法检测。
- 请确保传入的是**标签**(如 `yolo_labels/`)而不是图像。
- `analyse stats` 传入多个路径时,格式必须相同。

## "图片未找到"警告

标签与图像按文件名主干匹配 —— 标签文件**必须与图像同名**
(`000001.txt` ↔ `000001.jpg`)。

- 例外:**LabelMe** 文件只要含合法的 `imageWidth`/`imageHeight`,就不要求
  图像文件存在(尺寸从 JSON 读取)。

## "需要 classes.txt"

`yolo2coco`、`yolo2labelme`、`labelme2yolo`、`labelme2coco`、
`visualize yolo` 需要类别文件(每行一个类别名;行号即类别 id)。

- 先从 COCO 转起?`coco2yolo` / `coco2labelme` 会在输出旁自动生成
  `classes.txt`;用 `-c` 可控制顺序。

## 校验错误:strict 与 non-strict

默认(`strict_mode=True`,不带 flag)下非法标注直接报错中止。加
`--no-strict`(CLI)或 `strict_mode=False`(API)会跳过非法文件/行继续
处理 —— 问题以警告形式报告。(`analyse` 始终为 non-strict。)

## "No module named pycocotools"

RLE 转换与评估需要 pycocotools:

```bash
pip install dataflow-cv[coco]
```

## RLE 转换崩溃(UnicodeDecodeError)

RLE `counts` 字节必须使用 **latin1**,绝不使用 UTF-8。库内部已正确处理 ——
如果遇到解码错误,说明你喂入的是外部工具以 UTF-8 编码的 RLE 数据;请用
latin1 重新保存,或改用 polygon 分割。

## 姿态(关键点)数据被丢弃 / 行被拒绝

姿态处理是**显式**的 —— token 数无法安全区分 YOLO pose 行与分割行:

- 转换:加 `--pose`(六个方向);`--kpts-config kpts.yaml` 提供关键点
  名称/骨架。
- 可视化:YOLO 加 `--pose`(LabelMe/COCO 自动);`--kpts-config` 补充骨架线。
- 分析:`--kpts-config` 增加关键点统计(对 YOLO 源同时启用 pose 行解析)。
- 不加对应 flag 时,关键点被丢弃并警告(YOLO pose 行则会被误解析为分割
  或直接拒绝)。

## YOLO 预测文件被误解析或拒绝

预测文件每行多一个 `confidence` token(检测 6 个 token vs 标签 5 个)。
对 `yolo2coco` 使用 `--prediction` —— 输出纯 JSON 列表(`loadRes()` 的
标准 DT 格式)。

## COCO→YOLO / COCO→LabelMe:`images/` 输出目录为空

这是设计使然 —— 转换器按图像文件名主干命名输出,`images/` 目录留给你
自行放入图片。图片永远不会被复制。

## 转换后部分 COCO 标注消失

`coco2yolo` / `coco2labelme` 默认跳过 `iscrowd=1`(crowd)标注。加
`--save-crowd` 可将其写成普通行/shape。

## 可视化窗口关不掉

窗口管理器的关闭按钮(X)可能不可靠 —— 请始终按 `q` 或 `ESC`。键盘
控制:`←`/`↑` 上一张,`→`/`↓`/`Enter`/`Space` 下一张,`s` 快照,`h` 帮助。

## mAP 和 P/R/F1 能同时得到吗?

单次运行中二者互斥:`--prf1` 只计算单阈值 P/R/F1(快),不加 `--prf1`
则走完整 COCOeval mAP 管线。想要两者,运行两次。
