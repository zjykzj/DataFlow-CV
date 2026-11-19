# Pose 支持路线图（Roadmap）

> 状态：已确认，待执行 | 最后更新：2026-11-18
>
> 目标：为 DataFlow-CV 增加关键点（pose）数据集支持，按 Label → Visualize → Convert → Evaluate → Analyse 顺序推进，每阶段一个 release（v3.3.0 → v3.6.0）。

## 已锁定的设计决策

| # | 决策 | 内容 |
|---|------|------|
| D1 | 关键点定义文件 | **YAML 格式**（用户拍板 2026-11-18，原 O1）：关键点名称 + 可选骨架连线/sigmas，作为生态配置的超集扩展。可直接复用 X-AnyLabeling 的 `yolov8_pose.yaml`（`has_visible` + 每类名称列表）与 ultralytics 风格配置；COCO→X 时可从 COCO categories 自动导出 sidecar；X→COCO 缺失时用 `keypoint_0..k-1` 占位名 + 空骨架（合法 COCO，语义降级 + WARNING） |
| D2 | LabelMe pose 约定 | 每关键点一个 `point` shape（label = 关键点名称），同实例共用 `group_id`，可选同 group_id 的 `rectangle` 作 bbox —— X-AnyLabeling 官方文档原文："group_id 用于关联同一组内的多个对象 **(e.g., pose keypoints)**"（已核实：labelme 工具无原生关键点类型，point+group_id 是事实标准，X-AnyLabeling 的产出与本约定完全兼容） |
| D3 | 任务模式 | 显式 `--pose` 标志，不做 token 数自动检测（17 关键点行当前报错算幸运，18 关键点行会静默解析成 29 点分割，auto-detect 有真实误解析风险） |
| D4 | YOLO 关键点数量来源 | 行 token 数推断 k = (n−5)/3（预测行 k = (n−6)/3）；定义文件名称数必须与 k 一致，否则报错 |
| D5 | 可见性映射 | COCO ↔ YOLO 恒等映射 v: 0→0, 1→1, 2→2（ultralytics 官方文档已核实行格式 `<class> <x> <y> <w> <h> <px1> <py1> <p1-visibility> ...`；精确语义 P3 与 ultralytics json2yolo 交叉验证确认）；写 COCO 时 `num_keypoints` = count(v>0) |
| D6 | 开发顺序 | Label 底层 → Visualize → Convert → Evaluate → Analyse（用户拍板） |
| D7 | 版本切分 | 每阶段一个 release：P1+P2 = v3.3.0，P3 = v3.4.0，P4 = v3.5.0，P5 = v3.6.0 |
| D8 | 过渡期数据安全 | v3.3.0 的 6 个 converter 对含关键点数据**显式报错**（"pose 转换见 3.4.0"），不允许静默丢弃 —— handler 能读但 converter 不处理时，静默丢失是真实风险 |

## 待定项（已有推荐默认，执行前可推翻）

> O1（定义文件格式）已于 2026-11-18 拍板为 **YAML**，移入 D1。

| # | 待定 | 推荐默认 |
|---|------|----------|
| O2 | COCO→LabelMe 的 v=1（标注但遮挡） | 降级为 v=2 写入 + WARNING（LabelMe 无可见性概念，v=0 对应缺省） |
| O3 | pose 模式下混入无关键点对象 | 跳过 + WARNING（沿用 iscrowd 跳过先例，见 CLAUDE.md gotcha #46） |
| O4 | OKS sigmas | k=17 用 COCO 官方 17 点 sigmas；其他读定义文件，缺省均匀 0.05 + WARNING |
| O5 | 预测模式 | `yolo2coco --prediction` 支持 pose 行（6+3k token，含 confidence），随 P3 交付 |

## 阶段总览

| 阶段 | 版本 | 内容 | 依赖 |
|------|------|------|------|
| P1 | 3.3.0 | Label：Keypoints 数据模型 + 3 个 handler 的 pose 读写 + 定义文件加载器 | — |
| P2 | 3.3.0 | Visualize：骨架连线 + 关键点渲染（可见性配色） | P1 |
| P3 | 3.4.0 | Convert：6 方向 pose 变换 + `--pose`/`--kpts-config` CLI + 预测模式 | P1, P2 |
| P4 | 3.5.0 | Evaluate：KeypointEvaluator（OKS）+ PRF1-OKS | P1 |
| P5 | 3.6.0 | Analyse：关键点标注率 / 可见性分布统计 | P1 |

## P1 — Label 底层（v3.3.0）

**规格先行**（SDD）：`specs/modules/spec_label.md`（Keypoints 模型 + handler pose 契约，minor bump）、`specs/formats/spec_yolo_format.md`（pose 行布局：5+3k token、归一化坐标 + v 标志）、`specs/formats/spec_labelme_format.md`（point+group_id 约定）、`specs/formats/spec_coco_format.md`（keypoints 数组 + num_keypoints + categories.keypoints/skeleton）。交叉扫描：CLAUDE.md gotchas、README。

**实现步骤**：
1. `models.py`：新增 `Keypoints` dataclass（三元组序列 (x, y, v)，native 坐标）；`ObjectAnnotation` 加 `keypoints: Optional[Keypoints]`，`__post_init__` 放宽为"bbox / segmentation / keypoints 至少其一"
2. YOLO handler：pose 模式解析 5+3k 行（v ∈ {0,1,2}，数据集内 k 一致）；写 pose 行（分割优先规则 gotcha #31 需重新审视：pose 行不携带 seg）
3. LabelMe handler：按 group_id 分组读 point shapes + 可选 rectangle，按定义文件名称排序；写 point+group_id
4. COCO handler：读/写 keypoints 数组、num_keypoints、categories 的 keypoints/skeleton 透传
5. 定义文件加载器（`dataflow/label/utils.py` 或独立模块）：YAML 解析（新增 PyYAML 依赖，写入 `pyproject.toml`）+ 校验（名称唯一、skeleton 索引越界检查、兼容 X-AnyLabeling `has_visible`/per-class 结构）
6. **Converter 守卫**（D8）：6 个 converter 检测到关键点时抛清晰错误
7. 测试：三格式 pose 读写往返 + fixtures（coco-seg val2017 person_keypoints 子集 + 合成数据）

**验收**：三格式 handler 读真实 val2017 pose 数据无丢失；converter 对 pose 数据显式报错。

## P2 — Visualize（v3.3.0，与 P1 同 release）

**规格先行**：`specs/modules/spec_visualize.md`（骨架渲染契约，minor bump）。

**实现步骤**：
1. `RenderData` 扩展关键点 + 骨架字段
2. 绘制：关键点圆点按可见性配色（v=2 实心、v=1 异色空心、v=0 跳过）+ 骨架连线
3. 骨架来源：COCO 源从 categories 读；YOLO/LabelMe 源走 `--kpts-config` 定义文件
4. CLI：visualize 三个子命令加可选 `--kpts-config`
5. 测试：渲染冒烟测试 + fixture

**验收**：val2017 person 可视化人工抽查（骨架连线正确、遮挡点区分明显）。

## P3 — Convert（v3.4.0）

**规格先行**：`specs/formats/spec_conversion.md`（pose 坐标变换：COCO 像素 ↔ YOLO 归一化、可见性透传、num_keypoints 计算、骨架传播规则、有损边界，major bump）、`specs/modules/spec_convert.md`、`specs/modules/spec_cli.md`（`--pose`/`--kpts-config`）。

**实现步骤**：
1. `convert/utils.py`：关键点坐标变换纯函数（归一化/反归一化 + v 透传）
2. 6 方向 pose 处理：
   - COCO→YOLO：5+3k 行，seg 丢弃 + WARNING（YOLO pose 行无法携带分割）
   - COCO→LabelMe：rectangle + points + group_id；v=1 按 O2 降级
   - LabelMe→COCO/YOLO：定义文件排序，缺失关键点 → v=0
   - YOLO→COCO：num_keypoints 计算 + categories keypoints/skeleton 来自定义文件（缺省占位 + WARNING）
   - YOLO→LabelMe：point shapes，标签名来自定义文件
3. COCO→X 骨架导出：`--save-kpts-config` 从 COCO categories 自动生成定义文件 sidecar
4. `yolo2coco --prediction` pose 支持（O5）
5. CLI：6 个子命令加 `--pose` + `--kpts-config`
6. O3 混合数据规则
7. 测试：与 ultralytics json2yolo 输出交叉验证；COCO↔YOLO 往返 OKS/可见性对拍；混合数据规则

**验收**：val2017 person_keypoints → YOLO pose → 回 COCO，关键点坐标/可见性/num_keypoints 全对齐；LabelMe 产出手工在 labelme 工具中打开可编辑。

## P4 — Evaluate（v3.5.0）

**规格先行**：`specs/modules/spec_evaluate.md`（OKS 定义、sigmas 来源（O4）、KeypointEvaluator 契约，minor bump）。

**实现步骤**：
1. `KeypointEvaluator`：GT/DT COCO JSON（Variant A/B）含 keypoints；`iouType='keypoints'` COCOeval；sigmas 按 O4 注入
2. `compute_pr_f1` 扩展 `iou_type='keypoints'`（pycocotools `computeOks`）
3. CLI：新增 `evaluate keypoints` 子命令；校验规则对齐 segm 路径（缺 keypoints/score 即报错）
4. 测试：合成 OKS 用例 + val2017 person 官方 mAP 对拍

**验收**：val2017 person_keypoints mAP 与 pycocotools 参考值一致（容差内）。

## P5 — Analyse（v3.6.0）

**规格先行**：`specs/modules/spec_analyse.md`（pose 统计契约，minor bump）。

**实现步骤**：
1. StatsAnalyser 扩展：每类关键点标注率（各关键点 v>0 / v=2 占比）、可见性分布、缺失关键点统计表
2. 输出表遵循既有 format_table 风格
3. 测试：val2017 统计合理性抽查（如 person 关键点标注率显著分布）

**验收**：真实数据集统计输出与 COCO API 人工核对一致。

## 发布收尾（每个 release 通用）

按既有流程：CHANGELOG 条目 → 版本号三处 bump（pyproject.toml / `dataflow/__init__.py` / CHANGELOG）→ 提交 + tag → GitHub Release → PyPI（Actions 自动）→ claude-skills 插件同步（gotcha #44）→ py39/py312 环境更新。

## 验证数据

- **主验证集**：coco-seg val2017（同一文件含 person_keypoints 标注，零新增数据获取成本）
- **交叉验证**：ultralytics json2yolo 输出对拍（P3）
- **手工验证**：LabelMe 产出的 labelme 工具打开检查（P3）
