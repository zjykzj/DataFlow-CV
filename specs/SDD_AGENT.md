# SDD Agent 开发指南

> **本文档定义 DataFlow-CV 项目的 SDD Agent 开发方法论。**
>
> 目标读者：接手本项目的 AI Agent（如 Claude Code）。也适用于人类开发者。

---

## 一、核心理念

```
Specs（什么是对的）→ CLAUDE.md（代码怎么写）→ 代码实现
     ↑                                              |
     └──────────── 测试验证 ←────────────────────────┘
```

**SDD Agent 开发的三层体系：**

| 层级 | 文件 | 角色 | 修改频率 |
|------|------|------|----------|
| **Specs** | `specs/` | 行为契约——定义"什么是对的" | 很少（需求变更时才改） |
| **CLAUDE.md** | `CLAUDE.md` | 开发上下文——描述"代码怎么写的" | 随代码演进 |
| **Code** | `dataflow/` | 实现——实际运行的代码 | 日常 |

**开发铁律**：Specs 是最高权威。如果代码行为与 specs 冲突，以 specs 为准，改代码。

---

## 二、开发工作流

### 2.1 接到新任务时

**第一步：确定影响范围**

问自己三个问题：
1. 改动涉及哪个模块？（Label / Convert / Visualize / CLI）
2. 涉及哪个外部格式？（YOLO / COCO / LabelMe）
3. 是否跨模块？（如果跨 Convert 和 Visualize，需特别小心）

**第二步：按影响范围读 spec（按此顺序）**

| 改动类型 | 必读 spec |
|----------|-----------|
| 修改数据模型 | `specs/modules/spec_label.md` |
| 新增/修改转换方向 | `specs/modules/spec_convert.md` → `specs/formats/spec_conversion.md` |
| 新增/修改可视化 | `specs/modules/spec_visualize.md` |
| 新增/修改评估 | `specs/evaluate/spec_evaluate_metrics.md` → `specs/modules/spec_evaluate.md` |
| 新增 CLI 命令 | `specs/modules/spec_cli.md` |
| 修改 YOLO 读写 | `specs/formats/spec_yolo_format.md` + `specs/modules/spec_label.md` |
| 修改 COCO 读写 | `specs/formats/spec_coco_format.md` + `specs/modules/spec_label.md` |
| 修改 LabelMe 读写 | `specs/formats/spec_labelme_format.md` + `specs/modules/spec_label.md` |
| 跨模块改动 | `specs/modules/index.md`（架构约束图 + 依赖规则） |
| 不确定影响范围 | `specs/modules/index.md` 和 `specs/formats/index.md`（全局概览） |

**第三步：对照 CLAUDE.md**

读 CLAUDE.md 的相关章节，重点关注：
- **Known Gotchas**（9 条常见陷阱）
- **Critical Implementation Details**（坐标系统、RLE 编码、原生坐标存储）

### 2.2 动手写代码时

**架构硬约束（写代码前默念一遍）**

| # | 约束 | 违反后果 |
|---|------|----------|
| 1 | Convert ↔ Visualize：**零交叉依赖** | 循环导入、模块耦合 |
| 2 | Convert → Label：**仅通过公共接口** | 绕过 handler 直接操作文件 |
| 3 | Visualize → Label：**仅通过公共接口** | 同上 |
| 4 | CLI → Convert/Visualize/Evaluate：**不直接导入 label** | 打破分层 |
| 5 | Evaluate ↔ Convert/Visualize：**零交叉依赖** | 循环导入、模块耦合 |
| 6 | Evaluate → Label：**仅通过公共接口** | 绕过 handler 直接操作文件 |

**坐标系统（最容易出错的地方）**

```
内部模型：所有坐标 0-1 归一化，BoundingBox x/y 是中心点

YOLO  ↔  内部模型：center-based, normalized → 直通
COCO  →  内部模型：top-left, absolute pixels → 需要转换
        输出时用 BoundingBox.xyxy() → [x1, y1, w, h]
        ❌ 不要用 xywh_abs() 输出 COCO bbox
LabelMe → 内部模型：绝对像素 → 归一化
```

**RLE 编码（第二容易出错的地方）**

```
写入 JSON：counts_bytes.decode("latin1")  → 字符串
从 JSON 读：counts_str.encode("latin1")   → 字节
❌ 永远不要用 UTF-8 处理 RLE counts
```

**Converter state 清理**

```python
# 所有 converter 的 write 路径必须遵循这个模式：
self._source_annotations_for_target = converted_annotations
try:
    target_handler = self.create_target_handler(target_path, kwargs)
    write_result = target_handler.write(...)
finally:
    self._source_annotations_for_target = None  # 必须在 finally 中清理
```

**Validation 行为**

| 模式 | 行为 |
|------|------|
| `strict_mode=True`（默认） | 校验失败 → 立即报错终止 |
| `strict_mode=False` | 校验失败 → 跳过该条，记录 warning，继续处理 |
| 图片错误 | **总是** warning，不受 strict_mode 影响 |
| `--no-strict`（CLI） | 透传到 converter → handler 的 `strict_mode` |

### 2.3 提交前

```bash
# 1. 跑测试（必须通过）
pytest -x -q

# 2. 格式检查（如果装了 pre-commit 会自动做）
black --check dataflow tests samples
isort --check-only dataflow tests samples
flake8 dataflow tests samples

# 3. 如果改了 spec 相关行为，确认 spec 文档是否需要同步更新
```

**Git commit 格式：**

```bash
git commit -m "$(cat <<'EOF'
<type>(<scope>): <subject>

<body if needed>

Co-Authored-By: DeepSeek-V4.0 <noreply@deepseek.com>
EOF
)"
```

类型：`feat` / `fix` / `docs` / `refactor` / `test` / `style` / `chore`

---

## 三、Specs 导航地图

### 3.1 我要找什么？

```
"YOLO 文件格式是什么样的？"
  → specs/formats/spec_yolo_format.md

"COCO JSON 里 bbox 是左上角还是中心？"
  → specs/formats/spec_coco_format.md（第 4 节：Coordinate System）

"LabelMe JSON 有哪些必填字段？"
  → specs/formats/spec_labelme_format.md（第 7 节：Validation Rules）

"YOLO 转 COCO 的坐标怎么转换？"
  → specs/formats/spec_conversion.md（第 5 节：YOLO↔COCO）

"Converter 的 execute 流程是什么？"
  → specs/modules/spec_convert.md（第 2 节：BaseConverter Pipeline）

"Visualizer 怎么画多边形的？"
  → specs/modules/spec_visualize.md（第 3 节：Drawing Pipeline）

"CLI 的 exit code 分别代表什么？"
  → specs/modules/spec_cli.md（第 7 节：Exception Hierarchy）

"mAP50 和 mAP50_95 是怎么计算的？"
  → specs/evaluate/spec_evaluate_metrics.md（第 6 节：mAP）

"目标检测和实例分割的评估有什么区别？"
  → specs/evaluate/spec_evaluate_tasks.md（第 5 节：Detection vs Segmentation）

"Evaluate 模块的输入输出是什么？"
  → specs/modules/spec_evaluate.md（第 4 节：Public API）

"各模块之间的依赖关系是怎样的？"
  → specs/modules/index.md（Architecture Constraint 图）
```

### 3.2 文件清单

```
specs/
├── SDD_AGENT.md                  # 本文档 — SDD Agent 开发指南
│
├── formats/                      # WHAT — 外部格式契约
│   ├── index.md                  # Formats 层概览
│   ├── spec_yolo_format.md       # YOLO .txt 格式权威定义
│   ├── spec_labelme_format.md    # LabelMe .json 格式权威定义
│   ├── spec_coco_format.md       # COCO .json 格式权威定义
│   └── spec_conversion.md        # 转换规则（坐标变换、类别映射）
│
├── evaluate/                     # WHAT — 评估指标契约
│   ├── index.md                  # 评估层概览
│   ├── spec_evaluate_fundamentals.md  # IoU, 匹配规则, TP/FP/FN, 混淆矩阵
│   ├── spec_evaluate_metrics.md       # P/R/F1, PR曲线, AP/mAP/AR, 尺度分层
│   └── spec_evaluate_tasks.md         # 检测/分割评估, COCO 12项标准
│
├── formats/                      # WHAT — 外部格式契约
│   ├── index.md                  # Formats 层概览
│   ├── spec_yolo_format.md       # YOLO .txt 格式权威定义
│   ├── spec_labelme_format.md    # LabelMe .json 格式权威定义
│   ├── spec_coco_format.md       # COCO .json 格式权威定义
│   └── spec_conversion.md        # 转换规则（坐标变换、类别映射）
│
└── modules/                      # HOW — 内部模块架构
    ├── index.md                  # Modules 层概览 + 依赖图
    ├── spec_label.md             # Label 模块（数据模型 + Handler 接口）
    ├── spec_convert.md           # Convert 模块（Pipeline + 3 Converter + RLE）
    ├── spec_visualize.md         # Visualize 模块（渲染管线 + ColorManager）
    ├── spec_evaluate.md          # Evaluate 模块（评估管线 + API + 数据模型）
    └── spec_cli.md               # CLI 模块（命令签名 + 异常层次 + 退出码）
```

---

## 四、常见开发场景速查

### 场景：新增一种标注格式

1. 在 `specs/formats/` 新增格式 spec（定义文件结构、坐标系统、必填字段）
2. 在 `dataflow/label/` 实现 handler（继承 `BaseAnnotationHandler`）
3. 在 `dataflow/label/models.py` 的 `AnnotationFormat` enum 新增条目
4. 实现 converter（至少与一种现有格式互通）
5. 实现 visualizer
6. 添加 CLI 命令
7. 写测试（`tests/label/`、`tests/convert/`、`tests/visualize/`、`tests/evaluate/`）

### 场景：新增转换方向

1. 在 `specs/formats/spec_conversion.md` 记录坐标变换公式
2. 在 `dataflow/convert/` 新增或修改 converter
3. 如果涉及 CLI，在 `dataflow/cli/commands/convert.py` 新增子命令
4. 写集成测试

### 场景：新增评估能力

1. 在 `specs/evaluate/` 确认或新增指标定义（fundamentals → metrics → tasks 顺序）
2. 在 `dataflow/evaluate/` 实现 evaluator 或 metric 函数
3. 如需 CLI，在 `dataflow/cli/commands/evaluate.py` 新增子命令或选项
4. 写单元测试 + 集成测试（需准备 mini GT/DT 测试数据到 `assets/test_data/evaluate/`）

### 场景：修复 Bug

1. 先确认是 spec 问题还是代码问题
2. 如果是 spec 问题：修改 spec → 改代码 → 更新测试
3. 如果是代码问题：在 spec 中找到对应的行为定义 → 改代码 → 跑测试
4. 检查 CLAUDE.md 的 Known Gotchas 是否需要新增条目

### 场景：新增 CLI 选项

1. 在 `dataflow/cli/commands/utils.py` 的对应装饰器添加 option
2. 在 subcommand 函数中使用新参数
3. 如果是 convert 选项：需要透传到 converter → handler
4. 如果是 visualize 选项：需要透传到 visualizer
5. 写 CLI 测试

---

## 五、代码审查检查清单

每次改动后自查：

- [ ] 4 条架构硬约束未被违反
- [ ] 坐标转换使用了正确的方法（`xyxy()` vs `xywh_abs()`）
- [ ] RLE 编码使用了 `latin1` 而非 `utf-8`
- [ ] Converter state（`_source_annotations_for_target`）在 `finally` 中清理
- [ ] `strict_mode` 正确透传（converter/evaluator → handler）
- [ ] DT 的 `score` 字段存在校验（evaluator）
- [ ] 新增函数/类有对应的测试
- [ ] `pytest -x -q` 全部通过
- [ ] 如果行为变化影响 spec，已同步更新 spec 文档

---

## 六、参考

- **CLAUDE.md**：项目架构、关键细节、已知陷阱、开发命令
- **README.md**：用户文档、安装、快速开始、项目结构
- **Bug Report**：`~/.claude/plans/bug-p0-p1-p2-glowing-hellman.md`
