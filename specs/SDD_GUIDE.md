# SDD 开发指南

> **本文档定义 DataFlow-CV 工程的 SDD 开发指南。**
>
> 目标读者：接手本项目的 AI Agent（如 Claude Code）。也适用于人类开发者。
>
> **通用方法论见 [`SDD_METHODOLOGY.md`](SDD_METHODOLOGY.md)**。本文档聚焦 DataFlow-CV 工程的特有内容，是对通用方法论的工程化补充。

---

## 一、工程架构

### 模块体系

DataFlow-CV 的模块依赖关系（详见 [`modules/index.md`](modules/index.md)）：

```
┌──────────────────────────────────────────────────────────────┐
│                           CLI                                 │
│  (passes LogConfig to modules; click.echo() for terminal UI)  │
└──────┬─────────────────────┬──────────────────┬──────────────┘
       │                     │                  │
       ▼                     ▼                  ▼
┌──────────────┐    ┌──────────────────┐    ┌──────────────┐
│   Convert    │    │    Visualize     │    │   Evaluate   │
│  (pipeline)  │    │  (rendering)     │    │  (metrics)   │
│  LogManager  │    │  LogManager      │    │  LogManager  │
└──────┬───────┘    └───────┬──────────┘    └──────┬───────┘
       │                    │                      │
       │    ZERO CROSS-     │    ZERO CROSS-       │
       │    DEPENDENCY      │    DEPENDENCY        │
       │                    │                      │
       ▼                    ▼                      ▼
┌──────────────────────────────────────────────────────────────┐
│                         Label                                 │
│  Data Models + Handlers (receive logger from caller)          │
└──────────────────────────────────────────────────────────────┘
       │                    │                      │
       └────────────────────┼──────────────────────┘
                            ▼
┌──────────────────────────────────────────────────────────────┐
│                    util/logging.py                             │
│  LogManager + format helpers (shared infrastructure)           │
└──────────────────────────────────────────────────────────────┘
```

### 架构硬约束（写代码前默念一遍）

| # | 约束 | 违反后果 |
|---|------|----------|
| 1 | Convert ↔ Visualize：**零交叉依赖** | 循环导入、模块耦合 |
| 2 | Evaluate ↔ Convert：**零交叉依赖** | 循环导入、模块耦合 |
| 3 | Evaluate ↔ Visualize：**零交叉依赖** | 循环导入、模块耦合 |
| 4 | Convert → Label：**仅通过公共接口** | 绕过 handler 直接操作文件 |
| 5 | Visualize → Label：**仅通过公共接口** | 同上 |
| 6 | Evaluate → Label：**仅通过公共接口** | 同上 |
| 7 | CLI → Convert/Visualize/Evaluate：**不直接导入 label 或 pycocotools** | 打破分层 |
| 8 | **日志归属**：所有日志由模块产出，CLI 仅用 `click.echo()` | 日志混乱、不可控 |

### 外部格式

DataFlow-CV 涉及三种标注格式，按以下固定顺序排列（贯穿代码库的所有 listing）：

```
YOLO → LabelMe → COCO
```

---

## 二、开发工作流（工程特有补充）

> 通用四步工作流（影响范围 → spec → 上下文文档 → 计划）参见 [`SDD_METHODOLOGY.md`](SDD_METHODOLOGY.md) 第二章。

### 2.1 第一步：确定影响范围（工程模板）

1. 改动涉及哪个模块？（Label / Convert / Visualize / Evaluate / CLI）
2. 涉及哪个外部格式？（YOLO / LabelMe / COCO）
3. 是否跨模块？（如果跨 Convert 和 Visualize，需特别小心）

### 2.2 第二步：读 spec（工程映射表）

按改动类型找对应 spec：

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

### 2.3 第三步：对照 CLAUDE.md

重点关注：
- **Known Gotchas**（常见陷阱）
- **Critical Implementation Details**（坐标系统、RLE 编码、原生坐标存储）

### 2.4 工程特有实现细节

**坐标系统（最容易出错的地方）**

```
原生格式存储（无统一内部模型）—— 每个 handler 存原生坐标：

┌────────┬───────────┬──────────────────┬──────────────────────┐
│ Format │ Bbox 原点 │ 坐标空间          │ 示例                 │
├────────┼───────────┼──────────────────┼──────────────────────┤
│ YOLO   │ Center    │ 归一化 [0, 1]     │ (cx, cy, w, h)       │
│ LabelMe│ Top-left  │ 绝对像素          │ (x_tl, y_tl, w, h)   │
│ COCO   │ Top-left  │ 绝对像素          │ (x_tl, y_tl, w, h)   │
└────────┴───────────┴──────────────────┴──────────────────────┘

坐标转换仅在 converter.convert_annotations() 中进行：
- BoundingBox 是纯 dataclass（无 xyxy()/xywh_abs() 等方法）
- 转换时直接计算新 BoundingBox 实例：
  YOLO→COCO:  cx_abs = x * img_w → x_tl = cx_abs - w_abs/2
  COCO→YOLO:  cx_abs = x + w/2   → cx_norm = cx_abs / img_w
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

---

## 三、Specs 导航地图

### 3.1 我要找什么？

```
"YOLO 文件格式是什么样的？"
  → specs/formats/spec_yolo_format.md

"COCO JSON 里 bbox 是左上角还是中心？"
  → specs/formats/spec_coco_format.md（Coordinate System 节）

"LabelMe JSON 有哪些必填字段？"
  → specs/formats/spec_labelme_format.md（Validation Rules 节）

"YOLO 转 COCO 的坐标怎么转换？"
  → specs/formats/spec_conversion.md（YOLO↔COCO 节）

"Converter 的 execute 流程是什么？"
  → specs/modules/spec_convert.md（BaseConverter Pipeline 节）

"Visualizer 怎么画多边形的？"
  → specs/modules/spec_visualize.md（Drawing Pipeline 节）

"CLI 的 exit code 分别代表什么？"
  → specs/modules/spec_cli.md（Exception Hierarchy 节）

"mAP50 和 mAP50_95 是怎么计算的？"
  → specs/evaluate/spec_evaluate_metrics.md（mAP 节）

"目标检测和实例分割的评估有什么区别？"
  → specs/evaluate/spec_evaluate_tasks.md（Detection vs Segmentation 节）

"Evaluate 模块的输入输出是什么？"
  → specs/modules/spec_evaluate.md（Public API 节）

"各模块之间的依赖关系是怎样的？"
  → specs/modules/index.md（Architecture Constraint 图）
```

### 3.2 文件清单

```
specs/
├── SDD_METHODOLOGY.md             # 通用 SDD 方法论（可跨工程复用）
├── SDD_GUIDE.md                   # 本文档 — DataFlow-CV 工程开发指南
│
├── formats/                       # WHAT — 外部格式契约
│   ├── index.md                   # Formats 层概览
│   ├── spec_yolo_format.md        # YOLO .txt 格式权威定义
│   ├── spec_labelme_format.md     # LabelMe .json 格式权威定义
│   ├── spec_coco_format.md        # COCO .json 格式权威定义
│   └── spec_conversion.md         # 转换规则（坐标变换、类别映射）
│
├── evaluate/                      # WHAT — 评估指标契约
│   ├── index.md                   # 评估层概览
│   ├── spec_evaluate_fundamentals.md  # IoU, 匹配规则, TP/FP/FN, 混淆矩阵
│   ├── spec_evaluate_metrics.md       # P/R/F1, PR曲线, AP/mAP/AR, 尺度分层
│   └── spec_evaluate_tasks.md         # 检测/分割评估, COCO 12项标准
│
└── modules/                       # HOW — 内部模块架构
    ├── index.md                   # Modules 层概览 + 依赖图
    ├── spec_label.md              # Label 模块（数据模型 + Handler 接口）
    ├── spec_convert.md            # Convert 模块（Pipeline + 3 Converter + RLE）
    ├── spec_visualize.md          # Visualize 模块（渲染管线 + ColorManager）
    ├── spec_evaluate.md           # Evaluate 模块（评估管线 + API + 数据模型）
    ├── spec_cli.md                # CLI 模块（命令签名 + 异常层次 + 退出码）
    └── spec_logging.md            # Logging 模块（LogManager + LogConfig + format helpers）
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

> 通用清单（测试、格式、文档同步）见 [`SDD_METHODOLOGY.md`](SDD_METHODOLOGY.md) 第三章。
> 以下是 DataFlow-CV 工程的特有检查项。

每次改动后自查：

- [ ] 6 条架构硬约束未被违反（详见第一章：Convert/Visualize/Evaluate 模块间零交叉依赖、各模块仅通过公共接口访问 Label、CLI 不直接导入 label）
- [ ] 坐标转换直接构造新 BoundingBox 实例（BoundingBox 是纯 dataclass，无 xyxy()/xywh_abs() 方法）
- [ ] RLE 编码使用了 `latin1` 而非 `utf-8`
- [ ] Converter state（`_source_annotations_for_target`）在 `finally` 中清理
- [ ] `strict_mode` 正确透传（converter/evaluator → handler）
- [ ] DT 的 `score` 字段存在校验（evaluator）
- [ ] 新增函数/类有对应的测试
- [ ] `pytest -x -q` 全部通过
- [ ] 行为变化已同步更新 specs（P0）
- [ ] 新增架构细节/陷阱已同步更新 CLAUDE.md（P1）
- [ ] API / 功能入口变化已同步更新 README.md（P1）
- [ ] 用户接口变化已同步更新 samples/ 示例代码（P2）

---

## 六、参考

- **SDD_METHODOLOGY.md**：通用 SDD 方法论（可跨工程复用）
- **CLAUDE.md**：项目架构、关键细节、已知陷阱、开发命令
- **README.md**：用户文档、安装、快速开始、项目结构
- **Bug Report**：`~/.claude/plans/bug-p0-p1-p2-glowing-hellman.md`
