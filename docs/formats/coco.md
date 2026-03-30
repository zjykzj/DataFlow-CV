# COCO 标签格式

## 目录
- [COCO 标签格式](#coco-标签格式)
  - [目录](#目录)
  - [一、概述](#一概述)
  - [二、文件构成和属性](#二文件构成和属性)
    - [images 数组](#images-数组)
    - [annotations 数组](#annotations-数组)
    - [categories 数组](#categories-数组)
  - [三、标注类型](#三标注类型)
    - [目标检测（边界框）](#目标检测边界框)
    - [实例分割（多边形或 RLE）](#实例分割多边形或-rle)
      - [多边形格式](#多边形格式)
      - [RLE 格式](#rle-格式)
    - [RLE 编码详解](#rle-编码详解)
    - [RLE 编码原理](#rle-编码原理)
    - [COCO RLE 格式](#coco-rle-格式)
    - [二进制 RLE 格式](#二进制-rle-格式)
    - [pycocotools 使用说明](#pycocotools-使用说明)
    - [应用场景](#应用场景)
  - [四、坐标系统](#四坐标系统)
    - [边界框坐标](#边界框坐标)
    - [多边形坐标](#多边形坐标)
    - [RLE 坐标系统](#rle-坐标系统)
    - [坐标转换示例](#坐标转换示例)
      - [COCO → YOLO 转换](#coco--yolo-转换)
      - [COCO → LabelMe 转换](#coco--labelme-转换)
  - [五、与其他格式的转换](#五与其他格式的转换)
    - [COCO ↔ YOLO 转换](#coco--yolo-转换-1)
      - [转换步骤](#转换步骤)
      - [RLE 处理](#rle-处理)
      - [无损性分析](#无损性分析)
    - [COCO ↔ LabelMe 转换](#coco--labelme-转换-1)
      - [转换步骤](#转换步骤-1)
      - [无损性分析](#无损性分析-1)
  - [六、文件示例](#六文件示例)
    - [完整的 COCO JSON 示例](#完整的-coco-json-示例)
    - [主要字段说明](#主要字段说明)
    - [数据结构关系](#数据结构关系)
  - [七、注意事项](#七注意事项)
  - [八、参考链接](#八参考链接)

---

## 一、概述
COCO（Common Objects in Context）是一个大规模目标检测、分割和字幕数据集。其标签格式采用 JSON 结构，支持丰富的标注信息，包括目标检测、实例分割和关键点检测。

## 二、文件构成和属性
COCO 标签文件是一个 JSON 文件，包含以下主要字段：

```json
{
  "info": {...},
  "licenses": [...],
  "images": [...],
  "annotations": [...],
  "categories": [...]
}
```

### images 数组
每个图像的信息：
```json
{
  "id": 1,
  "width": 640,
  "height": 480,
  "file_name": "image1.jpg"
}
```

### annotations 数组
每个标注实例的信息：
```json
{
  "id": 1,
  "image_id": 1,
  "category_id": 1,
  "bbox": [x, y, width, height],
  "area": 面积,
  "segmentation": 多边形或 RLE,
  "iscrowd": 0 或 1
}
```

### categories 数组
类别定义：
```json
{
  "id": 1,
  "name": "person",
  "supercategory": "human"
}
```

## 三、标注类型

### 目标检测（边界框）
COCO 使用 `bbox` 字段存储边界框信息，格式为 `[x_min, y_min, width, height]`，单位为像素。

**边界框示例**：
- 图像尺寸：640×480 像素
- 目标边界框：左上角 (100, 80)，宽度 200 像素，高度 240 像素
- COCO 格式：`"bbox": [100, 80, 200, 240]`
- 对应矩形区域：x范围 [100, 300]，y范围 [80, 320]

**边界框计算**：
- `x_min`：边界框左上角 x 坐标
- `y_min`：边界框左上角 y 坐标
- `width`：边界框宽度（像素）
- `height`：边界框高度（像素）
- `area`：边界框面积 = `width × height`

### 实例分割（多边形或 RLE）
COCO 使用 `segmentation` 字段存储分割信息，支持两种格式：

#### 多边形格式
- **单多边形**：`[[x1, y1, x2, y2, ...]]`（单个列表的列表）
- **多多边形**：`[[x1, y1, x2, y2, ...], [x1, y1, x2, y2, ...]]`（多个多边形列表）
- 每个 `(x, y)` 是多边形顶点的像素坐标
- 多边形通常为顺时针或逆时针顺序

**多边形示例**：
- 三角形：`"segmentation": [[100, 80, 200, 80, 150, 150]]`
- 矩形（4个点）：`"segmentation": [[100, 80, 300, 80, 300, 320, 100, 320]]`

#### RLE 格式
- **COCO RLE**：`{"size": [height, width], "counts": "压缩字符串"}`
- **二进制 RLE**：`{"counts": [游程长度数组], "size": [height, width]}`
- 用于 `iscrowd: 1` 的密集或重叠目标

**RLE 示例**：
```json
"segmentation": {
  "size": [480, 640],
  "counts": "`VOh1k>Z2F2N2...（压缩字符串）"
}
```

### RLE 编码详解

### RLE 编码原理
Run-Length Encoding（游程编码）是一种无损数据压缩算法，特别适合二值掩码。原理是将连续的相同值序列压缩为（值, 长度）对。

**二进制掩码的 RLE**：
- 掩码尺寸：`[height, width]`（如 480×640）
- 像素值：0（背景）或 1（前景）
- 编码方式：只记录前景像素（值=1）的游程
- 游程：连续前景像素的数量

**示例编码**：
假设一行像素：`[0,0,0,1,1,1,0,0,1,1,0,0]`
- 游程编码（从行首开始计数）：
  - 前3个背景像素：跳过
  - 3个前景像素：起始位置 4，长度 3
  - 2个背景像素：跳过
  - 2个前景像素：起始位置 9，长度 2
- COCO RLE 格式：将这些游程编码为紧凑字符串

### COCO RLE 格式
COCO 使用特殊的 RLE 格式：
- **size**：`[height, width]` 图像尺寸
- **counts**：游程编码字符串（使用字符编码）
- **编码特点**：按行主序（row-major）扫描图像

**counts 字符串示例**：
```
"`VOh1k>Z2F2N2O2N2N3N2N2N2..."
```
- 字符串使用 Base64 类似的字符编码
- 每个字符代表一个游程长度
- 需要专门的解码器（如 `pycocotools`）解析

### 二进制 RLE 格式
另一种更直观的 RLE 表示：
- **size**：`[height, width]` 图像尺寸
- **counts**：整数数组，交替表示前景和背景的像素数
- **数组格式**：`[背景长度, 前景长度, 背景长度, 前景长度, ...]`

**二进制 RLE 示例**：
```json
{
  "size": [5, 5],
  "counts": [5, 3, 2, 5, 3, 2, 5, 3, 2]
}
```
- 表示：5背景, 3前景, 2背景, 5前景, ...

### pycocotools 使用说明
`pycocotools` 是处理 COCO 格式的 Python 工具包，提供 RLE 编码解码功能。

**安装**：
```bash
pip install pycocotools
```

**主要功能**：
```python
from pycocotools import mask as maskUtils

# 1. 多边形转 RLE
polygon = [[100, 80, 200, 80, 150, 150]]  # 三角形
rle = maskUtils.frPyObjects(polygon, 480, 640)  # 图像高度, 宽度

# 2. RLE 解码为二值掩码
binary_mask = maskUtils.decode(rle)  # 返回 numpy 数组 (480, 640)

# 3. 掩码转 RLE
rle_from_mask = maskUtils.encode(np.asfortranarray(binary_mask))

# 4. RLE 转多边形
polygons = maskUtils.toBbox(rle)  # 获取边界框
segmentation = maskUtils.toPolygon(rle)  # 获取多边形

# 5. RLE 面积计算
area = maskUtils.area(rle)  # 前景像素数

# 6. RLE 合并/交集
rle_union = maskUtils.merge([rle1, rle2])
```

**RLE 与多边形的转换**：
- **多边形 → RLE**：`maskUtils.frPyObjects()` 将多边形转换为 RLE
- **RLE → 多边形**：`maskUtils.toPolygon()` 将 RLE 解码为多边形（可能损失精度）
- **精度问题**：RLE 是像素级表示，转换为多边形时可能因轮廓提取损失边界精度

### 应用场景
1. **密集目标**：`iscrowd: 1` 的标注通常使用 RLE
2. **重叠实例**：多个目标重叠时，RLE 可以精确表示每个实例
3. **存储效率**：对于大目标或简单形状，RLE 比多边形更节省空间
4. **计算效率**：RLE 支持高效的掩码操作（合并、交集、面积计算）

## 四、坐标系统
COCO 使用**绝对像素坐标系统**，所有坐标值都是像素单位。

### 边界框坐标
格式：`[x_min, y_min, width, height]`

**坐标定义**：
- `x_min`：边界框左上角 x 坐标（像素）
- `y_min`：边界框左上角 y 坐标（像素）
- `width`：边界框宽度（像素）
- `height`：边界框高度（像素）

**坐标范围**：
- x 范围：`[x_min, x_min + width]`
- y 范围：`[y_min, y_min + height]`

**示例**（图像尺寸 640×480）：
- 边界框：左上角 (100, 80)，宽度 200，高度 240
- COCO 格式：`[100, 80, 200, 240]`
- 对应矩形：x∈[100, 300]，y∈[80, 320]

### 多边形坐标
格式：`[[x1, y1, x2, y2, ...]]`

**坐标特点**：
- 每个 `(x, y)` 是多边形顶点的像素坐标
- 坐标顺序通常为顺时针或逆时针
- 多边形可以自相交，但通常应为简单多边形
- 支持多个多边形表示一个实例（如有孔洞）

**多边形示例**：
- 三角形：`[[100, 80, 200, 80, 150, 150]]`
- 矩形：`[[100, 80, 300, 80, 300, 320, 100, 320]]`

### RLE 坐标系统
RLE 基于像素网格：
- `size`：`[height, width]` 指定参考图像尺寸
- 坐标隐含在游程编码中
- 每个前景像素对应图像中的一个像素位置

### 坐标转换示例

#### COCO → YOLO 转换
**边界框转换公式**：
- $cx = \frac{x_{min} + width/2}{image\_width}$
- $cy = \frac{y_{min} + height/2}{image\_height}$
- $w = \frac{width}{image\_width}$
- $h = \frac{height}{image\_height}$

**数值示例**：
- COCO 边界框：`[100, 80, 200, 240]`
- 图像尺寸：640×480
- 计算：
  - cx = (100 + 200/2) / 640 = 200 / 640 = 0.3125
  - cy = (80 + 240/2) / 480 = 200 / 480 = 0.4167
  - w = 200 / 640 = 0.3125
  - h = 240 / 480 = 0.5
- YOLO 格式：`0 0.3125 0.4167 0.3125 0.5`

#### COCO → LabelMe 转换
多边形坐标可以直接使用，只需调整格式：
- COCO 多边形：`[[100, 80, 200, 80, 150, 150]]`
- LabelMe points：`[[100, 80], [200, 80], [150, 150]]`

## 五、与其他格式的转换

### COCO ↔ YOLO 转换

#### 转换步骤
1. **读取 COCO JSON**：解析 `images`、`annotations`、`categories`
2. **坐标转换**：将绝对像素坐标转换为归一化坐标（需要图像尺寸）
3. **类别映射**：将 `category_id` 映射到 YOLO 类别索引
4. **处理分割**：将多边形或 RLE 转换为 YOLO 多边形点序列
5. **生成 YOLO 文件**：为每个图像创建 `.txt` 文件

#### RLE 处理
- **RLE → 多边形**：需要将 RLE 解码为二值掩码，然后提取轮廓点
- **多边形简化**：提取的轮廓点可能过多，需要适当简化

#### 无损性分析
- **COCO → YOLO**：多边形可以无损转换；RLE 转换为多边形时，由于轮廓提取和简化可能损失精度
- **YOLO → COCO**：可以无损转换，但需生成适当的 COCO 结构

### COCO ↔ LabelMe 转换

#### 转换步骤
1. **读取 COCO JSON**：获取图像和标注信息
2. **坐标转换**：COCO 多边形直接可用，RLE 需解码为多边形
3. **构建 LabelMe JSON**：创建 `shapes` 数组，每个形状为 `polygon` 类型
4. **处理类别**：将 `category_id` 映射到标签名称

#### 无损性分析
- **COCO → LabelMe**：多边形可以无损转换；RLE 转换可能损失精度
- **LabelMe → COCO**：可以无损转换，但需处理 LabelMe 的多种形状类型

## 六、文件示例

### 完整的 COCO JSON 示例
```json
{
  "info": {
    "year": "2023",
    "version": "1.0",
    "description": "示例数据集"
  },
  "licenses": [{"id": 1, "name": "CC BY 4.0"}],
  "images": [
    {
      "id": 1,
      "width": 640,
      "height": 480,
      "file_name": "image1.jpg"
    },
    {
      "id": 2,
      "width": 800,
      "height": 600,
      "file_name": "image2.jpg"
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [100, 80, 200, 240],
      "area": 48000,
      "segmentation": [[100, 80, 300, 80, 300, 320, 100, 320]],
      "iscrowd": 0
    },
    {
      "id": 2,
      "image_id": 1,
      "category_id": 2,
      "bbox": [400, 200, 150, 180],
      "area": 27000,
      "segmentation": {
        "size": [480, 640],
        "counts": "`VOh1k>Z2F2N2O2N2N3N2N2N2M3O2N2N2N2N2"
      },
      "iscrowd": 1
    }
  ],
  "categories": [
    {
      "id": 1,
      "name": "person",
      "supercategory": "human"
    },
    {
      "id": 2,
      "name": "car",
      "supercategory": "vehicle"
    }
  ]
}
```

### 主要字段说明
1. **images 数组**：数据集中的所有图像
   - `id`：唯一图像标识符
   - `width`, `height`：图像尺寸（像素）
   - `file_name`：图像文件名（相对路径）

2. **annotations 数组**：所有标注实例
   - `id`：唯一标注标识符
   - `image_id`：对应的图像 ID
   - `category_id`：类别 ID
   - `bbox`：边界框 `[x_min, y_min, width, height]`
   - `area`：区域面积（像素数）
   - `segmentation`：分割信息（多边形或 RLE）
   - `iscrowd`：0=单个实例，1=密集/重叠实例

3. **categories 数组**：类别定义
   - `id`：唯一类别标识符
   - `name`：类别名称
   - `supercategory`：父类别（可选）

### 数据结构关系
- 一个 `image` 可以有多个 `annotation`
- 一个 `annotation` 属于一个 `image`（通过 `image_id`）
- 一个 `annotation` 属于一个 `category`（通过 `category_id`）
- 一个 `category` 可以有多个 `annotation`

## 七、注意事项
1. **iscrowd 标志**：`iscrowd=1` 表示密集或重叠目标，通常使用 RLE 编码
2. **面积字段**：`area` 可以是边界框面积或多边形/RLE 掩码的面积
3. **类别层次**：`supercategory` 支持类别层次结构
4. **RLE 依赖**：处理 RLE 需要专门的库（如 `pycocotools`）
5. **大规模数据**：COCO JSON 文件可能很大，需要流式处理或分片

## 八、参考链接
1. **COCO 数据集官网**：https://cocodataset.org/
2. **pycocotools GitHub**：https://github.com/cocodataset/cocoapi
3. **COCO 格式说明**：https://cocodataset.org/#format-data
4. **RLE 编码原理**：https://en.wikipedia.org/wiki/Run-length_encoding
5. **LabelMe 标注工具**：https://github.com/wkentaro/labelme
6. **YOLO 官方项目**：https://github.com/pjreddie/darknet