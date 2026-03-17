

# COCO 数据格式

## 概述

COCO（Common Objects in Context）是一个大规模的目标检测、分割和字幕数据集。其数据格式采用单一的 JSON 文件来描述整个数据集，包含图像信息、标注信息、类别信息等。COCO 格式已成为计算机视觉领域广泛使用的标准之一。

### 核心特点
- **单一文件**：整个数据集的标注信息存储在一个 JSON 文件中。
- **结构化层次**：包含 `images`、`annotations`、`categories` 等顶层字段，结构清晰。
- **丰富的信息**：除了边界框和多边形分割，还包含面积、是否拥挤（iscrowd）等元数据。
- **官方标准**：由 COCO 数据集官方定义，被众多研究和工程项目采用。

## 文件结构

COCO JSON 文件是一个包含以下顶级字段的对象：

| 字段名 | 类型 | 描述 |
|--------|------|------|
| `info` | 对象 | 数据集的总体信息（可选） |
| `licenses` | 数组 | 许可证信息列表（可选） |
| `images` | 数组 | **必需**，图像信息列表 |
| `annotations` | 数组 | **必需**，标注信息列表 |
| `categories` | 数组 | **必需**，类别信息列表 |

### 1. `info` 对象（可选）
包含数据集的描述性信息。

**示例**：
```json
"info": {
  "description": "COCO 2017 Dataset",
  "url": "http://cocodataset.org",
  "version": "1.0",
  "year": 2017,
  "contributor": "COCO Consortium",
  "date_created": "2017/09/01"
}
```

### 2. `licenses` 数组（可选）
列出数据集中图像使用的许可证。

**示例**：
```json
"licenses": [
  {
    "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/",
    "id": 1,
    "name": "Attribution-NonCommercial-ShareAlike License"
  }
]
```

### 3. `images` 数组（必需）
每个元素描述数据集中的一张图像。

| 字段名 | 类型 | 描述 |
|--------|------|------|
| `id` | 整数 | **唯一** 图像标识符 |
| `file_name` | 字符串 | 图像文件名（可包含相对路径） |
| `height` | 整数 | 图像高度（像素） |
| `width` | 整数 | 图像宽度（像素） |
| `license` | 整数（可选） | 许可证 ID（指向 `licenses` 数组） |
| `flickr_url` | 字符串（可选） | Flickr 图像 URL |
| `coco_url` | 字符串（可选） | COCO 图像 URL |
| `date_captured` | 字符串（可选） | 拍摄日期 |

**示例**：
```json
{
  "id": 1,
  "file_name": "000000001.jpg",
  "height": 480,
  "width": 640,
  "license": 1,
  "flickr_url": "http://farm1.staticflickr.com/1/000000001.jpg",
  "coco_url": "http://images.cocodataset.org/train2017/000000001.jpg",
  "date_captured": "2013-11-14 11:18:45"
}
```

### 4. `annotations` 数组（必需）
每个元素描述一个标注实例（一个目标）。

| 字段名 | 类型 | 描述 |
|--------|------|------|
| `id` | 整数 | **唯一** 标注标识符 |
| `image_id` | 整数 | 对应图像的 ID（与 `images` 中的 `id` 匹配） |
| `category_id` | 整数 | 类别 ID（与 `categories` 中的 `id` 匹配） |
| `bbox` | 数组 `[x, y, width, height]` | 边界框坐标（检测任务） |
| `segmentation` | 数组 或 RLE 字典 | 分割标注（分割任务） |
| `area` | 浮点数 | 标注区域面积（像素²） |
| `iscrowd` | 整数（0 或 1） | 是否表示拥挤区域（crowd） |

**`bbox` 格式**：
- `[x, y, width, height]`
- `x`, `y` 为边界框左上角坐标（像素）。
- `width`, `height` 为边界框的宽度和高度（像素）。

**`segmentation` 格式**：
- **多边形格式**：`[[x1, y1, x2, y2, ...]]`，单个多边形用一维数组表示，多个多边形用二维数组。
- **RLE 格式**：`{"counts": [ ... ], "size": [height, width]}`，用于拥挤区域（`iscrowd=1`）。

**示例**：
```json
{
  "id": 1,
  "image_id": 1,
  "category_id": 1,
  "bbox": [100, 150, 200, 300],
  "segmentation": [[100, 150, 300, 150, 300, 450, 100, 450]],
  "area": 60000.0,
  "iscrowd": 0
}
```

### 5. `categories` 数组（必需）
定义数据集中的类别。

| 字段名 | 类型 | 描述 |
|--------|------|------|
| `id` | 整数 | **唯一** 类别标识符 |
| `name` | 字符串 | 类别名称（如 `"person"`, `"car"`） |
| `supercategory` | 字符串 | 父类别（如 `"vehicle"`, `"animal"`） |

**示例**：
```json
{
  "id": 1,
  "name": "person",
  "supercategory": "human"
}
```

## 完整示例

以下是一个简化的 COCO JSON 文件示例，包含 1 张图像、2 个标注和 2 个类别：

```json
{
  "info": {
    "description": "Example COCO dataset",
    "url": "",
    "version": "1.0",
    "year": 2026,
    "contributor": "",
    "date_created": "2026/03/09"
  },
  "licenses": [
    {
      "url": "",
      "id": 0,
      "name": "Unknown"
    }
  ],
  "images": [
    {
      "id": 1,
      "file_name": "example.jpg",
      "height": 480,
      "width": 640
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [100, 150, 200, 300],
      "segmentation": [[100, 150, 300, 150, 300, 450, 100, 450]],
      "area": 60000.0,
      "iscrowd": 0
    },
    {
      "id": 2,
      "image_id": 1,
      "category_id": 2,
      "bbox": [400, 200, 150, 100],
      "segmentation": [[400, 200, 550, 200, 550, 300, 400, 300]],
      "area": 15000.0,
      "iscrowd": 0
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

## 坐标系统

- **原点**：图像左上角为 `(0, 0)`。
- **X 轴**：向右为正方向。
- **Y 轴**：向下为正方向（与计算机图像坐标系一致）。
- **单位**：像素（整数）。

## 注意事项

1. **ID 唯一性**：`images.id`、`annotations.id`、`categories.id` 必须在各自范围内唯一。
2. **关联关系**：`annotations.image_id` 必须指向有效的 `images.id`；`annotations.category_id` 必须指向有效的 `categories.id`。
3. **分割格式**：
   - 多边形坐标按顺序列出，形成闭合区域（无需重复第一个点）。
   - 对于单个实例的多个不连通部分，可以使用多个多边形（二维数组）。
   - `iscrowd=1` 时，`segmentation` 通常使用 RLE（Run-Length Encoding）格式。
4. **面积计算**：`area` 字段用于评估指标（如 mAP），对于多边形通常是其像素面积，对于边界框是 `width * height`。
5. **拥挤标注**：`iscrowd=1` 表示该标注是一个拥挤区域（多个实例被标注为一个整体），在评估时通常被特殊处理。

## RLE 掩码与多边形格式

COCO 格式支持两种分割标注表示方式：多边形点列表（polygon point lists）和 RLE（Run-Length Encoding，游程编码）掩码。两者各有优劣，适用于不同场景。

### RLE 掩码格式

RLE 是一种高效的二值掩码编码方式，特别适合表示复杂、不规则的形状。

**数据结构**：
```json
{
  "counts": [192, 5, 10, 5, 25, 3, ...],
  "size": [height, width]
}
```

- `counts`：游程编码的字节数组，表示掩码中连续0和1的长度交替。
- `size`：掩码的尺寸 `[高度, 宽度]`。

**特点**：
- **紧凑存储**：对于大面积连续区域，RLE 可以大幅减小存储空间。
- **快速处理**：某些操作（如并集、交集）在 RLE 格式下更高效。
- **适合复杂形状**：对于细节丰富的分割掩码，RLE 比多边形点列表更精确。
- **`iscrowd=1` 时的标准格式**：COCO 数据集中拥挤区域通常使用 RLE。

### 多边形点列表格式

多边形点列表通过一系列顶点坐标描述分割区域的轮廓。

**数据结构**：
```json
[[x1, y1, x2, y2, x3, y3, ...]]
```

- 单个多边形：一维数组 `[x1, y1, x2, y2, ...]`
- 多个多边形（同一实例的不连通部分）：二维数组 `[[poly1], [poly2], ...]`

**特点**：
- **人类可读**：坐标点直观易懂，便于调试和验证。
- **精确描述简单形状**：对于规则形状（矩形、简单多边形），点列表更紧凑。
- **编辑友好**：可以手动调整个别顶点。
- **`iscrowd=0` 时的常用格式**：单个实例的标注通常使用多边形。

### 对比

| 特性 | RLE 掩码 | 多边形点列表 |
|------|----------|--------------|
| **存储效率** | 高（复杂形状） | 低（简单形状）或高（复杂形状） |
| **计算效率** | 掩码操作快 | 几何计算快 |
| **精度** | 像素级精度 | 受顶点数量限制 |
| **可读性** | 低（需解码） | 高（直接坐标） |
| **适用场景** | 复杂形状、拥挤区域 | 简单形状、规则多边形 |
| **COCO 标准** | `iscrowd=1` 时推荐 | `iscrowd=0` 时推荐 |

### 使用 pycocotools 进行转换

`pycocotools`（COCO API 的 Python 实现）提供了 RLE 与多边形之间的转换功能。

**安装**：
```bash
pip install pycocotools
```

**RLE → 多边形**：
```python
from pycocotools import mask as mask_utils
import numpy as np

# 假设 rle 是一个 RLE 字典
rle = {'counts': [192, 5, 10, ...], 'size': [480, 640]}

# 解码为二值掩码
binary_mask = mask_utils.decode(rle)

# 从掩码提取多边形（使用 OpenCV 或其他库）
# pycocotools 本身不直接提供掩码→多边形的转换，
# 但可以通过 findContours（OpenCV）实现
```

**多边形 → RLE**：
```python
from pycocotools import mask as mask_utils

# 多边形坐标列表（假设为单个多边形）
polygon = [[x1, y1, x2, y2, x3, y3, ...]]

# 图像尺寸
height, width = 480, 640

# 将多边形编码为 RLE
rle = mask_utils.frPyObjects(polygon, height, width)

# 如果需要合并多个多边形（同一实例）
rles = [mask_utils.frPyObjects(p, height, width) for p in polygons]
merged_rle = mask_utils.merge(rles)
```

**注意事项**：
1. **精度损失**：多边形→RLE→多边形 转换可能导致精度损失，因为 RLE 是像素级表示，而多边形是顶点近似。
2. **图像尺寸必需**：RLE 编码需要图像高度和宽度。
3. **多个多边形**：一个实例可能有多个不连通部分，需要分别编码后合并。

### 在 DataFlow-CV 中的支持

DataFlow-CV 通过 `--rle` 标志支持 RLE 格式的转换：

```bash
# 将 YOLO 分割转换为 COCO 格式，并使用 RLE 编码
dataflow convert yolo2coco images/ labels/ classes.names output.json --rle

# 将 LabelMe 分割转换为 COCO 格式，并使用 RLE 编码
dataflow convert labelme2coco labels/ classes.names output.json --rle
```

**Python API**：
```python
import dataflow

# YOLO → COCO 带 RLE
result = dataflow.yolo_to_coco("images/", "labels/", "classes.names", "output.json", rle=True)

# LabelMe → COCO 带 RLE
result = dataflow.labelme_to_coco("labels/", "classes.names", "output.json", rle=True)
```

**内部实现**：
- DataFlow-CV 使用 `pycocotools.mask.frPyObjects()` 将多边形转换为 RLE。
- 从 COCO 到 YOLO/LabelMe 的转换自动解码 RLE 为多边形（无需额外标志）。
- RLE 转换需要 `pycocotools>=2.0.0` 依赖。

## 参考

- COCO 官方网站：[http://cocodataset.org/](http://cocodataset.org/)
- COCO 格式详解：[https://cocodataset.org/#format-data](https://cocodataset.org/#format-data)
- COCO API（Python）：[https://github.com/cocodataset/cocoapi](https://github.com/cocodataset/cocoapi)
- pycocotools 文档：[https://github.com/cocodataset/cocoapi/tree/master/PythonAPI/pycocotools](https://github.com/cocodataset/cocoapi/tree/master/PythonAPI/pycocotools)