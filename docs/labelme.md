


# LabelMe 数据格式

## 概述

LabelMe 是一个用于图像标注的在线工具，其数据格式采用 JSON 结构，每个图像对应一个独立的 JSON 文件。LabelMe 格式支持多种标注形状（如矩形、多边形等），并包含图像元数据和标注信息。

### 核心特点
- **每图一JSON**：每个图像（如 `image.jpg`）对应一个同名的 `.json` 文件（如 `image.json`）。
- **灵活的形状类型**：支持 `rectangle`（矩形）和 `polygon`（多边形）等多种形状。
- **自包含信息**：JSON 文件中包含图像路径、尺寸、版本以及所有标注形状的详细信息。
- **易于人工编辑**：JSON 格式可读性好，便于手动修改或查看。

## 文件结构

LabelMe JSON 文件是一个包含以下顶级字段的对象：

| 字段名 | 类型 | 描述 |
|--------|------|------|
| `version` | 字符串 | LabelMe 格式版本（例如 `"5.3.1"`） |
| `flags` | 对象 | 可选的标志字典，用于存储自定义标记 |
| `shapes` | 数组 | **核心字段**，包含所有标注形状的列表 |
| `imagePath` | 字符串 | 图像文件名（相对于 JSON 文件的路径） |
| `imageData` | 字符串（可选） | Base64 编码的图像数据（通常省略，仅存储路径） |
| `imageHeight` | 整数 | 图像的高度（像素） |
| `imageWidth` | 整数 | 图像的宽度（像素） |

### `shapes` 数组
每个元素是一个形状对象，包含以下字段：

| 字段名 | 类型 | 描述 |
|--------|------|------|
| `label` | 字符串 | 标注的类别名称（如 `"person"`, `"car"`） |
| `points` | 数组 | 形状的顶点坐标列表，格式为 `[[x1, y1], [x2, y2], ...]` |
| `group_id` | 整数或 `null` | 用于分组相关标注的可选标识符 |
| `shape_type` | 字符串 | 形状类型，常见值为 `"rectangle"` 或 `"polygon"` |
| `flags` | 对象 | 形状特有的标志字典（通常为空） |

## 形状类型详解

### 1. 矩形（`shape_type: "rectangle"`）
用于表示轴对齐的边界框（bounding box）。

- **`points` 格式**：两个点的列表 `[[x1, y1], [x2, y2]]`，分别表示矩形的左上角和右下角。
- **坐标顺序**：实际矩形由这两个点定义的**最小外接矩形**决定，即：
  ```
  x_min = min(x1, x2)
  y_min = min(y1, y2)
  x_max = max(x1, x2)
  y_max = max(y1, y2)
  ```

**示例**：
```json
{
  "label": "car",
  "points": [[100, 150], [300, 350]],
  "group_id": null,
  "shape_type": "rectangle",
  "flags": {}
}
```
表示一个左上角在 `(100, 150)`、右下角在 `(300, 350)` 的矩形框。

### 2. 多边形（`shape_type: "polygon"`）
用于表示任意形状的多边形区域（实例分割）。

- **`points` 格式**：多个点的列表 `[[x1, y1], [x2, y2], ...]`，按顺序连接形成闭合多边形。
- **点数要求**：至少需要 **3** 个点才能构成有效的多边形。
- **闭合性**：多边形自动闭合（最后一个点连接回第一个点）。

**示例**：
```json
{
  "label": "person",
  "points": [[200, 100], [300, 150], [250, 300], [150, 250]],
  "group_id": null,
  "shape_type": "polygon",
  "flags": {}
}
```
表示一个由四个顶点组成的多边形区域。

## 示例文件

### 完整示例（`example.json`）
```json
{
  "version": "5.3.1",
  "flags": {},
  "shapes": [
    {
      "label": "person",
      "points": [[120, 80], [350, 80], [350, 400], [120, 400]],
      "group_id": null,
      "shape_type": "rectangle",
      "flags": {}
    },
    {
      "label": "car",
      "points": [[400, 200], [550, 200], [550, 300], [400, 300]],
      "group_id": null,
      "shape_type": "polygon",
      "flags": {}
    }
  ],
  "imagePath": "example.jpg",
  "imageData": null,
  "imageHeight": 480,
  "imageWidth": 640
}
```

### 解释
- **图像信息**：图像文件名为 `example.jpg`，尺寸为 640×480 像素。
- **第一个标注**：一个矩形框，类别为 `"person"`，左上角 `(120, 80)`，右下角 `(350, 400)`。
- **第二个标注**：一个多边形，类别为 `"car"`，四个顶点分别为 `(400,200)`, `(550,200)`, `(550,300)`, `(400,300)`。

## 坐标系统

- **原点**：图像左上角为 `(0, 0)`。
- **X 轴**：向右为正方向。
- **Y 轴**：向下为正方向（与计算机图像坐标系一致）。
- **单位**：像素（整数或浮点数）。

## 注意事项

1. **图像路径**：`imagePath` 可以是相对路径（相对于 JSON 文件的位置）或绝对路径。通常推荐使用相对路径以便于移动整个数据集。
2. **图像数据**：`imageData` 字段通常为 `null`，因为将图像编码为 Base64 会显著增加文件大小。建议仅存储路径，通过 `imagePath` 加载图像。
3. **形状类型**：除了 `rectangle` 和 `polygon`，LabelMe 还支持 `circle`、`line`、`point` 等形状，但矩形和多边形是最常用的两种。
4. **坐标范围**：坐标值应在图像尺寸范围内（`0 ≤ x < imageWidth`, `0 ≤ y < imageHeight`），超出范围的坐标可能被截断。
5. **版本兼容性**：不同版本的 LabelMe 可能略有差异，但 `version` 字段有助于解析器进行兼容性处理。

## 参考

- LabelMe 官方网站：[http://labelme.csail.mit.edu/](http://labelme.csail.mit.edu/)
- LabelMe GitHub 仓库：[https://github.com/wkentaro/labelme](https://github.com/wkentaro/labelme)