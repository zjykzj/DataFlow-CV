# LabelMe 标签格式

## 目录
- [LabelMe 标签格式](#labelme-标签格式)
  - [目录](#目录)
  - [一、概述](#一概述)
  - [二、文件构成和属性](#二文件构成和属性)
    - [shapes 数组](#shapes-数组)
  - [三、标注类型](#三标注类型)
    - [目标检测](#目标检测)
      - [矩形（rectangle）](#矩形rectangle)
      - [多边形（polygon）](#多边形polygon)
      - [圆形（circle）](#圆形circle)
      - [线段（line）](#线段line)
      - [点（point）](#点point)
    - [实例分割](#实例分割)
      - [多边形分割](#多边形分割)
      - [分组支持](#分组支持)
  - [四、坐标系统](#四坐标系统)
    - [坐标系定义](#坐标系定义)
    - [形状坐标格式](#形状坐标格式)
      - [矩形（rectangle）](#矩形rectangle-1)
      - [多边形（polygon）](#多边形polygon-1)
      - [圆形（circle）](#圆形circle-1)
    - [坐标转换示例](#坐标转换示例)
      - [LabelMe → YOLO 转换](#labelme--yolo-转换)
      - [LabelMe → COCO 转换](#labelme--coco-转换)
  - [五、与其他格式的转换](#五与其他格式的转换)
    - [LabelMe ↔ YOLO 转换](#labelme--yolo-转换-1)
      - [转换步骤](#转换步骤)
      - [形状转换处理](#形状转换处理)
      - [无损性分析](#无损性分析)
    - [LabelMe ↔ COCO 转换](#labelme--coco-转换-1)
      - [转换步骤](#转换步骤-1)
      - [RLE 生成](#rle-生成)
      - [无损性分析](#无损性分析-1)
  - [六、文件示例](#六文件示例)
    - [完整的 LabelMe JSON 示例](#完整的-labelme-json-示例)
    - [字段详细说明](#字段详细说明)
    - [形状类型示例汇总](#形状类型示例汇总)
    - [文件结构](#文件结构)
  - [七、注意事项](#七注意事项)
  - [八、参考链接](#八参考链接)

---

## 一、概述
LabelMe 是一个在线图像标注工具，其标签格式采用 JSON 结构，支持多种形状类型和丰富的属性。LabelMe 格式灵活，常用于学术研究和原型开发。

## 二、文件构成和属性
LabelMe 标签文件是一个 JSON 文件，每个图像对应一个 JSON 文件。基本结构如下：

```json
{
  "version": "5.1.1",
  "flags": {},
  "shapes": [...],
  "imagePath": "image.jpg",
  "imageData": "base64编码的图像数据（可选）",
  "imageHeight": 480,
  "imageWidth": 640
}
```

### shapes 数组
每个形状的标注信息：
```json
{
  "label": "person",
  "points": [[x1, y1], [x2, y2], ...],
  "group_id": null,
  "shape_type": "polygon",
  "flags": {}
}
```

- **label**：类别标签（字符串）
- **points**：形状的顶点坐标数组
- **group_id**：关联形状的分组 ID（用于实例分割）
- **shape_type**：形状类型（`polygon`、`rectangle`、`circle`、`line`、`point`）
- **flags**：自定义属性键值对

## 三、标注类型

### 目标检测
LabelMe 支持多种形状类型用于目标检测：

#### 矩形（rectangle）
- **points 格式**：`[[x_min, y_min], [x_max, y_max]]`
- **描述**：两个点定义矩形的左上角和右下角
- **示例**：`"points": [[100, 80], [300, 320]]`
  - 表示矩形：x∈[100, 300]，y∈[80, 320]

#### 多边形（polygon）
- **points 格式**：`[[x1, y1], [x2, y2], ...]`
- **描述**：多个顶点定义多边形轮廓
- **示例**：`"points": [[100, 80], [200, 80], [150, 150]]`
  - 表示三角形

#### 圆形（circle）
- **points 格式**：`[[center_x, center_y], [point_on_circle_x, point_on_circle_y]]`
- **描述**：圆心和圆周上的一个点定义圆形
- **半径计算**：`radius = sqrt((x2-x1)² + (y2-y1)²)`
- **示例**：`"points": [[250, 200], [300, 200]]`
  - 圆心：(250, 200)，半径：50

#### 线段（line）
- **points 格式**：`[[x1, y1], [x2, y2]]`
- **描述**：两个点定义线段
- **示例**：`"points": [[100, 100], [200, 200]]`

#### 点（point）
- **points 格式**：`[[x, y]]`
- **描述**：单个点
- **示例**：`"points": [[150, 120]]`

### 实例分割
LabelMe 主要使用多边形进行实例分割：

#### 多边形分割
- **shape_type**：`"polygon"`
- **points**：精确描述目标轮廓的多边形顶点
- **复杂形状**：可以通过多个多边形表示（如有孔洞）

#### 分组支持
- **group_id**：相同整数值表示属于同一实例
- **应用场景**：
  - 一个物体被遮挡成多个部分
  - 复杂形状需要多个多边形描述
  - 实例的不同组件

**分组示例**：
- 两个多边形有相同的 `group_id: 1`，表示属于同一实例
- 用于处理遮挡或复杂形状的分割

## 四、坐标系统
LabelMe 使用**绝对像素坐标系统**，所有坐标值都是像素单位。

### 坐标系定义
- **原点**：图像左上角 `(0, 0)`
- **x 轴**：向右为正方向
- **y 轴**：向下为正方向（与图像坐标系一致）

### 形状坐标格式

#### 矩形（rectangle）
- **points**：`[[x_min, y_min], [x_max, y_max]]`
- **坐标范围**：x∈[x_min, x_max]，y∈[y_min, y_max]
- **示例**：`[[100, 80], [300, 320]]`
  - x_min=100, y_min=80, x_max=300, y_max=320
  - 宽度：200 像素，高度：240 像素

#### 多边形（polygon）
- **points**：`[[x1, y1], [x2, y2], ...]`
- 每个 `[x, y]` 是多边形顶点坐标
- **示例**：`[[100, 80], [200, 80], [150, 150]]`
  - 三角形三个顶点

#### 圆形（circle）
- **points**：`[[center_x, center_y], [point_on_circle_x, point_on_circle_y]]`
- **圆心**：`(center_x, center_y)`
- **半径**：$r = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2}$
- **示例**：`[[250, 200], [300, 200]]`
  - 圆心：(250, 200)，半径：50 像素

### 坐标转换示例

#### LabelMe → YOLO 转换
**矩形转换公式**：
1. 计算边界框：
   - $width = x_{max} - x_{min}$
   - $height = y_{max} - y_{min}$
2. 计算归一化坐标（需要 $imageWidth$, $imageHeight$）：
   - $cx = \frac{x_{min} + width/2}{imageWidth}$
   - $cy = \frac{y_{min} + height/2}{imageHeight}$
   - $w = \frac{width}{imageWidth}$
   - $h = \frac{height}{imageHeight}$

**数值示例**：
- LabelMe 矩形：`[[100, 80], [300, 320]]`
- 图像尺寸：640×480
- 计算：
  - width = 300-100 = 200, height = 320-80 = 240
  - cx = (100 + 200/2) / 640 = 200 / 640 = 0.3125
  - cy = (80 + 240/2) / 480 = 200 / 480 = 0.4167
  - w = 200 / 640 = 0.3125
  - h = 240 / 480 = 0.5
- YOLO 格式：`0 0.3125 0.4167 0.3125 0.5`

**多边形转换**：
- 每个顶点坐标归一化：$x_{norm} = \frac{x}{imageWidth}$, $y_{norm} = \frac{y}{imageHeight}$
- 示例：顶点 (100, 80) → (0.15625, 0.16667)

#### LabelMe → COCO 转换
**矩形转换**：
- COCO 边界框：`[x_min, y_min, width, height]`
- 示例：`[[100, 80], [300, 320]]` → `[100, 80, 200, 240]`

**多边形转换**：
- LabelMe 格式：`[[x1, y1], [x2, y2], ...]`
- COCO 格式：`[[x1, y1, x2, y2, ...]]`（展平）
- 示例：`[[100, 80], [200, 80], [150, 150]]` → `[[100, 80, 200, 80, 150, 150]]`

## 五、与其他格式的转换

### LabelMe ↔ YOLO 转换

#### 转换步骤
1. **读取 LabelMe JSON**：解析 `shapes` 数组，获取每个形状的 `label`、`points` 和 `shape_type`
2. **形状统一化**：将矩形、圆形等转换为多边形表示（需要采样或近似）
3. **坐标转换**：将绝对像素坐标转换为归一化坐标（需要 `imageWidth` 和 `imageHeight`）
4. **类别映射**：将字符串标签映射到 YOLO 类别索引
5. **生成 YOLO 文件**：创建 `.txt` 文件，每行包含边界框和多边形数据

#### 形状转换处理
- **矩形 → 多边形**：四个顶点 `[左上, 右上, 右下, 左下]`
- **圆形 → 多边形**：采样圆周上的点（如 36 个点）
- **多边形**：直接使用，但可能需要简化点数

#### 无损性分析
- **LabelMe → YOLO**：矩形和圆形转换为多边形时会引入近似误差；多边形可以无损转换（坐标精度限制内）
- **YOLO → LabelMe**：可以无损转换，但 YOLO 只支持多边形，需设置 `shape_type: "polygon"`

### LabelMe ↔ COCO 转换

#### 转换步骤
1. **读取 LabelMe JSON**：获取所有形状标注
2. **坐标处理**：LabelMe 多边形坐标可直接用于 COCO
3. **形状转换**：将矩形、圆形等转换为多边形
4. **构建 COCO 结构**：创建 `images`、`annotations`、`categories` 数组
5. **RLE 生成**：（可选）将多边形转换为 RLE 编码

#### RLE 生成
- **多边形 → RLE**：需要将多边形渲染为二值掩码，然后进行 RLE 编码
- **精度考虑**：渲染过程可能因像素化损失边界精度

#### 无损性分析
- **LabelMe → COCO**：多边形可以无损转换；矩形和圆形转换可能损失精度；RLE 生成会引入像素化误差
- **COCO → LabelMe**：多边形可以无损转换；RLE 转换为多边形时可能损失精度

## 六、文件示例

### 完整的 LabelMe JSON 示例
```json
{
  "version": "5.1.1",
  "flags": {},
  "shapes": [
    {
      "label": "person",
      "points": [[100, 80], [300, 80], [300, 320], [100, 320]],
      "group_id": null,
      "shape_type": "polygon",
      "flags": {}
    },
    {
      "label": "car",
      "points": [[400, 200], [550, 200], [550, 380], [400, 380]],
      "group_id": null,
      "shape_type": "rectangle",
      "flags": {}
    },
    {
      "label": "ball",
      "points": [[250, 200], [300, 200]],
      "group_id": null,
      "shape_type": "circle",
      "flags": {}
    },
    {
      "label": "road",
      "points": [[100, 100], [200, 200]],
      "group_id": null,
      "shape_type": "line",
      "flags": {}
    }
  ],
  "imagePath": "image1.jpg",
  "imageData": null,
  "imageHeight": 480,
  "imageWidth": 640
}
```

### 字段详细说明
1. **version**：LabelMe 格式版本（如 "5.1.1"）
2. **flags**：全局标志（自定义键值对）
3. **shapes**：标注形状数组，每个元素包含：
   - `label`：类别标签（字符串）
   - `points`：形状坐标点数组
   - `group_id`：分组标识（相同值表示同一实例）
   - `shape_type`：形状类型（`polygon`、`rectangle`、`circle`、`line`、`point`）
   - `flags`：形状特定标志
4. **imagePath**：图像文件路径（相对或绝对）
5. **imageData**：Base64 编码的图像数据（可选，通常为 `null`）
6. **imageHeight**：图像高度（像素）
7. **imageWidth**：图像宽度（像素）

### 形状类型示例汇总
| 形状类型 | points 格式 | 示例 |
|---------|------------|------|
| 矩形 | `[[x_min,y_min], [x_max,y_max]]` | `[[100,80], [300,320]]` |
| 多边形 | `[[x1,y1], [x2,y2], ...]` | `[[100,80], [200,80], [150,150]]` |
| 圆形 | `[[cx,cy], [px,py]]` | `[[250,200], [300,200]]`（半径50） |
| 线段 | `[[x1,y1], [x2,y2]]` | `[[100,100], [200,200]]` |
| 点 | `[[x,y]]` | `[[150,120]]` |

### 文件结构
```
dataset/
├── images/
│   ├── image1.jpg
│   └── image2.jpg
└── annotations/
    ├── image1.json
    └── image2.json
```

**说明**：
- 每个图像对应一个 JSON 文件（同名）
- JSON 文件包含该图像的所有标注
- 图像和标注文件通常分开存储

## 七、注意事项
1. **形状多样性**：LabelMe 支持多种形状类型，转换时需要统一处理
2. **图像路径**：`imagePath` 可以是相对路径或绝对路径，需注意路径解析
3. **Base64 图像**：`imageData` 字段包含 Base64 编码的图像数据，但通常建议分开存储图像文件
4. **分组支持**：`group_id` 可用于复杂实例的多个部分标注
5. **版本兼容性**：`version` 字段表示格式版本，不同版本可能有细微差异
6. **标签灵活性**：`label` 字段支持任意字符串，便于快速原型开发

## 八、参考链接
1. **LabelMe 官方 GitHub**：https://github.com/wkentaro/labelme
2. **LabelMe 在线标注工具**：http://labelme.csail.mit.edu/
3. **COCO 数据集官网**：https://cocodataset.org/
4. **YOLO 官方项目**：https://github.com/pjreddie/darknet
5. **Ultralytics YOLO**：https://github.com/ultralytics/yolov5
6. **pycocotools GitHub**：https://github.com/cocodataset/cocoapi