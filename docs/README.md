# 数据格式文档

本目录包含 DataFlow-CV 支持的数据格式的详细说明。

## 支持的格式

1. **[YOLO 格式](yolo.md)** – YOLO（You Only Look Once）目标检测与分割格式
   - 每图像一个 `.txt` 标签文件
   - 归一化坐标，独立的类别文件
   - 支持检测（边界框）和分割（多边形）两种标注类型

2. **[LabelMe 格式](labelme.md)** – LabelMe 标注工具格式
   - 每图像一个 `.json` 文件
   - 支持矩形（rectangle）和多边形（polygon）形状
   - 包含图像路径、尺寸、版本等元数据

3. **[COCO 格式](coco.md)** – COCO（Common Objects in Context）数据集格式
   - 单一 JSON 文件描述整个数据集
   - 结构化层次：`images`、`annotations`、`categories`
   - 包含边界框、多边形分割、面积、是否拥挤等丰富信息

## 使用说明

这些文档仅描述数据格式本身的规范，不涉及 DataFlow-CV 工具的具体使用方法。如需了解如何使用 DataFlow-CV 进行格式转换或可视化，请参阅项目根目录的 [README.md](../README.md)。

## 贡献

如果您发现文档中有任何错误或遗漏，欢迎提交 Issue 或 Pull Request。

## 参考

- [YOLO 官方网站](https://pjreddie.com/darknet/yolo/)
- [LabelMe 官方网站](http://labelme.csail.mit.edu/)
- [COCO 官方网站](http://cocodataset.org/)