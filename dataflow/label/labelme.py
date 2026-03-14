# -*- coding: utf-8 -*-
"""
@Time    : 2026/3/9 20:43
@File    : labelme.py
@Author  : zj
@Description: LabelMe格式标签处理器

LabelMe格式描述:
LabelMe格式是JSON格式，每张图片对应一个JSON文件，包含：
- `version`: 版本号
- `flags`: 标志
- `shapes`: 标注形状列表，每个形状包含：
  - `label`: 类别标签
  - `points`: 多边形点坐标列表
  - `group_id`: 分组ID
  - `shape_type`: 形状类型 ("polygon", "rectangle")
  - `flags`: 形状标志
- `imagePath`: 图像文件名
- `imageData`: base64编码的图像数据（可选）
- `imageHeight`: 图像高度
- `imageWidth`: 图像宽度
"""

import os
import json
import glob
import logging
from typing import Dict, List, Tuple, Union, Optional


class LabelMeHandler:
    """LabelMe格式标签处理器

    支持目标检测（bbox）和实例分割（polygon）标注。
    提供单个文件和批量文件的读写功能。
    """

    def __init__(self, verbose: bool = False):
        """初始化LabelMe处理器

        Args:
            verbose: 是否输出详细信息
        """
        self.verbose = verbose
        self.logger = self._setup_logger()

    def _setup_logger(self):
        """设置日志记录器"""
        logger = logging.getLogger(self.__class__.__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        logger.setLevel(logging.INFO if self.verbose else logging.WARNING)
        return logger

    def read(self, json_path: str, require_segmentation: bool = False) -> Dict:
        """读取单个LabelMe JSON文件

        Args:
            json_path: LabelMe JSON文件路径
            require_segmentation: 是否要求分割格式。如果True，只接受shape_type="polygon"的标注

        Returns:
            统一格式的图像标注数据字典，结构如下：
            {
                "image_id": str,          # 图像ID（文件名）
                "image_path": str,        # 图像路径
                "width": int,             # 图像宽度
                "height": int,            # 图像高度
                "annotations": List[dict] # 标注列表，每个标注包含：
                                          #   "bbox": [x_min, y_min, width, height]
                                          #   "segmentation": [[x1, y1, x2, y2, ...]]
                                          #   "category_id": int
                                          #   "category_name": str
            }

        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 文件格式错误
        """
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"LabelMe文件不存在: {json_path}")

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                labelme_data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON解析错误: {e}")

        # 验证必需字段
        required_fields = ['shapes', 'imagePath', 'imageHeight', 'imageWidth']
        for field in required_fields:
            if field not in labelme_data:
                raise ValueError(f"缺少必需字段: {field}")

        # 构建图像标注数据
        image_id = os.path.splitext(os.path.basename(labelme_data['imagePath']))[0]
        image_path = os.path.join(os.path.dirname(json_path), labelme_data['imagePath'])

        image_annotations = {
            "image_id": image_id,
            "image_path": image_path,
            "width": labelme_data['imageWidth'],
            "height": labelme_data['imageHeight'],
            "annotations": []
        }

        # 解析所有shapes
        for shape in labelme_data['shapes']:
            annotation = self._parse_shape(shape, (labelme_data['imageWidth'], labelme_data['imageHeight']), require_segmentation)
            if annotation:
                image_annotations["annotations"].append(annotation)

        if self.verbose:
            self.logger.info(f"读取LabelMe文件: {json_path}")
            self.logger.info(f"  图像: {image_annotations['image_path']}")
            self.logger.info(f"  标注数量: {len(image_annotations['annotations'])}")

        return image_annotations

    def read_batch(self, json_dir: str, pattern: str = "*.json", require_segmentation: bool = False) -> List[Dict]:
        """批量读取LabelMe JSON文件目录

        Args:
            json_dir: 包含LabelMe JSON文件的目录
            pattern: 文件匹配模式
            require_segmentation: 是否要求分割格式。如果True，只接受shape_type="polygon"的标注

        Returns:
            图像标注数据列表，每个元素为read()返回的格式

        Raises:
            FileNotFoundError: 目录不存在
        """
        if not os.path.exists(json_dir):
            raise FileNotFoundError(f"目录不存在: {json_dir}")

        json_files = glob.glob(os.path.join(json_dir, pattern))
        if not json_files:
            if self.verbose:
                self.logger.info(f"警告: 目录中没有找到JSON文件: {json_dir}")
            return []

        results = []
        for json_file in json_files:
            try:
                image_annotations = self.read(json_file, require_segmentation)
                results.append(image_annotations)
            except FileNotFoundError as e:
                if self.verbose:
                    self.logger.info(f"警告: 跳过文件 {json_file}, 错误: {e}")
            # ValueError from read() indicates invalid JSON or format, propagate it

        if self.verbose:
            self.logger.info(f"批量读取完成: 目录 {json_dir}")
            self.logger.info(f"  成功读取文件数: {len(results)}")
            self.logger.info(f"  总标注数量: {sum(len(r['annotations']) for r in results)}")

        return results

    def write(self, data: Dict, output_path: str) -> bool:
        """写入单个LabelMe JSON文件

        Args:
            data: 统一格式的图像标注数据字典（与read()返回格式相同）
            output_path: 输出JSON文件路径

        Returns:
            是否写入成功

        Raises:
            ValueError: 数据格式错误
        """
        # 验证必需字段
        required_fields = ['image_id', 'image_path', 'width', 'height', 'annotations']
        for field in required_fields:
            if field not in data:
                raise ValueError(f"缺少必需字段: {field}")

        # 构建LabelMe格式数据
        labelme_data = {
            "version": "5.3.1",
            "flags": {},
            "shapes": [],
            "imagePath": os.path.basename(data['image_path']),
            "imageData": None,  # 默认不包含图像数据
            "imageHeight": data['height'],
            "imageWidth": data['width']
        }

        # 转换所有标注
        for annotation in data['annotations']:
            shape = self._create_shape(annotation, (data['width'], data['height']))
            if shape:
                labelme_data["shapes"].append(shape)

        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(labelme_data, f, indent=2, ensure_ascii=False)

            if self.verbose:
                self.logger.info(f"写入LabelMe文件: {output_path}")
                self.logger.info(f"  图像: {data['image_path']}")
                self.logger.info(f"  标注数量: {len(labelme_data['shapes'])}")

            return True
        except Exception as e:
            if self.verbose:
                self.logger.info(f"写入文件失败: {output_path}, 错误: {e}")
            return False

    def write_batch(self, data_list: List[Dict], output_dir: str) -> bool:
        """批量写入LabelMe JSON文件

        Args:
            data_list: 图像标注数据列表
            output_dir: 输出目录

        Returns:
            是否全部写入成功
        """
        if not data_list:
            if self.verbose:
                self.logger.info(f"警告: 数据列表为空")
            return False

        os.makedirs(output_dir, exist_ok=True)

        success_count = 0
        for data in data_list:
            # 根据image_id生成输出文件名
            image_id = data.get('image_id', 'unknown')
            output_path = os.path.join(output_dir, f"{image_id}.json")

            try:
                if self.write(data, output_path):
                    success_count += 1
            except Exception as e:
                if self.verbose:
                    self.logger.info(f"警告: 跳过图像 {image_id}, 错误: {e}")

        all_success = success_count == len(data_list)

        if self.verbose:
            self.logger.info(f"批量写入完成: 目录 {output_dir}")
            self.logger.info(f"  成功写入文件数: {success_count}/{len(data_list)}")

        return all_success

    def _parse_shape(self, shape: Dict, image_size: Tuple[int, int], require_segmentation: bool = False) -> Optional[Dict]:
        """解析单个shape为统一格式

        Args:
            shape: LabelMe shape字典
            image_size: 图像尺寸 (width, height)
            require_segmentation: 是否要求分割格式。如果True，只接受shape_type="polygon"的标注

        Returns:
            统一格式的标注字典，包含bbox和segmentation
        """
        if 'label' not in shape or 'points' not in shape or 'shape_type' not in shape:
            return None

        width, height = image_size
        points = shape['points']

        # 初始化标注
        annotation = {
            "category_id": 0,  # 默认类别ID
            "category_name": shape['label'],
            "bbox": None,
            "segmentation": None
        }

        # 根据shape_type处理
        if shape['shape_type'] == 'rectangle' and len(points) == 2:
            # 矩形框
            if require_segmentation:
                # 要求分割格式但标注是矩形框，跳过
                if self.verbose:
                    self.logger.info(f"警告: 要求分割格式但标注是矩形框，跳过: {shape['label']}")
                return None
            # 矩形框: points = [[x1, y1], [x2, y2]]
            x1, y1 = points[0]
            x2, y2 = points[1]

            # 确保坐标顺序
            x_min = min(x1, x2)
            y_min = min(y1, y2)
            x_max = max(x1, x2)
            y_max = max(y1, y2)

            # 转换为[x_min, y_min, width, height]
            bbox_width = x_max - x_min
            bbox_height = y_max - y_min

            # 边界检查
            x_min = max(0, min(x_min, width - 1))
            y_min = max(0, min(y_min, height - 1))
            bbox_width = max(1, min(bbox_width, width - x_min))
            bbox_height = max(1, min(bbox_height, height - y_min))

            annotation["bbox"] = [x_min, y_min, bbox_width, bbox_height]

            # 为矩形生成多边形（如果需要分割）
            annotation["segmentation"] = [[
                x_min, y_min,
                x_max, y_min,
                x_max, y_max,
                x_min, y_max
            ]]

        elif shape['shape_type'] == 'polygon' and len(points) >= 3:
            # 多边形: points = [[x1, y1], [x2, y2], ...]
            # 展平点列表
            flattened_points = []
            for point in points:
                if len(point) >= 2:
                    # 边界检查
                    x = max(0, min(point[0], width - 1))
                    y = max(0, min(point[1], height - 1))
                    flattened_points.extend([x, y])

            if len(flattened_points) >= 6:  # 至少3个点
                annotation["segmentation"] = [flattened_points]

                # 计算边界框
                xs = flattened_points[0::2]
                ys = flattened_points[1::2]
                x_min = min(xs)
                y_min = min(ys)
                x_max = max(xs)
                y_max = max(ys)

                bbox_width = x_max - x_min
                bbox_height = y_max - y_min

                annotation["bbox"] = [x_min, y_min, bbox_width, bbox_height]

        else:
            # 不支持的shape类型
            if self.verbose:
                self.logger.info(f"警告: 跳过不支持的shape类型: {shape['shape_type']}")
            return None

        return annotation

    def _create_shape(self, annotation: Dict, image_size: Tuple[int, int]) -> Optional[Dict]:
        """从统一格式创建LabelMe shape

        Args:
            annotation: 统一格式的标注字典
            image_size: 图像尺寸 (width, height)

        Returns:
            LabelMe shape字典
        """
        if 'category_name' not in annotation:
            return None

        shape = {
            "label": annotation["category_name"],
            "group_id": None,
            "flags": {}
        }

        width, height = image_size

        # 检查分割数据
        if annotation.get("segmentation") and annotation["segmentation"][0]:
            points_flat = annotation["segmentation"][0]

            # 检查是否为从边界框生成的4点多边形（8个坐标）
            is_bbox_polygon = (len(points_flat) == 8 and
                              annotation.get("force_polygon", False))

            if is_bbox_polygon:
                # 强制分割模式：从边界框生成的多边形→多边形
                # 将展平的坐标转换为点列表
                points = []
                for i in range(0, len(points_flat), 2):
                    if i + 1 < len(points_flat):
                        x = max(0, min(points_flat[i], width - 1))
                        y = max(0, min(points_flat[i + 1], height - 1))
                        points.append([x, y])

                if len(points) >= 3:
                    shape["points"] = points
                    shape["shape_type"] = "polygon"
                    return shape
            else:
                # 真实分割数据→多边形
                # 将展平的坐标转换为点列表
                points = []
                for i in range(0, len(points_flat), 2):
                    if i + 1 < len(points_flat):
                        x = max(0, min(points_flat[i], width - 1))
                        y = max(0, min(points_flat[i + 1], height - 1))
                        points.append([x, y])

                if len(points) >= 3:
                    shape["points"] = points
                    shape["shape_type"] = "polygon"
                    return shape

        # 使用边界框数据→矩形
        if annotation.get("bbox"):
            bbox = annotation["bbox"]
            x_min, y_min, bbox_width, bbox_height = bbox

            # 边界检查
            x_min = max(0, min(x_min, width - 1))
            y_min = max(0, min(y_min, height - 1))
            bbox_width = max(1, min(bbox_width, width - x_min))
            bbox_height = max(1, min(bbox_height, height - y_min))

            x_max = x_min + bbox_width
            y_max = y_min + bbox_height

            shape["points"] = [[x_min, y_min], [x_max, y_max]]
            shape["shape_type"] = "rectangle"
            return shape

        # 无有效数据
        if self.verbose:
            self.logger.info(f"警告: 标注无有效bbox或segmentation数据")
        return None
