# -*- coding: utf-8 -*-
"""
@Time    : 2026/3/9 20:43
@File    : coco.py
@Author  : zj
@Description: COCO格式标签处理器

COCO格式描述:
COCO格式是单个JSON文件包含整个数据集，包含：
- `info`: 数据集信息
- `licenses`: 许可证信息
- `images`: 图像列表，每个图像包含id, file_name, height, width
- `annotations`: 标注列表，每个标注包含：
  - `id`: 标注ID
  - `image_id`: 图像ID
  - `category_id`: 类别ID
  - `bbox`: [x, y, width, height]（检测）
  - `segmentation`: 多边形点列表（分割）
  - `area`: 面积
  - `iscrowd`: 是否拥挤
- `categories`: 类别列表，每个类别包含id, name, supercategory
"""

import os
import json
import logging
from typing import Dict, List, Tuple, Union, Optional

try:
    from pycocotools import mask as cocomask
    PYCOCO_AVAILABLE = True
except ImportError:
    PYCOCO_AVAILABLE = False
    cocomask = None


class CocoHandler:
    """COCO格式标签处理器

    支持目标检测（bbox）和实例分割（polygon）标注。
    提供COCO JSON文件的读写和查询功能。
    """

    def __init__(self, verbose: bool = False):
        """初始化COCO处理器

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

    def _polygon_to_rle(self, polygon: List[float], height: int, width: int) -> Dict:
        """将多边形点列表转换为RLE格式。

        Args:
            polygon: 多边形点列表 [x1, y1, x2, y2, ...]
            height: 图像高度
            width: 图像宽度

        Returns:
            RLE格式字典 {'counts': bytes, 'size': [height, width]}

        Raises:
            ImportError: pycocotools不可用
            ValueError: 多边形格式无效
        """
        if not PYCOCO_AVAILABLE:
            raise ImportError("pycocotools is not available. Please install pycocotools>=2.0.0 for RLE support.")

        if len(polygon) < 6 or len(polygon) % 2 != 0:
            raise ValueError(f"Invalid polygon format: expected at least 3 points (6 coordinates), got {len(polygon)} values")

        # 将多边形点列表转换为pycocotools所需的格式
        # pycocotools需要多边形作为numpy数组 [[x1, y1, x2, y2, ...]]
        import numpy as np
        poly_array = np.array(polygon, dtype=np.float32).reshape(1, -1)

        # 转换多边形为RLE
        rle = cocomask.frPyObjects(poly_array, height, width)
        if isinstance(rle, list) and len(rle) == 1:
            rle = rle[0]

        return rle

    def _rle_to_polygon(self, rle: Dict, height: int, width: int) -> List[float]:
        """将RLE格式转换为多边形点列表。

        Args:
            rle: RLE格式字典 {'counts': bytes, 'size': [height, width]}
            height: 图像高度
            width: 图像宽度

        Returns:
            多边形点列表 [x1, y1, x2, y2, ...]

        Raises:
            ImportError: pycocotools不可用
            ValueError: RLE格式无效
        """
        if not PYCOCO_AVAILABLE:
            raise ImportError("pycocotools is not available. Please install pycocotools>=2.0.0 for RLE support.")

        # 验证RLE格式
        if 'counts' not in rle or 'size' not in rle:
            raise ValueError("Invalid RLE format: missing 'counts' or 'size' key")

        # 解码RLE为二进制掩码
        mask = cocomask.decode(rle)

        # 从掩码中提取多边形
        contours = cocomask.toBbox(mask)
        if len(contours) == 0:
            # 尝试使用polygon extraction
            try:
                polygons = cocomask.toPolygons(mask)
                if polygons and len(polygons) > 0:
                    # 取第一个多边形并展平
                    polygon = polygons[0].flatten().tolist()
                    return polygon
            except:
                pass

            # 如果没有找到多边形，从边界框创建
            # 通常RLE应该可以转换为多边形，但作为回退
            if 'bbox' in rle:
                bbox = rle['bbox']
                x, y, w, h = bbox
                return [x, y, x + w, y, x + w, y + h, x, y + h]
            else:
                # 无法转换
                raise ValueError("Failed to convert RLE to polygon: empty mask")

        # 使用边界框创建简单多边形（作为最后的手段）
        # 注意：这可能会损失精度，但对于大多数情况应该足够
        if len(contours) > 0:
            x, y, w, h = contours[0]
            return [x, y, x + w, y, x + w, y + h, x, y + h]

        raise ValueError("Failed to convert RLE to polygon")

    def read(self, json_path: str) -> Dict:
        """读取COCO JSON文件

        Args:
            json_path: COCO JSON文件路径

        Returns:
            COCO格式数据字典，包含完整的数据集信息

        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 文件格式错误
        """
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"COCO文件不存在: {json_path}")

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                coco_data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON解析错误: {e}")

        # 验证必需字段
        required_sections = ['images', 'annotations', 'categories']
        for section in required_sections:
            if section not in coco_data:
                raise ValueError(f"缺少必需部分: {section}")

        # 添加默认字段（如果不存在）
        if 'info' not in coco_data:
            coco_data['info'] = {
                "description": "COCO dataset",
                "url": "",
                "version": "1.0",
                "year": 2026,
                "contributor": "",
                "date_created": "2026/03/09"
            }

        if 'licenses' not in coco_data:
            coco_data['licenses'] = [{
                "url": "",
                "id": 0,
                "name": "Unknown"
            }]

        if self.verbose:
            self.logger.info(f"读取COCO文件: {json_path}")
            self.logger.info(f"  图像数量: {len(coco_data['images'])}")
            self.logger.info(f"  标注数量: {len(coco_data['annotations'])}")
            self.logger.info(f"  类别数量: {len(coco_data['categories'])}")

        return coco_data

    def write(self, data: Dict, output_path: str) -> bool:
        """写入COCO JSON文件

        Args:
            data: COCO格式数据字典（与read()返回格式相同）
            output_path: 输出JSON文件路径

        Returns:
            是否写入成功

        Raises:
            ValueError: 数据格式错误
        """
        # 验证必需字段
        required_sections = ['images', 'annotations', 'categories']
        for section in required_sections:
            if section not in data:
                raise ValueError(f"缺少必需部分: {section}")

        # 确保字段存在（添加默认值）
        if 'info' not in data:
            data['info'] = {
                "description": "COCO dataset",
                "url": "",
                "version": "1.0",
                "year": 2026,
                "contributor": "",
                "date_created": "2026/03/09"
            }

        if 'licenses' not in data:
            data['licenses'] = [{
                "url": "",
                "id": 0,
                "name": "Unknown"
            }]

        # 确保输出目录存在
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            if self.verbose:
                self.logger.info(f"写入COCO文件: {output_path}")
                self.logger.info(f"  图像数量: {len(data['images'])}")
                self.logger.info(f"  标注数量: {len(data['annotations'])}")
                self.logger.info(f"  类别数量: {len(data['categories'])}")

            return True
        except Exception as e:
            if self.verbose:
                self.logger.info(f"写入文件失败: {output_path}, 错误: {e}")
            return False

    def get_image_annotations(self, coco_data: Dict, image_id: int) -> List[Dict]:
        """获取指定图像的所有标注

        Args:
            coco_data: COCO格式数据字典
            image_id: 图像ID

        Returns:
            该图像的标注列表（统一格式）
        """
        if 'annotations' not in coco_data:
            return []

        # 获取类别映射
        category_map = self.get_category_map(coco_data)

        image_annotations = []
        for annotation in coco_data['annotations']:
            if annotation['image_id'] == image_id:
                # 转换为统一格式
                unified_annotation = self._coco_to_unified(annotation, category_map)
                if unified_annotation:
                    image_annotations.append(unified_annotation)

        return image_annotations

    def get_image_info(self, coco_data: Dict, image_id: int) -> Optional[Dict]:
        """获取指定图像的信息

        Args:
            coco_data: COCO格式数据字典
            image_id: 图像ID

        Returns:
            图像信息字典，包含id, file_name, height, width
        """
        if 'images' not in coco_data:
            return None

        for image in coco_data['images']:
            if image['id'] == image_id:
                return image

        return None

    def get_category_map(self, coco_data: Dict) -> Dict[int, str]:
        """获取类别ID到名称的映射

        Args:
            coco_data: COCO格式数据字典

        Returns:
            字典：{category_id: category_name}
        """
        if 'categories' not in coco_data:
            return {}

        category_map = {}
        for category in coco_data['categories']:
            category_map[category['id']] = category['name']

        return category_map

    def convert_to_unified_format(self, coco_data: Dict, image_dir: str = "",
                                  require_segmentation: bool = False) -> List[Dict]:
        """将COCO数据转换为统一格式的图像标注列表

        Args:
            coco_data: COCO格式数据字典
            image_dir: 图像目录路径（用于构建完整图像路径）
            require_segmentation: 是否要求分割格式。如果True，只接受包含分割数据的标注

        Returns:
            统一格式的图像标注数据列表，每个元素格式如下：
            {
                "image_id": str/int,      # 图像ID
                "image_path": str,        # 完整图像路径
                "width": int,             # 图像宽度
                "height": int,            # 图像高度
                "annotations": List[dict] # 标注列表（统一格式）
            }
        """
        if 'images' not in coco_data or 'annotations' not in coco_data:
            return []

        # 获取类别映射
        category_map = self.get_category_map(coco_data)

        # 创建图像ID到图像信息的映射
        image_info_map = {}
        for image in coco_data['images']:
            image_info_map[image['id']] = {
                'height': image['height'],
                'width': image['width']
            }

        # 按图像ID分组标注
        annotations_by_image = {}
        for annotation in coco_data['annotations']:
            image_id = annotation['image_id']
            if image_id not in annotations_by_image:
                annotations_by_image[image_id] = []

            # 获取图像尺寸用于RLE解码
            image_height = None
            image_width = None
            if image_id in image_info_map:
                image_height = image_info_map[image_id]['height']
                image_width = image_info_map[image_id]['width']

            unified_annotation = self._coco_to_unified(
                annotation, category_map, require_segmentation,
                image_height=image_height, image_width=image_width
            )
            if unified_annotation:
                annotations_by_image[image_id].append(unified_annotation)

        # 构建结果列表
        results = []
        for image in coco_data['images']:
            image_id = image['id']
            image_path = os.path.join(image_dir, image['file_name']) if image_dir else image['file_name']

            image_data = {
                "image_id": image_id,
                "image_path": image_path,
                "width": image['width'],
                "height": image['height'],
                "annotations": annotations_by_image.get(image_id, [])
            }
            results.append(image_data)

        return results

    def convert_from_unified_format(self, unified_data: List[Dict],
                                   info: Optional[Dict] = None,
                                   licenses: Optional[List[Dict]] = None,
                                   rle: bool = False) -> Dict:
        """将统一格式的图像标注列表转换为COCO格式

        Args:
            unified_data: 统一格式的图像标注数据列表
            info: 数据集信息（可选）
            licenses: 许可证信息（可选）
            rle: 是否将分割转换为RLE格式

        Returns:
            COCO格式数据字典
        """
        if not unified_data:
            raise ValueError("输入数据为空")

        # 初始化COCO数据结构
        coco_data = {
            "info": info or {
                "description": "COCO dataset",
                "url": "",
                "version": "1.0",
                "year": 2026,
                "contributor": "",
                "date_created": "2026/03/09"
            },
            "licenses": licenses or [{
                "url": "",
                "id": 0,
                "name": "Unknown"
            }],
            "images": [],
            "annotations": [],
            "categories": []
        }

        # 收集所有类别
        category_name_to_id = {}
        next_category_id = 1
        next_annotation_id = 1

        # 处理每个图像
        for image_data in unified_data:
            # 添加图像信息
            image_id = image_data.get('image_id', next_annotation_id)
            coco_data["images"].append({
                "id": image_id,
                "file_name": os.path.basename(image_data['image_path']),
                "width": image_data['width'],
                "height": image_data['height']
            })

            # 处理每个标注
            for annotation in image_data.get('annotations', []):
                # 确保类别存在
                category_name = annotation.get('category_name', 'unknown')
                if category_name not in category_name_to_id:
                    category_name_to_id[category_name] = next_category_id
                    coco_data["categories"].append({
                        "id": next_category_id,
                        "name": category_name,
                        "supercategory": category_name
                    })
                    next_category_id += 1

                category_id = category_name_to_id[category_name]

                # 转换为COCO标注格式
                coco_annotation = self._unified_to_coco(
                    annotation, image_id, next_annotation_id, category_id,
                    rle=rle, image_height=image_data['height'], image_width=image_data['width']
                )
                if coco_annotation:
                    coco_data["annotations"].append(coco_annotation)
                    next_annotation_id += 1

        return coco_data

    def _coco_to_unified(self, coco_annotation: Dict, category_map: Dict[int, str],
                         require_segmentation: bool = False, image_height: Optional[int] = None,
                         image_width: Optional[int] = None) -> Optional[Dict]:
        """将COCO标注转换为统一格式

        Args:
            coco_annotation: COCO标注字典
            category_map: 类别ID到名称的映射
            require_segmentation: 是否要求分割格式。如果True，只接受包含分割数据的标注
            image_height: 图像高度（用于RLE解码）
            image_width: 图像宽度（用于RLE解码）

        Returns:
            统一格式的标注字典，如果require_segmentation=True但标注没有分割数据则返回None
        """
        if 'category_id' not in coco_annotation or 'image_id' not in coco_annotation:
            return None

        category_id = coco_annotation['category_id']
        category_name = category_map.get(category_id, f"class_{category_id}")

        annotation = {
            "category_id": category_id,
            "category_name": category_name,
            "bbox": None,
            "segmentation": None,
            "_is_rle": False  # 标记原始数据是否为RLE格式
        }

        # 边界框
        if 'bbox' in coco_annotation and coco_annotation['bbox']:
            bbox = coco_annotation['bbox']
            # COCO格式: [x, y, width, height]
            if len(bbox) >= 4:
                annotation["bbox"] = bbox

        # 分割多边形或RLE
        has_original_segmentation = False
        if 'segmentation' in coco_annotation and coco_annotation['segmentation']:
            segmentation = coco_annotation['segmentation']

            # 检查是否为RLE格式（字典包含'counts'和'size'键）
            is_rle = isinstance(segmentation, dict) and 'counts' in segmentation and 'size' in segmentation

            if is_rle:
                # RLE格式处理
                if image_height is None or image_width is None:
                    # 尝试从RLE的size字段获取尺寸
                    if 'size' in segmentation and isinstance(segmentation['size'], list) and len(segmentation['size']) == 2:
                        rle_height, rle_width = segmentation['size']
                        if rle_height and rle_width:
                            image_height = rle_height
                            image_width = rle_width
                        else:
                            raise ValueError("RLE segmentation requires image dimensions but none provided and RLE size is invalid")
                    else:
                        raise ValueError("RLE segmentation requires image dimensions but none provided and RLE size missing")

                try:
                    # 解码RLE为多边形
                    polygon = self._rle_to_polygon(segmentation, image_height, image_width)
                    annotation["segmentation"] = [polygon]
                    has_original_segmentation = True
                    annotation["_is_rle"] = True
                    if self.verbose:
                        self.logger.info(f"Decoded RLE segmentation for annotation {coco_annotation.get('id', 'unknown')}")
                except Exception as e:
                    if self.verbose:
                        self.logger.warning(f"Failed to decode RLE segmentation: {e}, falling back to polygon format if available")
                    # RLE解码失败，回退到多边形格式（如果可用）
                    pass

            # 多边形格式处理
            elif isinstance(segmentation, list) and segmentation:
                # 取第一个多边形
                if isinstance(segmentation[0], list):
                    annotation["segmentation"] = segmentation
                    has_original_segmentation = True

        # 如果没有分割但有边界框，从边界框创建简单多边形
        if not annotation.get("segmentation") and annotation.get("bbox"):
            bbox = annotation["bbox"]
            x, y, width, height = bbox
            annotation["segmentation"] = [[
                x, y,
                x + width, y,
                x + width, y + height,
                x, y + height
            ]]

        # 如果要求分割格式但标注没有原始分割数据，返回None
        if require_segmentation and not has_original_segmentation:
            return None

        # 移除调试标记（如果需要可以保留）
        # annotation.pop("_is_rle", None)

        return annotation

    def _unified_to_coco(self, unified_annotation: Dict, image_id: int,
                        annotation_id: int, category_id: int, rle: bool = False,
                        image_height: Optional[int] = None, image_width: Optional[int] = None) -> Optional[Dict]:
        """将统一格式标注转换为COCO格式

        Args:
            unified_annotation: 统一格式标注字典
            image_id: 图像ID
            annotation_id: 标注ID
            category_id: 类别ID
            rle: 是否将分割转换为RLE格式
            image_height: 图像高度（用于RLE编码）
            image_width: 图像宽度（用于RLE编码）

        Returns:
            COCO标注字典
        """
        if 'category_name' not in unified_annotation:
            return None

        coco_annotation = {
            "id": annotation_id,
            "image_id": image_id,
            "category_id": category_id,
            "iscrowd": 0
        }

        # 检查是否有原始分割数据（不是从边界框生成的）
        has_original_segmentation = unified_annotation.get("segmentation") and unified_annotation["segmentation"][0]

        # 优先使用分割数据
        if has_original_segmentation:
            segmentation = unified_annotation["segmentation"][0]

            # 检查是否为从边界框生成的简单多边形（4个点/8个坐标）
            is_simple_bbox_polygon = False
            if len(segmentation) == 8:  # 4个点
                # 检查是否是矩形多边形
                xs = segmentation[0::2]
                ys = segmentation[1::2]
                if len(set(xs)) == 2 and len(set(ys)) == 2:
                    is_simple_bbox_polygon = True

            # 尝试RLE转换（如果请求且不是从边界框生成的简单多边形）
            if rle and not is_simple_bbox_polygon and image_height and image_width:
                try:
                    rle_mask = self._polygon_to_rle(segmentation, image_height, image_width)
                    coco_annotation["segmentation"] = rle_mask
                    if self.verbose:
                        self.logger.info(f"Encoded polygon to RLE for annotation {annotation_id}")
                except Exception as e:
                    if self.verbose:
                        self.logger.warning(f"Failed to encode polygon to RLE: {e}, falling back to polygon format")
                    # RLE编码失败，回退到多边形格式
                    coco_annotation["segmentation"] = [segmentation]
            else:
                # 使用多边形格式
                coco_annotation["segmentation"] = [segmentation]

            # 计算边界框
            if len(segmentation) >= 6:
                xs = segmentation[0::2]
                ys = segmentation[1::2]
                x_min = min(xs)
                y_min = min(ys)
                x_max = max(xs)
                y_max = max(ys)
                width = x_max - x_min
                height = y_max - y_min
                coco_annotation["bbox"] = [x_min, y_min, width, height]
                coco_annotation["area"] = width * height
            else:
                # 无效分割，使用边界框
                if unified_annotation.get("bbox"):
                    bbox = unified_annotation["bbox"]
                    coco_annotation["bbox"] = bbox
                    coco_annotation["area"] = bbox[2] * bbox[3]

        # 使用边界框数据（无原始分割数据）
        elif unified_annotation.get("bbox"):
            bbox = unified_annotation["bbox"]
            coco_annotation["bbox"] = bbox
            coco_annotation["area"] = bbox[2] * bbox[3]

            # 从边界框创建简单多边形（即使rle=True，边界框也保持多边形格式）
            x, y, width, height = bbox
            simple_polygon = [
                x, y,
                x + width, y,
                x + width, y + height,
                x, y + height
            ]
            coco_annotation["segmentation"] = [simple_polygon]

            # 注意：对于边界框，即使rle=True也不转换为RLE，保持多边形格式
            # 这符合用户要求：检测标注不受--rle影响
        else:
            # 无有效数据
            return None

        return coco_annotation
