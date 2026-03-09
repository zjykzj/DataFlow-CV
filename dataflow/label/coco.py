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
from typing import Dict, List, Tuple, Union, Optional


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
            print(f"读取COCO文件: {json_path}")
            print(f"  图像数量: {len(coco_data['images'])}")
            print(f"  标注数量: {len(coco_data['annotations'])}")
            print(f"  类别数量: {len(coco_data['categories'])}")

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
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            if self.verbose:
                print(f"写入COCO文件: {output_path}")
                print(f"  图像数量: {len(data['images'])}")
                print(f"  标注数量: {len(data['annotations'])}")
                print(f"  类别数量: {len(data['categories'])}")

            return True
        except Exception as e:
            if self.verbose:
                print(f"写入文件失败: {output_path}, 错误: {e}")
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

        # 按图像ID分组标注
        annotations_by_image = {}
        for annotation in coco_data['annotations']:
            image_id = annotation['image_id']
            if image_id not in annotations_by_image:
                annotations_by_image[image_id] = []
            unified_annotation = self._coco_to_unified(annotation, category_map, require_segmentation)
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
                                   licenses: Optional[List[Dict]] = None) -> Dict:
        """将统一格式的图像标注列表转换为COCO格式

        Args:
            unified_data: 统一格式的图像标注数据列表
            info: 数据集信息（可选）
            licenses: 许可证信息（可选）

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
                    annotation, image_id, next_annotation_id, category_id
                )
                if coco_annotation:
                    coco_data["annotations"].append(coco_annotation)
                    next_annotation_id += 1

        return coco_data

    def _coco_to_unified(self, coco_annotation: Dict, category_map: Dict[int, str],
                         require_segmentation: bool = False) -> Optional[Dict]:
        """将COCO标注转换为统一格式

        Args:
            coco_annotation: COCO标注字典
            category_map: 类别ID到名称的映射
            require_segmentation: 是否要求分割格式。如果True，只接受包含分割数据的标注

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
            "segmentation": None
        }

        # 边界框
        if 'bbox' in coco_annotation and coco_annotation['bbox']:
            bbox = coco_annotation['bbox']
            # COCO格式: [x, y, width, height]
            if len(bbox) >= 4:
                annotation["bbox"] = bbox

        # 分割多边形
        has_original_segmentation = False
        if 'segmentation' in coco_annotation and coco_annotation['segmentation']:
            segmentation = coco_annotation['segmentation']
            # COCO分割格式: [[x1, y1, x2, y2, ...]] 或 RLE
            if isinstance(segmentation, list) and segmentation:
                # 取第一个多边形（忽略RLE格式）
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

        return annotation

    def _unified_to_coco(self, unified_annotation: Dict, image_id: int,
                        annotation_id: int, category_id: int) -> Optional[Dict]:
        """将统一格式标注转换为COCO格式

        Args:
            unified_annotation: 统一格式标注字典
            image_id: 图像ID
            annotation_id: 标注ID
            category_id: 类别ID

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

        # 优先使用分割数据
        if unified_annotation.get("segmentation") and unified_annotation["segmentation"][0]:
            segmentation = unified_annotation["segmentation"][0]
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

        # 使用边界框数据
        elif unified_annotation.get("bbox"):
            bbox = unified_annotation["bbox"]
            coco_annotation["bbox"] = bbox
            coco_annotation["area"] = bbox[2] * bbox[3]

            # 从边界框创建简单多边形
            x, y, width, height = bbox
            coco_annotation["segmentation"] = [[
                x, y,
                x + width, y,
                x + width, y + height,
                x, y + height
            ]]
        else:
            # 无有效数据
            return None

        return coco_annotation
