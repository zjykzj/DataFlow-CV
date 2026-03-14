# -*- coding: utf-8 -*-
"""
@Time    : 2026/3/9 20:44
@File    : yolo.py
@Author  : zj
@Description: YOLO格式标签处理器

YOLO格式描述:
YOLO格式每张图片对应一个.txt文件：
- 每行一个标注
- 检测格式：`class_id x_center y_center width height`（归一化坐标）
- 分割格式：`class_id x1 y1 x2 y2 ...`（多边形顶点，归一化坐标）
- 需要单独的类别文件（如class.names），每行一个类别名称

坐标系统:
- YOLO使用归一化坐标（0-1），需要图像尺寸进行转换
- 归一化公式: x_center = (x_min + width/2) / image_width
- 反归一化: x_min = (x_center - width/2) * image_width
"""

import os
import glob
import logging
from typing import Dict, List, Tuple, Union, Optional


class YoloHandler:
    """YOLO格式标签处理器

    支持目标检测（bbox）和实例分割（polygon）标注。
    提供单个文件和批量文件的读写功能。
    """

    def __init__(self, verbose: bool = False):
        """初始化YOLO处理器

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

    def read(self, label_path: str, image_path: str, classes: List[str],
             image_size: Optional[Tuple[int, int]] = None,
             require_segmentation: bool = False) -> Dict:
        """读取单个YOLO标签文件

        Args:
            label_path: YOLO标签文件路径 (.txt)
            image_path: 对应图像文件路径
            classes: 类别名称列表
            image_size: 可选图像尺寸 (width, height)。如果未提供，将尝试从图像文件读取
            require_segmentation: 是否强制分割模式。如果True，检测标注（4个坐标）将生成为从边界框创建的多边形，分割标注（6+个坐标）正常处理。如果False，自动检测格式类型。

        Returns:
            统一格式的图像标注数据字典，结构如下：
            {
                "image_id": str,          # 图像ID（文件名）
                "image_path": str,        # 图像路径
                "width": int,             # 图像宽度
                "height": int,            # 图像高度
                "annotations": List[dict] # 标注列表
            }

        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 文件格式错误
        """
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"标签文件不存在: {label_path}")

        # 获取图像尺寸
        width, height = 0, 0
        if image_size is not None:
            width, height = image_size
        elif os.path.exists(image_path):
            # 尝试从图像文件读取尺寸
            size = self._get_image_size(image_path)
            if size:
                width, height = size
            elif self.verbose:
                self.logger.info(f"警告: 无法获取图像尺寸: {image_path}")
        else:
            if self.verbose:
                self.logger.info(f"警告: 图像文件不存在: {image_path}")

        # 读取标签文件
        annotations = []
        try:
            with open(label_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except Exception as e:
            raise ValueError(f"读取文件失败: {e}")

        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 1:
                continue

            try:
                class_id = int(parts[0])
                if class_id < 0 or class_id >= len(classes):
                    if self.verbose:
                        self.logger.info(f"警告: 行 {line_num}: 类别ID {class_id} 超出范围")
                    continue

                coords = [float(x) for x in parts[1:]]

                # 判断是检测格式还是分割格式
                # 检测格式: 4个坐标 (x_center, y_center, width, height)
                # 分割格式: 2n个坐标 (多边形顶点)
                if len(coords) == 4:
                    # 检测格式
                    # require_segmentation=True 表示强制分割模式，检测标注应生成为多边形
                    annotation = self._parse_detection(coords, class_id, classes, (width, height),
                                                      force_polygon=require_segmentation)
                elif len(coords) >= 6 and len(coords) % 2 == 0:
                    # 分割格式（至少3个点）
                    # require_segmentation=True 表示强制分割模式，但真实分割标注不需要force_polygon标记
                    annotation = self._parse_segmentation(coords, class_id, classes, (width, height),
                                                         force_polygon=require_segmentation)
                else:
                    if self.verbose:
                        self.logger.info(f"警告: 行 {line_num}: 坐标数量无效: {len(coords)}")
                    continue

                if annotation:
                    annotations.append(annotation)

            except ValueError as e:
                if self.verbose:
                    self.logger.info(f"警告: 行 {line_num}: 解析错误: {e}")

        image_id = os.path.splitext(os.path.basename(image_path))[0]
        image_annotations = {
            "image_id": image_id,
            "image_path": image_path,
            "width": width,
            "height": height,
            "annotations": annotations
        }

        if self.verbose:
            self.logger.info(f"读取YOLO标签文件: {label_path}")
            self.logger.info(f"  图像: {image_path}")
            self.logger.info(f"  标注数量: {len(annotations)}")

        return image_annotations

    def _get_image_size(self, image_path: str) -> Optional[Tuple[int, int]]:
        """获取图像尺寸

        Args:
            image_path: 图像文件路径

        Returns:
            图像尺寸 (width, height)，如果无法获取则返回None
        """
        # 尝试使用PIL（如果可用）
        try:
            from PIL import Image
            with Image.open(image_path) as img:
                return img.size  # (width, height)
        except ImportError:
            if self.verbose:
                self.logger.info("警告: PIL (Pillow) 未安装，无法自动获取图像尺寸")
            return None
        except Exception:
            if self.verbose:
                self.logger.info(f"警告: 无法读取图像尺寸: {image_path}")
            return None

    def read_batch(self, labels_dir: str, images_dir: str, classes_path: str,
                   label_ext: str = ".txt", image_exts: Tuple[str] = (".jpg", ".png", ".jpeg"),
                   require_segmentation: bool = False) -> List[Dict]:
        """批量读取YOLO标签文件

        Args:
            labels_dir: 标签文件目录
            images_dir: 图像文件目录
            classes_path: 类别文件路径
            label_ext: 标签文件扩展名
            image_exts: 图像文件扩展名元组
            require_segmentation: 是否强制分割模式。如果True，检测标注（4个坐标）将生成为从边界框创建的多边形，分割标注（6+个坐标）正常处理。如果False，自动检测格式类型。

        Returns:
            图像标注数据列表，每个元素为read()返回的格式

        Raises:
            FileNotFoundError: 目录或文件不存在
        """
        if not os.path.exists(labels_dir):
            raise FileNotFoundError(f"标签目录不存在: {labels_dir}")
        if not os.path.exists(images_dir):
            raise FileNotFoundError(f"图像目录不存在: {images_dir}")
        if not os.path.exists(classes_path):
            raise FileNotFoundError(f"类别文件不存在: {classes_path}")

        # 读取类别
        classes = self.read_classes(classes_path)
        if not classes:
            raise ValueError("类别文件为空")

        # 获取所有标签文件
        label_files = glob.glob(os.path.join(labels_dir, f"*{label_ext}"))
        if not label_files:
            if self.verbose:
                self.logger.info(f"警告: 目录中没有找到标签文件: {labels_dir}")
            return []

        results = []
        for label_file in label_files:
            # 根据标签文件名查找对应的图像文件
            label_basename = os.path.splitext(os.path.basename(label_file))[0]

            # 尝试不同扩展名
            image_file = None
            for ext in image_exts:
                possible_image = os.path.join(images_dir, f"{label_basename}{ext}")
                if os.path.exists(possible_image):
                    image_file = possible_image
                    break

            if not image_file:
                if self.verbose:
                    self.logger.info(f"警告: 找不到标签文件对应的图像: {label_basename}")
                continue

            try:
                image_annotations = self.read(label_file, image_file, classes, require_segmentation=require_segmentation)
                results.append(image_annotations)
            except (FileNotFoundError, ValueError) as e:
                if self.verbose:
                    self.logger.info(f"警告: 跳过文件 {label_file}, 错误: {e}")

        if self.verbose:
            self.logger.info(f"批量读取完成:")
            self.logger.info(f"  标签目录: {labels_dir}")
            self.logger.info(f"  图像目录: {images_dir}")
            self.logger.info(f"  成功读取文件数: {len(results)}")
            self.logger.info(f"  总标注数量: {sum(len(r['annotations']) for r in results)}")

        return results

    def write(self, image_annotations: Dict, output_path: str, classes: List[str]) -> bool:
        """写入单个YOLO标签文件

        Args:
            image_annotations: 统一格式的图像标注数据字典
            output_path: 输出标签文件路径
            classes: 类别名称列表

        Returns:
            是否写入成功

        Raises:
            ValueError: 数据格式错误
        """
        # 验证必需字段
        required_fields = ['image_id', 'annotations']
        for field in required_fields:
            if field not in image_annotations:
                raise ValueError(f"缺少必需字段: {field}")

        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        lines = []
        for annotation in image_annotations.get('annotations', []):
            # 获取类别ID
            category_name = annotation.get('category_name', '')
            try:
                class_id = classes.index(category_name)
            except ValueError:
                if self.verbose:
                    self.logger.info(f"警告: 类别 '{category_name}' 不在类别列表中")
                continue

            # 获取图像尺寸
            width = image_annotations.get('width', 0)
            height = image_annotations.get('height', 0)

            if width <= 0 or height <= 0:
                if self.verbose:
                    self.logger.info(f"警告: 图像尺寸无效: {width}x{height}")
                continue

            # 优先使用分割数据
            if annotation.get("segmentation") and annotation["segmentation"][0]:
                # 分割格式
                segmentation = annotation["segmentation"][0]
                normalized_coords = self._normalize_coords(segmentation, (width, height))
                if normalized_coords:
                    line = f"{class_id} " + " ".join(f"{coord:.6f}" for coord in normalized_coords)
                    lines.append(line)
            elif annotation.get("bbox"):
                # 检测格式
                bbox = annotation["bbox"]
                # 转换为YOLO格式: [x_center, y_center, width, height] (归一化)
                x_min, y_min, bbox_width, bbox_height = bbox
                x_center = x_min + bbox_width / 2
                y_center = y_min + bbox_height / 2

                normalized_bbox = [
                    x_center / width,
                    y_center / height,
                    bbox_width / width,
                    bbox_height / height
                ]

                # 边界检查（归一化坐标应在0-1之间）
                normalized_bbox = [max(0.0, min(1.0, coord)) for coord in normalized_bbox]

                line = f"{class_id} " + " ".join(f"{coord:.6f}" for coord in normalized_bbox)
                lines.append(line)
            else:
                if self.verbose:
                    self.logger.info(f"警告: 标注无有效bbox或segmentation数据")

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(lines))

            if self.verbose:
                self.logger.info(f"写入YOLO标签文件: {output_path}")
                self.logger.info(f"  图像: {image_annotations.get('image_path', 'unknown')}")
                self.logger.info(f"  标注行数: {len(lines)}")

            return True
        except Exception as e:
            if self.verbose:
                self.logger.info(f"写入文件失败: {output_path}, 错误: {e}")
            return False

    def write_batch(self, images_annotations: List[Dict], output_dir: str, classes_path: str) -> bool:
        """批量写入YOLO标签文件

        Args:
            images_annotations: 图像标注数据列表
            output_dir: 输出目录
            classes_path: 类别文件路径

        Returns:
            是否全部写入成功
        """
        if not images_annotations:
            if self.verbose:
                self.logger.info(f"警告: 数据列表为空")
            return False

        # 读取类别
        classes = self.read_classes(classes_path)
        if not classes:
            raise ValueError("类别文件为空")

        # 写入类别文件（如果需要）
        if not os.path.exists(classes_path):
            self.write_classes(classes, classes_path)

        os.makedirs(output_dir, exist_ok=True)

        success_count = 0
        for image_data in images_annotations:
            # 根据image_id生成输出文件名
            image_id = image_data.get('image_id', 'unknown')
            output_path = os.path.join(output_dir, f"{image_id}.txt")

            try:
                if self.write(image_data, output_path, classes):
                    success_count += 1
            except Exception as e:
                if self.verbose:
                    self.logger.info(f"警告: 跳过图像 {image_id}, 错误: {e}")

        all_success = success_count == len(images_annotations)

        if self.verbose:
            self.logger.info(f"批量写入完成: 目录 {output_dir}")
            self.logger.info(f"  成功写入文件数: {success_count}/{len(images_annotations)}")

        return all_success

    def read_classes(self, classes_path: str) -> List[str]:
        """读取类别文件

        Args:
            classes_path: 类别文件路径

        Returns:
            类别名称列表

        Raises:
            FileNotFoundError: 文件不存在
        """
        if not os.path.exists(classes_path):
            raise FileNotFoundError(f"类别文件不存在: {classes_path}")

        classes = []
        try:
            with open(classes_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        classes.append(line)
        except Exception as e:
            raise ValueError(f"读取类别文件失败: {e}")

        if self.verbose:
            self.logger.info(f"读取类别文件: {classes_path}")
            self.logger.info(f"  类别数量: {len(classes)}")

        return classes

    def write_classes(self, classes: List[str], output_path: str) -> bool:
        """写入类别文件

        Args:
            classes: 类别名称列表
            output_path: 输出文件路径

        Returns:
            是否写入成功
        """
        if not classes:
            raise ValueError("类别列表为空")

        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                for class_name in classes:
                    f.write(f"{class_name}\n")

            if self.verbose:
                self.logger.info(f"写入类别文件: {output_path}")
                self.logger.info(f"  类别数量: {len(classes)}")

            return True
        except Exception as e:
            if self.verbose:
                self.logger.info(f"写入类别文件失败: {output_path}, 错误: {e}")
            return False

    def _parse_detection(self, coords: List[float], class_id: int,
                        classes: List[str], image_size: Tuple[int, int],
                        force_polygon: bool = False) -> Optional[Dict]:
        """解析检测格式标注

        Args:
            coords: 归一化坐标 [x_center, y_center, width, height]
            class_id: 类别ID
            classes: 类别名称列表
            image_size: 图像尺寸 (width, height)
            force_polygon: 是否强制生成多边形分割数据

        Returns:
            统一格式的标注字典
        """
        if len(coords) != 4:
            return None

        width, height = image_size
        if width <= 0 or height <= 0:
            return None

        x_center, y_center, norm_width, norm_height = coords

        # 反归一化
        bbox_width = norm_width * width
        bbox_height = norm_height * height
        x_min = (x_center - norm_width / 2) * width
        y_min = (y_center - norm_height / 2) * height

        # 边界检查
        x_min = max(0, min(x_min, width - 1))
        y_min = max(0, min(y_min, height - 1))
        bbox_width = max(1, min(bbox_width, width - x_min))
        bbox_height = max(1, min(bbox_height, height - y_min))

        # 构建标注
        annotation = {
            "category_id": class_id,
            "category_name": classes[class_id] if class_id < len(classes) else f"class_{class_id}",
            "bbox": [x_min, y_min, bbox_width, bbox_height],
            "segmentation": None
        }

        # 只有在强制分割模式时才从边界框创建简单多边形
        if force_polygon:
            x_max = x_min + bbox_width
            y_max = y_min + bbox_height
            annotation["segmentation"] = [[
                x_min, y_min,
                x_max, y_min,
                x_max, y_max,
                x_min, y_max
            ]]
            annotation["force_polygon"] = True  # 标记为强制转换的多边形

        return annotation

    def _parse_segmentation(self, coords: List[float], class_id: int,
                           classes: List[str], image_size: Tuple[int, int],
                           force_polygon: bool = False) -> Optional[Dict]:
        """解析分割格式标注

        Args:
            coords: 归一化多边形顶点坐标 [x1, y1, x2, y2, ...]
            class_id: 类别ID
            classes: 类别名称列表
            image_size: 图像尺寸 (width, height)
            force_polygon: 是否强制生成多边形分割数据（对于真实分割标注应为False）

        Returns:
            统一格式的标注字典
        """
        if len(coords) < 6 or len(coords) % 2 != 0:
            return None

        width, height = image_size
        if width <= 0 or height <= 0:
            return None

        # 反归一化
        denormalized_coords = self._denormalize_coords(coords, (width, height))
        if not denormalized_coords:
            return None

        # 计算边界框
        xs = denormalized_coords[0::2]
        ys = denormalized_coords[1::2]
        x_min = min(xs)
        y_min = min(ys)
        x_max = max(xs)
        y_max = max(ys)

        bbox_width = x_max - x_min
        bbox_height = y_max - y_min

        # 边界检查
        x_min = max(0, min(x_min, width - 1))
        y_min = max(0, min(y_min, height - 1))
        bbox_width = max(1, min(bbox_width, width - x_min))
        bbox_height = max(1, min(bbox_height, height - y_min))

        # 构建标注
        annotation = {
            "category_id": class_id,
            "category_name": classes[class_id] if class_id < len(classes) else f"class_{class_id}",
            "bbox": [x_min, y_min, bbox_width, bbox_height],
            "segmentation": [denormalized_coords]
        }

        # 真实分割标注不标记force_polygon，或标记为False
        if force_polygon:
            annotation["force_polygon"] = False

        return annotation

    def _normalize_coords(self, coords: List[float], image_size: Tuple[int, int]) -> Optional[List[float]]:
        """坐标归一化

        Args:
            coords: 原始坐标列表 [x1, y1, x2, y2, ...]
            image_size: 图像尺寸 (width, height)

        Returns:
            归一化坐标列表
        """
        if len(coords) % 2 != 0:
            return None

        width, height = image_size
        if width <= 0 or height <= 0:
            return None

        normalized = []
        for i in range(0, len(coords), 2):
            x = coords[i] / width
            y = coords[i + 1] / height
            normalized.extend([x, y])

        return normalized

    def _denormalize_coords(self, coords: List[float], image_size: Tuple[int, int]) -> Optional[List[float]]:
        """坐标反归一化

        Args:
            coords: 归一化坐标列表 [x1, y1, x2, y2, ...]
            image_size: 图像尺寸 (width, height)

        Returns:
            原始坐标列表
        """
        if len(coords) % 2 != 0:
            return None

        width, height = image_size
        if width <= 0 or height <= 0:
            return None

        denormalized = []
        for i in range(0, len(coords), 2):
            x = coords[i] * width
            y = coords[i + 1] * height
            denormalized.extend([x, y])

        return denormalized
