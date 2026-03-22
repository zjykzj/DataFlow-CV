#!/usr/bin/env python3
"""
Lossless annotation processing demonstration.

展示DataFlow-CV标签模块的无损读/写功能。该功能确保读取和重新写入标注文件时
产生完全相同的输出，无论坐标转换或格式转换如何。

关键特性：
1. 原始数据保存：读取时保存完整的原始标注数据
2. 格式识别：跟踪每个标注组件的源格式
3. 原始数据优先写入：写入时优先使用原始数据而非转换数据
4. 混合数据处理：支持同时包含原始数据和新创建数据的标注
"""

import sys
import tempfile
import shutil
from pathlib import Path
import json

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dataflow.label import (
    LabelMeAnnotationHandler,
    YoloAnnotationHandler,
    CocoAnnotationHandler,
    verify_lossless_roundtrip,
    OriginalData,
    AnnotationFormat
)
from dataflow.util import LoggingOperations


def demo_labelme_lossless(logger):
    """演示LabelMe格式的无损处理功能"""
    logger.info("=" * 60)
    logger.info("LabelMe格式无损处理演示")
    logger.info("=" * 60)

    # 使用目标检测测试数据
    data_dir = project_root / "assets" / "test_data" / "det" / "labelme"
    class_file = data_dir / "classes.txt"

    if not data_dir.exists():
        logger.error(f"数据目录不存在: {data_dir}")
        return False

    # 创建临时目录用于输出
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        output_dir = temp_dir_path / "output_labelme"

        # 1. 读取原始数据
        logger.info(f"1. 读取LabelMe标注数据: {data_dir}")
        handler = LabelMeAnnotationHandler(
            label_dir=str(data_dir),
            class_file=str(class_file) if class_file.exists() else None,
            strict_mode=True
        )

        read_result = handler.read()
        if not read_result.success:
            logger.error(f"读取失败: {read_result.message}")
            return False

        dataset = read_result.data
        logger.info(f"   成功读取 {len(dataset.images)} 张图片")

        # 检查原始数据保存情况
        image_with_original = 0
        objects_with_original = 0

        for image_ann in dataset.images:
            if image_ann.has_original_data():
                image_with_original += 1
                logger.debug(f"   图片 {image_ann.image_id} 保存了原始数据")

            for obj in image_ann.objects:
                if obj.has_original_data():
                    objects_with_original += 1
                    # 验证原始数据格式
                    if obj.original_data.format == AnnotationFormat.LABELME.value:
                        logger.debug(f"     对象 {obj.class_name} 保存了LabelMe原始数据")

        logger.info(f"   {image_with_original} 张图片保存了原始数据")
        logger.info(f"   {objects_with_original} 个对象保存了原始数据")

        # 2. 写入数据（应使用原始数据）
        logger.info(f"2. 写入LabelMe标注到: {output_dir}")
        write_result = handler.write(dataset, str(output_dir))
        if not write_result.success:
            logger.error(f"写入失败: {write_result.message}")
            return False

        # 3. 验证无损性
        logger.info("3. 验证无损性")
        input_files = sorted(data_dir.glob("*.json"))
        output_files = sorted(output_dir.glob("*.json"))

        if len(input_files) != len(output_files):
            logger.error(f"文件数量不匹配: 输入={len(input_files)}, 输出={len(output_files)}")
            return False

        all_match = True
        for in_file, out_file in zip(input_files, output_files):
            with open(in_file, 'r', encoding='utf-8') as f1:
                data1 = json.load(f1)
            with open(out_file, 'r', encoding='utf-8') as f2:
                data2 = json.load(f2)

            # 移除imageData字段（可能为null）
            data1.pop("imageData", None)
            data2.pop("imageData", None)

            if data1 != data2:
                logger.error(f"文件 {in_file.name} 不匹配")
                all_match = False
            else:
                logger.info(f"   ✓ {in_file.name}: 文件完全匹配")

        if all_match:
            logger.info("   ✓ LabelMe无损验证通过")
            return True
        else:
            logger.error("   ✗ LabelMe无损验证失败")
            return False


def demo_yolo_lossless(logger):
    """演示YOLO格式的无损处理功能"""
    logger.info("\n" + "=" * 60)
    logger.info("YOLO格式无损处理演示")
    logger.info("=" * 60)

    # 使用目标检测测试数据
    data_dir = project_root / "assets" / "test_data" / "det" / "yolo"
    class_file = data_dir / "classes.txt"
    labels_dir = data_dir / "labels"

    if not labels_dir.exists():
        logger.error(f"标签目录不存在: {labels_dir}")
        return False

    image_dir = data_dir / "images"
    if not image_dir.exists():
        logger.error(f"图片目录不存在: {image_dir}")
        return False

    # 创建临时目录用于输出
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        output_dir = temp_dir_path / "output_yolo"

        # 1. 读取原始数据
        logger.info(f"1. 读取YOLO标注数据: {labels_dir}")
        handler = YoloAnnotationHandler(
            image_dir=str(image_dir),
            label_dir=str(labels_dir),
            class_file=str(class_file) if class_file.exists() else None,
            strict_mode=True
        )

        read_result = handler.read()
        if not read_result.success:
            logger.error(f"读取失败: {read_result.message}")
            return False

        dataset = read_result.data
        logger.info(f"   成功读取 {len(dataset.images)} 张图片")

        # 检查原始数据保存情况
        objects_with_original = 0
        for image_ann in dataset.images:
            for obj in image_ann.objects:
                if obj.has_original_data():
                    objects_with_original += 1
                    if obj.original_data.format == AnnotationFormat.YOLO.value:
                        logger.debug(f"     对象 {obj.class_name} 保存了YOLO原始数据")
                        # 显示原始行数据
                        if "line" in obj.original_data.raw_data:
                            logger.debug(f"       原始行: {obj.original_data.raw_data['line'].strip()}")

        logger.info(f"   {objects_with_original} 个对象保存了原始数据")

        # 2. 写入数据
        logger.info(f"2. 写入YOLO标注到: {output_dir}")
        write_result = handler.write(dataset, str(output_dir))
        if not write_result.success:
            logger.error(f"写入失败: {write_result.message}")
            return False

        # 3. 验证无损性
        logger.info("3. 验证无损性")
        input_files = sorted(labels_dir.glob("*.txt"))
        output_files = sorted(output_dir.glob("*.txt"))

        if len(input_files) != len(output_files):
            logger.error(f"文件数量不匹配: 输入={len(input_files)}, 输出={len(output_files)}")
            return False

        all_match = True
        for in_file, out_file in zip(input_files, output_files):
            with open(in_file, 'r', encoding='utf-8') as f1:
                lines1 = [line.rstrip() for line in f1.readlines()]
            with open(out_file, 'r', encoding='utf-8') as f2:
                lines2 = [line.rstrip() for line in f2.readlines()]

            if lines1 != lines2:
                logger.error(f"文件 {in_file.name} 不匹配")
                logger.error(f"  输入行: {lines1}")
                logger.error(f"  输出行: {lines2}")
                all_match = False
            else:
                logger.info(f"   ✓ {in_file.name}: 文件完全匹配")

        if all_match:
            logger.info("   ✓ YOLO无损验证通过")
            return True
        else:
            logger.error("   ✗ YOLO无损验证失败")
            return False


def demo_coco_lossless(logger):
    """演示COCO格式的无损处理功能"""
    logger.info("\n" + "=" * 60)
    logger.info("COCO格式无损处理演示")
    logger.info("=" * 60)

    # 使用目标检测测试数据
    data_file = project_root / "assets" / "test_data" / "det" / "coco" / "annotations.json"

    if not data_file.exists():
        logger.error(f"COCO文件不存在: {data_file}")
        return False

    # 创建临时目录用于输出
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        output_file = temp_dir_path / "output_coco.json"

        # 1. 读取原始数据
        logger.info(f"1. 读取COCO标注数据: {data_file}")
        handler = CocoAnnotationHandler(
            annotation_file=str(data_file),
            strict_mode=True
        )

        read_result = handler.read()
        if not read_result.success:
            logger.error(f"读取失败: {read_result.message}")
            return False

        dataset = read_result.data
        logger.info(f"   成功读取 {len(dataset.images)} 张图片")

        # 检查原始数据保存情况
        images_with_original = 0
        objects_with_original = 0
        objects_with_rle = 0

        for image_ann in dataset.images:
            if image_ann.has_original_data():
                images_with_original += 1

            for obj in image_ann.objects:
                if obj.has_original_data():
                    objects_with_original += 1
                    if obj.original_data.format == AnnotationFormat.COCO.value:
                        logger.debug(f"     对象 {obj.class_name} 保存了COCO原始数据")

                # 检查RLE数据保存
                if obj.segmentation and obj.segmentation.has_rle():
                    objects_with_rle += 1
                    logger.debug(f"     对象 {obj.class_name} 保存了RLE数据")

        logger.info(f"   {images_with_original} 张图片保存了原始数据")
        logger.info(f"   {objects_with_original} 个对象保存了原始数据")
        logger.info(f"   {objects_with_rle} 个对象保存了RLE数据")

        # 2. 写入数据（保留多边形格式）
        logger.info(f"2. 写入COCO标注到: {output_file} (保留多边形格式)")
        write_result = handler.write(dataset, str(output_file), output_rle=False)
        if not write_result.success:
            logger.error(f"写入失败: {write_result.message}")
            return False

        # 3. 验证无损性
        logger.info("3. 验证无损性")
        with open(data_file, 'r', encoding='utf-8') as f1:
            data1 = json.load(f1)
        with open(output_file, 'r', encoding='utf-8') as f2:
            data2 = json.load(f2)

        # 移除自动生成的字段
        for data in [data1, data2]:
            data.pop("__coco_original_data__", None)
            # 移除可能不同的描述字段
            for field in ["description", "url", "version", "year", "contributor", "date_created"]:
                data.pop(field, None)

        # 比较标注（ID可能重新生成，比较内容）
        if len(data1["annotations"]) != len(data2["annotations"]):
            logger.error(f"标注数量不匹配: 输入={len(data1['annotations'])}, 输出={len(data2['annotations'])}")
            return False

        all_match = True
        for i, (ann1, ann2) in enumerate(zip(data1["annotations"], data2["annotations"])):
            # 移除ID比较
            ann1_copy = {k: v for k, v in ann1.items() if k != "id"}
            ann2_copy = {k: v for k, v in ann2.items() if k != "id"}

            if ann1_copy != ann2_copy:
                logger.error(f"标注 {i} 内容不同")
                logger.error(f"  输入: {ann1_copy}")
                logger.error(f"  输出: {ann2_copy}")
                all_match = False

        # 比较图片和类别
        if data1["images"] != data2["images"]:
            logger.error("图片信息不同")
            all_match = False

        if data1["categories"] != data2["categories"]:
            logger.error("类别信息不同")
            all_match = False

        if all_match:
            logger.info("   ✓ COCO无损验证通过")
            return True
        else:
            logger.error("   ✗ COCO无损验证失败")
            return False


def demo_utility_functions(logger):
    """演示无损验证工具函数"""
    logger.info("\n" + "=" * 60)
    logger.info("无损验证工具函数演示")
    logger.info("=" * 60)

    # 使用目标检测测试数据
    labelme_dir = project_root / "assets" / "test_data" / "det" / "labelme"
    yolo_dir = project_root / "assets" / "test_data" / "det" / "yolo" / "labels"
    yolo_class_file = project_root / "assets" / "test_data" / "det" / "yolo" / "classes.txt"
    coco_file = project_root / "assets" / "test_data" / "det" / "coco" / "annotations.json"

    # 测试LabelMe验证
    logger.info("1. 测试LabelMe无损验证")
    if labelme_dir.exists():
        # 创建自定义处理器（演示使用方法）
        from dataflow.label.labelme_handler import LabelMeAnnotationHandler
        result = verify_lossless_roundtrip(
            input_path=str(labelme_dir),
            output_path="/tmp/test_output",
            handler_class=LabelMeAnnotationHandler
        )
        logger.info(f"   LabelMe无损验证结果: {'通过' if result else '失败'}")
    else:
        logger.warning("   LabelMe测试数据不存在")

    # 测试YOLO验证
    logger.info("2. 测试YOLO无损验证")
    if yolo_dir.exists() and yolo_class_file.exists():
        # 注意：YOLO验证需要类文件参数，这里简化演示
        logger.info("   YOLO验证需要自定义处理器，跳过演示")
    else:
        logger.warning("   YOLO测试数据不存在")

    # 测试COCO验证
    logger.info("3. 测试COCO无损验证")
    if coco_file.exists():
        from dataflow.label.coco_handler import CocoAnnotationHandler
        result = verify_lossless_roundtrip(
            input_path=str(coco_file),
            output_path="/tmp/test_output_coco.json",
            handler_class=CocoAnnotationHandler
        )
        logger.info(f"   COCO无损验证结果: {'通过' if result else '失败'}")
    else:
        logger.warning("   COCO测试数据不存在")

    logger.info("\n工具函数演示完成")


def main():
    """主函数"""
    # 配置日志
    log_ops = LoggingOperations()
    logger = log_ops.get_logger("lossless_demo", level="INFO")

    logger.info("=" * 60)
    logger.info("DataFlow-CV无损标注处理演示")
    logger.info("=" * 60)
    logger.info("")
    logger.info("本演示展示DataFlow-CV标签模块的无损读/写功能。")
    logger.info("该功能确保读取和重新写入标注文件时产生完全相同的输出。")
    logger.info("")

    # 运行各个演示
    success_count = 0
    total_demos = 3

    try:
        if demo_labelme_lossless(logger):
            success_count += 1
    except Exception as e:
        logger.error(f"LabelMe演示异常: {e}")

    try:
        if demo_yolo_lossless(logger):
            success_count += 1
    except Exception as e:
        logger.error(f"YOLO演示异常: {e}")

    try:
        if demo_coco_lossless(logger):
            success_count += 1
    except Exception as e:
        logger.error(f"COCO演示异常: {e}")

    # 演示工具函数
    try:
        demo_utility_functions(logger)
    except Exception as e:
        logger.error(f"工具函数演示异常: {e}")

    # 总结
    logger.info("\n" + "=" * 60)
    logger.info("演示总结")
    logger.info("=" * 60)
    logger.info(f"成功演示: {success_count}/{total_demos}")

    if success_count == total_demos:
        logger.info("✓ 所有无损功能演示成功")
        logger.info("✓ DataFlow-CV标签模块提供完全无损的标注处理")
    else:
        logger.warning(f"⚠ {total_demos - success_count} 个演示失败")
        logger.info("请检查测试数据或模块实现")

    logger.info("\n演示完成！")


if __name__ == "__main__":
    main()