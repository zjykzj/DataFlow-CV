#!/usr/bin/env python3
"""
COCO标注格式处理示例

展示如何使用CocoAnnotationHandler读取、处理和写入COCO格式标注。
支持RLE格式和多边形格式。
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dataflow.label import CocoAnnotationHandler
from dataflow.util import LoggingOperations


def demo_standard_format():
    """标准COCO格式（多边形）演示"""
    log_ops = LoggingOperations()
    logger = log_ops.get_logger("coco_demo_std", level="INFO")

    logger.info("COCO标准格式（多边形）标注处理示例")
    logger.info("=" * 50)

    # 示例数据路径
    data_dir = project_root / "assets" / "test_data" / "det" / "coco"
    annotation_file = data_dir / "annotations.json"

    if not annotation_file.exists():
        logger.error(f"标注文件不存在: {annotation_file}")
        logger.info("请确保已准备示例数据")
        return

    # 创建COCO处理器
    logger.info(f"创建COCO处理器，标注文件: {annotation_file}")
    handler = CocoAnnotationHandler(
        annotation_file=str(annotation_file),
        strict_mode=True,
        logger=logger
    )

    # 读取标注数据
    logger.info("正在读取COCO标注...")
    result = handler.read()

    if not result.success:
        logger.error(f"读取失败: {result.message}")
        if result.errors:
            for error in result.errors:
                logger.error(f"  - {error}")
        return

    # 显示RLE格式检测结果
    logger.info(f"RLE格式检测: {handler.is_rle}")

    logger.info(f"\n成功读取标注信息:")
    logger.info(f"  图片数量: {len(result.data.images)}")
    logger.info(f"  标注数量: {result.data.num_objects}")
    logger.info(f"  类别数量: {len(result.data.categories)}")

    # 显示数据集信息
    if result.data.dataset_info:
        logger.info("\n数据集信息:")
        for key, value in result.data.dataset_info.items():
            logger.info(f"  {key}: {value}")

    # 显示第一张图片的详细信息
    if result.data.images:
        img = result.data.images[0]
        logger.info(f"\n第一张图片信息:")
        logger.info(f"  图片ID: {img.image_id}")
        logger.info(f"  路径: {img.image_path}")
        logger.info(f"  尺寸: {img.width}x{img.height}")
        logger.info(f"  对象数量: {len(img.objects)}")

        for i, obj in enumerate(img.objects[:3]):  # 显示前3个对象
            logger.info(f"  对象 {i+1}: {obj.class_name} (ID: {obj.class_id})")
            if obj.bbox:
                logger.info(f"    边界框: x={obj.bbox.x:.3f}, y={obj.bbox.y:.3f}, "
                          f"w={obj.bbox.width:.3f}, h={obj.bbox.height:.3f}")
            if obj.segmentation:
                logger.info(f"    分割点数: {len(obj.segmentation.points)}")
                # 显示前3个点
                for j, (x, y) in enumerate(obj.segmentation.points[:3]):
                    logger.info(f"      点 {j+1}: x={x:.3f}, y={y:.3f}")

    # 格式转换示例
    logger.info("\n进行格式转换测试...")
    output_file = data_dir / "annotations_converted.json"

    write_result = handler.write(result.data, str(output_file))

    if write_result.success:
        logger.info(f"成功写入到: {output_file}")

        # 验证写入的文件
        logger.info("验证写入的文件...")
        verify_handler = CocoAnnotationHandler(
            annotation_file=str(output_file),
            strict_mode=False  # 验证时使用非严格模式
        )
        verify_result = verify_handler.read()

        if verify_result.success:
            logger.info("验证通过！写入的文件可以正确读取。")
        else:
            logger.warning(f"验证警告: {verify_result.message}")
    else:
        logger.error(f"写入失败: {write_result.message}")

    logger.info("\n标准格式示例完成！\n")


def demo_rle_format():
    """RLE格式COCO标注演示"""
    log_ops = LoggingOperations()
    logger = log_ops.get_logger("coco_demo_rle", level="INFO")

    logger.info("COCO RLE格式标注处理示例")
    logger.info("=" * 50)

    # 示例数据路径
    data_dir = project_root / "assets" / "test_data" / "seg" / "coco"
    annotation_file = data_dir / "annotations-rle.json"

    if not annotation_file.exists():
        logger.error(f"标注文件不存在: {annotation_file}")
        logger.info("请确保已准备示例数据")
        return

    # 检查pycocotools是否安装
    try:
        from pycocotools import mask as coco_mask
        logger.info("pycocotools已安装，支持RLE处理")
    except ImportError:
        logger.warning("pycocotools未安装，RLE功能受限")
        logger.info("安装命令: pip install pycocotools")
        # 继续演示，但RLE解码会失败

    # 创建COCO处理器
    logger.info(f"创建COCO处理器，标注文件: {annotation_file}")
    handler = CocoAnnotationHandler(
        annotation_file=str(annotation_file),
        strict_mode=False,  # 使用非严格模式，因为RLE解码可能失败
        logger=logger
    )

    # 读取标注数据
    logger.info("正在读取COCO RLE标注...")
    result = handler.read()

    if not result.success:
        logger.error(f"读取失败: {result.message}")
        if result.errors:
            for error in result.errors:
                logger.error(f"  - {error}")
        return

    # 显示RLE格式检测结果
    logger.info(f"RLE格式检测: {handler.is_rle}")

    logger.info(f"\n成功读取标注信息:")
    logger.info(f"  图片数量: {len(result.data.images)}")
    logger.info(f"  标注数量: {result.data.num_objects}")
    logger.info(f"  类别数量: {len(result.data.categories)}")

    # 显示标注类型
    logger.info(f"\n标注类型检测:")
    logger.info(f"  目标检测: {handler.is_det}")
    logger.info(f"  实例分割: {handler.is_seg}")

    # 测试RLE输出
    logger.info("\n测试RLE格式输出...")
    output_file = data_dir / "annotations_rle_output.json"

    # 尝试使用RLE格式输出
    write_result = handler.write(result.data, str(output_file), output_rle=True)

    if write_result.success:
        logger.info(f"成功写入RLE格式到: {output_file}")

        # 检查输出是否包含RLE
        import json
        with open(output_file, 'r', encoding='utf-8') as f:
            output_data = json.load(f)

        rle_count = sum(
            1 for ann in output_data['annotations']
            if isinstance(ann.get('segmentation'), dict) and 'counts' in ann.get('segmentation', {})
        )
        logger.info(f"  输出中包含 {rle_count} 个RLE格式标注")

    else:
        logger.warning(f"RLE格式写入失败，尝试多边形格式...")
        write_result = handler.write(result.data, str(output_file), output_rle=False)
        if write_result.success:
            logger.info(f"成功写入多边形格式到: {output_file}")
        else:
            logger.error(f"写入失败: {write_result.message}")

    logger.info("\nRLE格式示例完成！\n")


def demo_format_conversion():
    """格式转换演示：多边形 <-> RLE"""
    log_ops = LoggingOperations()
    logger = log_ops.get_logger("coco_demo_conv", level="INFO")

    logger.info("COCO格式转换演示")
    logger.info("=" * 50)

    # 使用标准格式数据
    data_dir = project_root / "assets" / "test_data" / "det" / "coco"
    annotation_file = data_dir / "annotations.json"

    if not annotation_file.exists():
        logger.error(f"标注文件不存在: {annotation_file}")
        return

    # 检查pycocotools
    try:
        from pycocotools import mask as coco_mask
        has_pycocotools = True
    except ImportError:
        has_pycocotools = False
        logger.warning("pycocotools未安装，跳过RLE转换测试")
        logger.info("安装命令: pip install pycocotools")

    handler = CocoAnnotationHandler(
        annotation_file=str(annotation_file),
        strict_mode=True,
        logger=logger
    )

    # 读取数据
    result = handler.read()
    if not result.success:
        logger.error(f"读取失败: {result.message}")
        return

    logger.info(f"读取成功: {result.data.num_objects} 个对象")

    # 转换测试
    test_dir = data_dir / "conversion_test"
    test_dir.mkdir(exist_ok=True)

    # 1. 原始格式输出
    output1 = test_dir / "original.json"
    handler.write(result.data, str(output1), output_rle=False)
    logger.info(f"1. 多边形格式输出: {output1}")

    if has_pycocotools:
        # 2. RLE格式输出
        output2 = test_dir / "rle.json"
        handler.write(result.data, str(output2), output_rle=True)
        logger.info(f"2. RLE格式输出: {output2}")

        # 3. 读取RLE输出并验证
        handler2 = CocoAnnotationHandler(
            annotation_file=str(output2),
            strict_mode=False,
            logger=logger
        )
        result2 = handler2.read()
        if result2.success:
            logger.info(f"3. RLE文件验证: 成功读取 {result2.data.num_objects} 个对象")
            logger.info(f"   RLE格式检测: {handler2.is_rle}")
        else:
            logger.warning(f"3. RLE文件验证失败: {result2.message}")

    logger.info("\n格式转换演示完成！\n")


def main():
    """主函数"""
    log_ops = LoggingOperations()
    logger = log_ops.get_logger("coco_demo", level="INFO")

    logger.info("=" * 60)
    logger.info("COCO标注格式处理示例")
    logger.info("=" * 60)

    # 运行标准格式演示
    demo_standard_format()

    # 运行RLE格式演示
    demo_rle_format()

    # 运行格式转换演示
    demo_format_conversion()

    logger.info("所有示例完成！")


if __name__ == "__main__":
    main()