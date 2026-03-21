#!/usr/bin/env python3
"""
YOLO标注格式处理示例

展示如何使用YoloAnnotationHandler读取、处理和写入YOLO格式标注。
支持自动检测目标检测和实例分割格式。
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dataflow.label import YoloAnnotationHandler
from dataflow.util import LoggingOperations

def demo_detection():
    """目标检测数据演示"""
    log_ops = LoggingOperations()
    logger = log_ops.get_logger("yolo_demo_det", level="INFO")

    logger.info("YOLO目标检测标注处理示例")
    logger.info("=" * 50)

    # 示例数据路径
    data_dir = project_root / "assets" / "test_data" / "det" / "yolo"
    class_file = data_dir / "classes.txt"
    image_dir = data_dir / "images"
    label_dir = data_dir / "labels"

    if not data_dir.exists():
        logger.error(f"数据目录不存在: {data_dir}")
        logger.info("请确保已准备示例数据")
        return

    # 创建YOLO处理器
    logger.info(f"创建YOLO处理器:")
    logger.info(f"  标签目录: {label_dir}")
    logger.info(f"  类别文件: {class_file}")
    logger.info(f"  图片目录: {image_dir}")

    handler = YoloAnnotationHandler(
        label_dir=str(label_dir),
        class_file=str(class_file),
        image_dir=str(image_dir),
        strict_mode=True,
        logger=logger
    )

    # 读取标注数据
    logger.info("\n正在读取YOLO标注...")
    result = handler.read()

    if not result.success:
        logger.error(f"读取失败: {result.message}")
        if result.errors:
            for error in result.errors:
                logger.error(f"  - {error}")
        return

    # 显示标注类型检测结果
    logger.info(f"标注类型检测:")
    logger.info(f"  目标检测: {handler.is_det}")
    logger.info(f"  实例分割: {handler.is_seg}")

    logger.info(f"\n成功读取 {len(result.data.images)} 张图片的标注")
    logger.info(f"类别数量: {len(result.data.categories)}")

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

    # 格式转换示例：YOLO -> 统一格式 -> 新的YOLO
    logger.info("\n进行格式转换测试...")
    output_dir = data_dir / "output"
    output_dir.mkdir(exist_ok=True)

    write_result = handler.write(result.data, str(output_dir))

    if write_result.success:
        logger.info(f"成功写入到: {output_dir}")
        logger.info(f"生成文件数量: {len(list(output_dir.glob('*.txt')))}")
    else:
        logger.error(f"写入失败: {write_result.message}")

    logger.info("\n目标检测示例完成！\n")

def demo_segmentation():
    """实例分割数据演示"""
    log_ops = LoggingOperations()
    logger = log_ops.get_logger("yolo_demo_seg", level="INFO")

    logger.info("YOLO实例分割标注处理示例")
    logger.info("=" * 50)

    # 示例数据路径
    data_dir = project_root / "assets" / "test_data" / "seg" / "yolo"
    class_file = data_dir / "classes.txt"
    image_dir = data_dir / "images"
    label_dir = data_dir / "labels"

    if not data_dir.exists():
        logger.error(f"数据目录不存在: {data_dir}")
        logger.info("请确保已准备示例数据")
        return

    # 创建YOLO处理器
    logger.info(f"创建YOLO处理器:")
    logger.info(f"  标签目录: {label_dir}")
    logger.info(f"  类别文件: {class_file}")
    logger.info(f"  图片目录: {image_dir}")

    handler = YoloAnnotationHandler(
        label_dir=str(label_dir),
        class_file=str(class_file),
        image_dir=str(image_dir),
        strict_mode=True,
        logger=logger
    )

    # 读取标注数据
    logger.info("\n正在读取YOLO标注...")
    result = handler.read()

    if not result.success:
        logger.error(f"读取失败: {result.message}")
        if result.errors:
            for error in result.errors:
                logger.error(f"  - {error}")
        return

    # 显示标注类型检测结果
    logger.info(f"标注类型检测:")
    logger.info(f"  目标检测: {handler.is_det}")
    logger.info(f"  实例分割: {handler.is_seg}")

    logger.info(f"\n成功读取 {len(result.data.images)} 张图片的标注")
    logger.info(f"类别数量: {len(result.data.categories)}")

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
            if obj.segmentation:
                logger.info(f"    分割点数: {len(obj.segmentation.points)}")
                # 显示前3个点
                for j, (x, y) in enumerate(obj.segmentation.points[:3]):
                    logger.info(f"      点 {j+1}: x={x:.3f}, y={y:.3f}")

    # 格式转换示例：YOLO -> 统一格式 -> 新的YOLO
    logger.info("\n进行格式转换测试...")
    output_dir = data_dir / "output"
    output_dir.mkdir(exist_ok=True)

    write_result = handler.write(result.data, str(output_dir))

    if write_result.success:
        logger.info(f"成功写入到: {output_dir}")
        logger.info(f"生成文件数量: {len(list(output_dir.glob('*.txt')))}")
    else:
        logger.error(f"写入失败: {write_result.message}")

    logger.info("\n实例分割示例完成！\n")

def main():
    """主函数"""
    log_ops = LoggingOperations()
    logger = log_ops.get_logger("yolo_demo", level="INFO")

    logger.info("=" * 60)
    logger.info("YOLO标注格式处理示例")
    logger.info("=" * 60)

    # 运行目标检测演示
    demo_detection()

    # 运行实例分割演示
    demo_segmentation()

    logger.info("所有示例完成！")

if __name__ == "__main__":
    main()