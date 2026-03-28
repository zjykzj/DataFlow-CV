#!/usr/bin/env python3
"""
LabelMe到COCO格式转换示例

展示如何使用CocoAndLabelMeConverter将LabelMe格式标注转换为COCO格式。

使用方法：
    python labelme_to_coco_demo.py [--verbose]

示例：
    python labelme_to_coco_demo.py           # 普通模式
    python labelme_to_coco_demo.py --verbose # 详细日志模式
"""

import argparse
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dataflow.convert import CocoAndLabelMeConverter
from dataflow.util import LoggingOperations, VerboseLoggingOperations


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="LabelMe到COCO格式转换示例")
    parser.add_argument("--verbose", action="store_true", help="启用详细日志模式")
    args = parser.parse_args()

    # 配置日志
    if args.verbose:
        log_ops = VerboseLoggingOperations()
        logger = log_ops.get_verbose_logger(
            name="labelme_to_coco_demo",
            verbose=True,
            log_dir=str(project_root / "logs")
        )
        logger.info("详细日志模式已启用")
    else:
        log_ops = LoggingOperations()
        logger = log_ops.get_logger("labelme_to_coco_demo", level="INFO")

    # 示例数据路径
    data_dir = project_root / "assets" / "test_data" / "det" / "labelme"
    class_file = data_dir / "classes.txt"
    output_dir = project_root / "samples" / "convert" / "output" / "labelme_to_coco"
    output_dir.mkdir(parents=True, exist_ok=True)
    coco_file = output_dir / "annotations.json"

    if not data_dir.exists():
        logger.error(f"数据目录不存在: {data_dir}")
        logger.info("请确保已准备示例数据")
        return

    if not class_file.exists():
        logger.error(f"类别文件不存在: {class_file}")
        return

    logger.info("=" * 50)
    logger.info("LabelMe到COCO格式转换示例")
    logger.info("=" * 50)

    # 创建转换器（LabelMe→COCO方向）
    logger.info("创建LabelMe→COCO转换器")
    converter = CocoAndLabelMeConverter(
        source_to_target=False,  # LabelMe→COCO
        verbose=args.verbose,  # 详细日志模式
        strict_mode=True,
        logger=logger,
    )

    # 执行转换（不使用RLE，保持多边形格式）
    logger.info(f"执行转换 (do_rle=False):")
    logger.info(f"  源目录: {data_dir}")
    logger.info(f"  目标文件: {coco_file}")
    logger.info(f"  类别文件: {class_file}")

    result = converter.convert(
        source_path=str(data_dir),
        target_path=str(coco_file),
        class_file=str(class_file),
        do_rle=False,  # 不使用RLE，输出多边形点列表
    )

    # 显示结果
    logger.info("\n转换结果:")
    logger.info(f"  成功: {result.success}")
    logger.info(f"  转换图片数: {result.num_images_converted}")
    logger.info(f"  转换对象数: {result.num_objects_converted}")

    if result.success:
        logger.info(f"  输出文件: {coco_file}")
        logger.info(
            f"  文件大小: {coco_file.stat().st_size if coco_file.exists() else 0} 字节"
        )

        # 显示COCO数据集信息
        if "dataset_info" in result.metadata:
            info = result.metadata["dataset_info"]
            logger.info(f"  COCO数据集信息:")
            logger.info(f"    - 描述: {info.get('description', 'N/A')}")
            logger.info(f"    - 版本: {info.get('version', 'N/A')}")
            logger.info(f"    - 贡献者: {info.get('contributor', 'N/A')}")
            logger.info(f"    - 创建日期: {info.get('year', 'N/A')}")

        if "categories" in result.metadata:
            categories = result.metadata["categories"]
            logger.info(f"  生成类别数: {len(categories)}")
            for cat_id, cat_name in categories.items():
                logger.info(f"    - ID {cat_id}: {cat_name}")
    else:
        logger.error(f"  错误数: {len(result.errors)}")
        for error in result.errors:
            logger.error(f"    - {error}")

    if result.warnings:
        logger.warning(f"  警告数: {len(result.warnings)}")
        for warning in result.warnings:
            logger.warning(f"    - {warning}")

    logger.info("\n示例完成！")


if __name__ == "__main__":
    main()
