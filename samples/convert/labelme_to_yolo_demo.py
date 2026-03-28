#!/usr/bin/env python3
"""
LabelMe到YOLO格式转换示例

展示如何使用LabelMeAndYoloConverter将LabelMe格式标注转换为YOLO格式。

使用方法：
    python labelme_to_yolo_demo.py [--verbose]

示例：
    python labelme_to_yolo_demo.py           # 普通模式
    python labelme_to_yolo_demo.py --verbose # 详细日志模式
"""

import argparse
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dataflow.convert import LabelMeAndYoloConverter
from dataflow.util import LoggingOperations, VerboseLoggingOperations


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="LabelMe到YOLO格式转换示例")
    parser.add_argument("--verbose", action="store_true", help="启用详细日志模式")
    args = parser.parse_args()

    # 配置日志
    if args.verbose:
        log_ops = VerboseLoggingOperations()
        logger = log_ops.get_verbose_logger(
            name="labelme_to_yolo_demo",
            verbose=True,
            log_dir=str(project_root / "logs")
        )
        logger.info("详细日志模式已启用")
    else:
        log_ops = LoggingOperations()
        logger = log_ops.get_logger("labelme_to_yolo_demo", level="INFO")

    # 示例数据路径
    data_dir = project_root / "assets" / "test_data" / "det" / "labelme"
    class_file = data_dir / "classes.txt"
    output_dir = project_root / "samples" / "convert" / "output" / "labelme_to_yolo"

    if not data_dir.exists():
        logger.error(f"数据目录不存在: {data_dir}")
        logger.info("请确保已准备示例数据")
        return

    logger.info("=" * 50)
    logger.info("LabelMe到YOLO格式转换示例")
    logger.info("=" * 50)

    # 创建转换器（LabelMe→YOLO方向）
    logger.info("创建LabelMe→YOLO转换器")
    converter = LabelMeAndYoloConverter(
        source_to_target=True,  # LabelMe→YOLO
        verbose=args.verbose,  # 详细日志模式
        strict_mode=True,
        logger=logger,
    )

    # 执行转换
    logger.info(f"执行转换:")
    logger.info(f"  源目录: {data_dir}")
    logger.info(f"  目标目录: {output_dir}")
    logger.info(f"  类别文件: {class_file}")

    result = converter.convert(
        source_path=str(data_dir),
        target_path=str(output_dir),
        class_file=str(class_file),
    )

    # 显示结果
    logger.info("\n转换结果:")
    logger.info(f"  成功: {result.success}")
    logger.info(f"  转换图片数: {result.num_images_converted}")
    logger.info(f"  转换对象数: {result.num_objects_converted}")

    if result.success:
        logger.info(f"  输出目录: {output_dir}")
        logger.info(f"  生成的文件:")
        for file in output_dir.rglob("*"):
            if file.is_file():
                logger.info(f"    - {file.relative_to(output_dir)}")
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
