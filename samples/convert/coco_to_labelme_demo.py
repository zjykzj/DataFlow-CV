#!/usr/bin/env python3
"""
COCO到LabelMe格式转换示例

展示如何使用CocoAndLabelMeConverter将COCO格式标注转换为LabelMe格式。

使用方法：
    python coco_to_labelme_demo.py [--verbose]

示例：
    python coco_to_labelme_demo.py           # 普通模式
    python coco_to_labelme_demo.py --verbose # 详细日志模式
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
    parser = argparse.ArgumentParser(description="COCO到LabelMe格式转换示例")
    parser.add_argument("--verbose", action="store_true", help="启用详细日志模式")
    args = parser.parse_args()

    # 配置日志
    if args.verbose:
        log_ops = VerboseLoggingOperations()
        logger = log_ops.get_verbose_logger(
            name="coco_to_labelme_demo",
            verbose=True,
            log_dir=str(project_root / "logs")
        )
        logger.info("详细日志模式已启用")
    else:
        log_ops = LoggingOperations()
        logger = log_ops.get_logger("coco_to_labelme_demo", level="INFO")

    # 示例数据路径
    data_dir = project_root / "assets" / "test_data" / "det" / "coco"
    coco_file = data_dir / "annotations.json"
    output_dir = project_root / "samples" / "convert" / "output" / "coco_to_labelme"

    if not data_dir.exists():
        logger.error(f"数据目录不存在: {data_dir}")
        logger.info("请确保已准备示例数据")
        return

    if not coco_file.exists():
        logger.error(f"COCO文件不存在: {coco_file}")
        return

    logger.info("=" * 50)
    logger.info("COCO到LabelMe格式转换示例")
    logger.info("=" * 50)

    # 创建转换器（COCO→LabelMe方向）
    logger.info("创建COCO→LabelMe转换器")
    converter = CocoAndLabelMeConverter(
        source_to_target=True,  # COCO→LabelMe
        verbose=args.verbose,  # 详细日志模式
        strict_mode=True,
        logger=logger,
    )

    # 执行转换
    logger.info(f"执行转换:")
    logger.info(f"  源文件: {coco_file}")
    logger.info(f"  目标目录: {output_dir}")

    # COCO→LabelMe转换会自动从COCO JSON提取类别信息并生成classes.txt
    result = converter.convert(
        source_path=str(coco_file),
        target_path=str(output_dir),
        # image_dir可选，如果不提供会尝试从COCO JSON中提取图片路径
    )

    # 显示结果
    logger.info("\n转换结果:")
    logger.info(f"  成功: {result.success}")
    logger.info(f"  转换图片数: {result.num_images_converted}")
    logger.info(f"  转换对象数: {result.num_objects_converted}")

    if result.success:
        logger.info(f"  输出目录: {output_dir}")
        logger.info(f"  生成的文件:")
        json_files = list(output_dir.glob("*.json"))
        txt_files = list(output_dir.glob("*.txt"))

        for file in json_files:
            logger.info(f"    - {file.relative_to(output_dir)}")

        for file in txt_files:
            logger.info(f"    - {file.relative_to(output_dir)}")

        logger.info(f"  总共生成 {len(json_files)} 个LabelMe JSON文件")
        logger.info(f"  总共生成 {len(txt_files)} 个文本文件（包括classes.txt）")

        # 检查是否生成了classes.txt
        classes_file = output_dir / "classes.txt"
        if classes_file.exists():
            logger.info(f"  生成的类别文件: {classes_file}")
            try:
                with open(classes_file, "r", encoding="utf-8") as f:
                    classes = [line.strip() for line in f if line.strip()]
                    logger.info(f"  包含类别数: {len(classes)}")
            except Exception as e:
                logger.warning(f"  读取类别文件失败: {e}")
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
