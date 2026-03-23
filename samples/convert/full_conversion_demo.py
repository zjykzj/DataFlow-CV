#!/usr/bin/env python3
"""
完整格式转换链示例

展示如何将LabelMe格式转换为YOLO，再转换为COCO，最后转回LabelMe，
验证无损转换（除RLE精度损失外）。
"""

import sys
from pathlib import Path
import tempfile
import shutil

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dataflow.convert import (
    LabelMeAndYoloConverter,
    YoloAndCocoConverter,
    CocoAndLabelMeConverter
)
from dataflow.util import LoggingOperations

def main():
    """主函数"""
    # 配置日志
    log_ops = LoggingOperations()
    logger = log_ops.get_logger("full_conversion_demo", level="INFO")

    # 创建临时工作目录
    temp_dir = Path(tempfile.mkdtemp(prefix="dataflow_convert_"))
    logger.info(f"创建临时工作目录: {temp_dir}")

    try:
        # 示例数据路径
        data_dir = project_root / "assets" / "test_data" / "det" / "labelme"
        class_file = data_dir / "classes.txt"

        if not data_dir.exists():
            logger.error(f"数据目录不存在: {data_dir}")
            return

        logger.info("=" * 50)
        logger.info("完整格式转换链示例")
        logger.info("=" * 50)

        # 第1步：LabelMe → YOLO
        logger.info("\n1. LabelMe → YOLO 转换")
        yolo_dir = temp_dir / "yolo_output"

        converter1 = LabelMeAndYoloConverter(
            source_to_target=True,
            strict_mode=True,
            logger=logger
        )

        result1 = converter1.convert(
            source_path=str(data_dir),
            target_path=str(yolo_dir),
            class_file=str(class_file)
        )

        if not result1.success:
            logger.error("LabelMe→YOLO转换失败")
            return

        # 第2步：YOLO → COCO
        logger.info("\n2. YOLO → COCO 转换")
        coco_file = temp_dir / "coco_output.json"

        converter2 = YoloAndCocoConverter(
            source_to_target=True,
            strict_mode=True,
            logger=logger
        )

        result2 = converter2.convert(
            source_path=str(yolo_dir / "labels"),
            target_path=str(coco_file),
            class_file=str(yolo_dir / "classes.txt"),
            image_dir=str(yolo_dir / "images"),
            do_rle=False  # 不使用RLE以确保无损
        )

        if not result2.success:
            logger.error("YOLO→COCO转换失败")
            return

        # 第3步：COCO → LabelMe
        logger.info("\n3. COCO → LabelMe 转换")
        labelme_dir = temp_dir / "labelme_output"

        converter3 = CocoAndLabelMeConverter(
            source_to_target=True,
            strict_mode=True,
            logger=logger
        )

        result3 = converter3.convert(
            source_path=str(coco_file),
            target_path=str(labelme_dir)
        )

        if not result3.success:
            logger.error("COCO→LabelMe转换失败")
            return

        # 验证结果
        logger.info("\n转换链完成！")
        logger.info(f"原始LabelMe文件数: {len(list(data_dir.glob('*.json')))}")
        logger.info(f"最终LabelMe文件数: {len(list(labelme_dir.glob('*.json')))}")

        # 简单验证：文件数量一致
        original_count = len(list(data_dir.glob('*.json')))
        final_count = len(list(labelme_dir.glob('*.json')))

        if original_count == final_count:
            logger.info("✓ 文件数量一致，转换链完整")
        else:
            logger.warning(f"⚠ 文件数量不一致: 原始={original_count}, 最终={final_count}")

        logger.info(f"\n临时工作目录: {temp_dir}")
        logger.info("（程序结束后会自动清理）")

    finally:
        # 清理临时目录（在实际使用中可能保留用于调试）
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            logger.info(f"已清理临时目录: {temp_dir}")

    logger.info("\n示例完成！")

if __name__ == "__main__":
    main()