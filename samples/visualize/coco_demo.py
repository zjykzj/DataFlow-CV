#!/usr/bin/env python3
"""
COCO标注可视化示例

展示如何使用COCOVisualizer可视化COCO格式标注。
支持RLE格式和多边形格式。
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dataflow.visualize import COCOVisualizer
from dataflow.util import LoggingOperations


def main():
    """主函数"""
    # 配置日志
    log_ops = LoggingOperations()
    logger = log_ops.get_logger("coco_visualize_demo", level="INFO")

    # 示例数据路径（使用目标检测的COCO测试数据）
    data_dir = project_root / "assets" / "test_data" / "det" / "coco"
    annotation_file = data_dir / "annotations.json"
    image_dir = data_dir / "images"

    if not annotation_file.exists():
        logger.error(f"标注文件不存在: {annotation_file}")
        logger.info("请确保已准备示例数据")
        return

    logger.info("=" * 50)
    logger.info("COCO标注可视化示例")
    logger.info("=" * 50)

    # 创建可视化器
    logger.info(f"创建COCO可视化器:")
    logger.info(f"  标注文件: {annotation_file}")
    logger.info(f"  图片目录: {image_dir}")

    visualizer = COCOVisualizer(
        annotation_file=str(annotation_file),
        image_dir=str(image_dir),
        is_show=True,      # 显示窗口
        is_save=False,     # 不保存
        strict_mode=True,
        logger=logger
    )

    # 执行可视化
    logger.info("\n开始可视化（按Enter键下一张，按q键退出）...")
    result = visualizer.visualize()

    if result.success:
        logger.info(f"可视化完成: {result.message}")
    else:
        logger.error(f"可视化失败: {result.message}")
        if result.errors:
            for error in result.errors:
                logger.error(f"  - {error}")


if __name__ == "__main__":
    main()