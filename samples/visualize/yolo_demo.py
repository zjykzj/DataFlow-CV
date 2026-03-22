#!/usr/bin/env python3
"""
YOLO标注可视化示例

展示如何使用YOLOVisualizer可视化YOLO格式标注。
支持自动检测目标检测和实例分割格式。
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dataflow.visualize import YOLOVisualizer
from dataflow.util import LoggingOperations


def main():
    """主函数"""
    # 配置日志
    log_ops = LoggingOperations()
    logger = log_ops.get_logger("yolo_visualize_demo", level="INFO")

    # 示例数据路径（使用目标检测的YOLO测试数据）
    data_dir = project_root / "assets" / "test_data" / "det" / "yolo"
    image_dir = data_dir / "images"
    label_dir = data_dir / "labels"
    class_file = data_dir / "classes.txt"

    if not data_dir.exists():
        logger.error(f"数据目录不存在: {data_dir}")
        logger.info("请确保已准备示例数据")
        return

    logger.info("=" * 50)
    logger.info("YOLO标注可视化示例")
    logger.info("=" * 50)

    # 创建可视化器
    logger.info(f"创建YOLO可视化器:")
    logger.info(f"  标签目录: {label_dir}")
    logger.info(f"  图片目录: {image_dir}")
    logger.info(f"  类别文件: {class_file}")

    visualizer = YOLOVisualizer(
        label_dir=str(label_dir),
        image_dir=str(image_dir),
        class_file=str(class_file),
        is_show=True,      # 显示窗口
        is_save=True,      # 同时保存
        output_dir=data_dir / "visualized_output",
        strict_mode=True,
        logger=logger
    )

    # 执行可视化
    logger.info("\n开始可视化（按Enter键下一张，按q键退出）...")
    result = visualizer.visualize()

    if result.success:
        logger.info(f"可视化完成: {result.message}")
        logger.info(f"结果保存到: {data_dir / 'visualized_output'}")
    else:
        logger.error(f"可视化失败: {result.message}")
        if result.errors:
            for error in result.errors:
                logger.error(f"  - {error}")


if __name__ == "__main__":
    main()