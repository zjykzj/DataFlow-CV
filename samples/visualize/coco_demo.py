#!/usr/bin/env python3
"""
COCO标注可视化示例

展示如何使用COCOVisualizer可视化COCO格式标注。
支持目标检测和实例分割标注，以及多边形格式和RLE格式。

使用方法：
    python coco_demo.py [--task {det,seg}] [--format {polygon,rle}] [--verbose]

示例：
    python coco_demo.py                      # 可视化目标检测标注（多边形格式，默认）
    python coco_demo.py --task seg           # 可视化实例分割标注（多边形格式）
    python coco_demo.py --task seg --format rle  # 可视化实例分割标注（RLE格式）
    python coco_demo.py --verbose            # 启用详细日志模式
    python coco_demo.py --task seg --verbose # 实例分割+详细日志

注意：目标检测任务仅支持多边形格式。
"""

import argparse
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dataflow.util import LoggingOperations, VerboseLoggingOperations
from dataflow.visualize import COCOVisualizer


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="COCO标注可视化示例")
    parser.add_argument(
        "--task",
        choices=["det", "seg"],
        default="det",
        help="任务类型: det=目标检测, seg=实例分割 (默认: det)",
    )
    parser.add_argument(
        "--format",
        choices=["polygon", "rle"],
        default="polygon",
        help="标注格式: polygon=多边形格式, rle=RLE格式 (默认: polygon)",
    )
    parser.add_argument("--verbose", action="store_true", help="启用详细日志模式")
    args = parser.parse_args()

    # 参数验证
    if args.task == "det" and args.format == "rle":
        print("错误: 目标检测任务仅支持多边形格式")
        print("请使用: python coco_demo.py --task det --format polygon")
        return

    # 配置日志
    if args.verbose:
        log_ops = VerboseLoggingOperations()
        logger = log_ops.get_logger("coco_visualize_demo", level="INFO")
        logger.info("详细日志模式已启用")
    else:
        log_ops = LoggingOperations()
        logger = log_ops.get_logger("coco_visualize_demo", level="INFO")

    # 根据任务类型和格式选择数据路径
    if args.task == "det":
        task_name = "目标检测"
        data_dir = project_root / "assets" / "test_data" / "det" / "coco"
        annotation_file = data_dir / "annotations.json"
        format_name = "多边形格式"
    else:  # args.task == "seg"
        task_name = "实例分割"
        data_dir = project_root / "assets" / "test_data" / "seg" / "coco"
        if args.format == "polygon":
            annotation_file = data_dir / "annotations.json"
            format_name = "多边形格式"
        else:  # args.format == "rle"
            annotation_file = data_dir / "annotations-rle.json"
            format_name = "RLE格式"

    image_dir = data_dir / "images"

    if not annotation_file.exists():
        logger.error(f"标注文件不存在: {annotation_file}")
        logger.info("请确保已准备示例数据")
        return

    logger.info("=" * 50)
    logger.info(f"COCO {task_name}标注可视化示例 ({format_name})")
    logger.info("=" * 50)

    # 创建可视化器
    logger.info(f"创建COCO可视化器:")
    logger.info(f"  任务类型: {task_name}")
    logger.info(f"  标注格式: {format_name}")
    logger.info(f"  标注文件: {annotation_file}")
    logger.info(f"  图片目录: {image_dir}")

    visualizer = COCOVisualizer(
        annotation_file=str(annotation_file),
        image_dir=str(image_dir),
        verbose=args.verbose,  # 详细日志模式
        is_show=True,  # 显示窗口
        is_save=False,  # 不保存
        strict_mode=True,
        logger=logger,
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
