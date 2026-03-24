#!/usr/bin/env python3
"""
LabelMe标注可视化示例

展示如何使用LabelMeVisualizer可视化LabelMe格式标注。
支持目标检测和实例分割标注的显示和保存。

使用方法：
    python labelme_demo.py [--task {det,seg}] [--verbose]

示例：
    python labelme_demo.py                     # 可视化目标检测标注（默认）
    python labelme_demo.py --task seg          # 可视化实例分割标注
    python labelme_demo.py --verbose           # 启用详细日志模式
    python labelme_demo.py --task seg --verbose # 实例分割+详细日志
"""

import argparse
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dataflow.util import LoggingOperations, VerboseLoggingOperations
from dataflow.visualize import LabelMeVisualizer


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="LabelMe标注可视化示例")
    parser.add_argument(
        "--task",
        choices=["det", "seg"],
        default="det",
        help="任务类型: det=目标检测, seg=实例分割 (默认: det)",
    )
    parser.add_argument("--verbose", action="store_true", help="启用详细日志模式")
    args = parser.parse_args()

    # 配置日志
    if args.verbose:
        log_ops = VerboseLoggingOperations()
        logger = log_ops.get_logger("labelme_visualize_demo", level="INFO")
        logger.info("详细日志模式已启用")
    else:
        log_ops = LoggingOperations()
        logger = log_ops.get_logger("labelme_visualize_demo", level="INFO")

    # 根据任务类型选择数据路径
    if args.task == "det":
        task_name = "目标检测"
        data_dir = project_root / "assets" / "test_data" / "det" / "labelme"
    else:  # args.task == "seg"
        task_name = "实例分割"
        data_dir = project_root / "assets" / "test_data" / "seg" / "labelme"

    image_dir = data_dir  # LabelMe格式中JSON文件和图片在同一目录
    class_file = data_dir / "classes.txt"

    if not data_dir.exists():
        logger.error(f"数据目录不存在: {data_dir}")
        logger.info("请确保已准备示例数据")
        return

    logger.info("=" * 50)
    logger.info(f"LabelMe {task_name}标注可视化示例")
    logger.info("=" * 50)

    # 创建可视化器
    logger.info(f"创建LabelMe可视化器:")
    logger.info(f"  任务类型: {task_name}")
    logger.info(f"  标签目录: {data_dir}")
    logger.info(f"  图片目录: {image_dir}")
    logger.info(f"  类别文件: {class_file if class_file.exists() else '无'}")

    visualizer = LabelMeVisualizer(
        label_dir=str(data_dir),
        image_dir=str(image_dir),
        class_file=str(class_file) if class_file.exists() else None,
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
