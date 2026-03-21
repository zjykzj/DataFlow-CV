#!/usr/bin/env python3
"""
LabelMe标注格式处理示例

展示如何使用LabelMeAnnotationHandler读取、处理和写入LabelMe格式标注。
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dataflow.label import LabelMeAnnotationHandler
from dataflow.util import LoggingOperations

def main():
    """主函数"""
    # 配置日志
    log_ops = LoggingOperations()
    logger = log_ops.get_logger("labelme_demo", level="INFO")

    # 示例数据路径（使用目标检测的LabelMe测试数据）
    data_dir = project_root / "assets" / "test_data" / "det" / "labelme"
    class_file = data_dir / "classes.txt"

    if not data_dir.exists():
        logger.error(f"数据目录不存在: {data_dir}")
        logger.info("请确保已准备示例数据")
        return

    logger.info("=" * 50)
    logger.info("LabelMe标注处理示例")
    logger.info("=" * 50)

    # 创建LabelMe处理器
    logger.info(f"创建LabelMe处理器，数据目录: {data_dir}")
    handler = LabelMeAnnotationHandler(
        label_dir=str(data_dir),
        class_file=str(class_file) if class_file.exists() else None,
        strict_mode=True,
        logger=logger
    )

    # 读取标注数据
    logger.info("正在读取LabelMe标注...")
    result = handler.read()

    if not result.success:
        logger.error(f"读取失败: {result.message}")
        if result.errors:
            for error in result.errors:
                logger.error(f"  - {error}")
        return

    logger.info(f"成功读取 {len(result.data.images)} 张图片的标注")
    logger.info(f"类别数量: {len(result.data.categories)}")

    # 显示部分信息
    for i, image_ann in enumerate(result.data.images[:3]):  # 只显示前3张
        logger.info(f"\n图片 {i+1}: {image_ann.image_id}")
        logger.info(f"  路径: {image_ann.image_path}")
        logger.info(f"  尺寸: {image_ann.width}x{image_ann.height}")
        logger.info(f"  对象数量: {len(image_ann.objects)}")

        for j, obj in enumerate(image_ann.objects[:2]):  # 每张图片显示前2个对象
            logger.info(f"    对象 {j+1}: {obj.class_name} (ID: {obj.class_id})")
            if obj.bbox:
                logger.info(f"      边界框: x={obj.bbox.x:.3f}, y={obj.bbox.y:.3f}, "
                          f"w={obj.bbox.width:.3f}, h={obj.bbox.height:.3f}")

    logger.info("\n示例完成！")

if __name__ == "__main__":
    main()