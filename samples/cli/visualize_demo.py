#!/usr/bin/env python3
"""
CLI可视化功能示例脚本

展示如何使用dataflow-cv visualize命令可视化三种标签格式：
1. YOLO格式可视化
2. COCO格式可视化
3. LabelMe格式可视化

使用方法：
1. 在项目根目录运行：python samples/cli/visualize_demo.py
2. 或者直接运行：./samples/cli/visualize_demo.py
"""

import subprocess
import sys
from pathlib import Path

# 添加项目根目录到PATH，以便导入dataflow模块
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def run_cli_command(cmd_args):
    """运行CLI命令并返回结果"""
    print(f"执行命令: dataflow-cv {' '.join(cmd_args)}")
    try:
        result = subprocess.run(
            ["dataflow-cv"] + cmd_args,
            capture_output=True,
            text=True,
            cwd=project_root,
            timeout=30,
        )
        print(f"退出码: {result.returncode}")
        if result.stdout:
            print(f"标准输出:\n{result.stdout}")
        if result.stderr:
            print(f"标准错误:\n{result.stderr}")
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("错误: 命令执行超时")
        return False
    except FileNotFoundError:
        print("错误: 未找到dataflow-cv命令，请先安装包: pip install -e .")
        return False
    except Exception as e:
        print(f"错误: 执行命令时发生异常: {e}")
        return False

def demo_yolo_visualization():
    """演示YOLO格式可视化"""
    print("\n" + "="*60)
    print("演示YOLO格式可视化")
    print("="*60)

    yolo_dir = project_root / "assets" / "test_data" / "seg" / "yolo"
    class_file = yolo_dir / "classes.txt"
    output_dir = project_root / "temp_output" / "visualize" / "yolo"

    # 清理旧输出目录
    if output_dir.exists():
        import shutil
        shutil.rmtree(output_dir)

    cmd = [
        "visualize", "yolo",
        str(yolo_dir),
        "--class-file", str(class_file),
        "--output-dir", str(output_dir),
        "--save",
        "--verbose"
    ]

    success = run_cli_command(cmd)
    if success:
        print(f"✓ YOLO可视化成功，结果保存在: {output_dir}")
        # 统计生成的文件
        image_files = list(output_dir.glob("*.jpg")) + list(output_dir.glob("*.png"))
        print(f"  生成了 {len(image_files)} 张可视化图像")
    else:
        print("✗ YOLO可视化失败")
    return success

def demo_coco_visualization():
    """演示COCO格式可视化"""
    print("\n" + "="*60)
    print("演示COCO格式可视化")
    print("="*60)

    coco_file = project_root / "assets" / "test_data" / "seg" / "coco" / "annotations.json"
    image_dir = project_root / "assets" / "test_data" / "seg" / "coco" / "images"
    output_dir = project_root / "temp_output" / "visualize" / "coco"

    # 清理旧输出目录
    if output_dir.exists():
        import shutil
        shutil.rmtree(output_dir)

    cmd = [
        "visualize", "coco",
        str(coco_file),
        "--image-dir", str(image_dir),
        "--output-dir", str(output_dir),
        "--save",
        "--verbose"
    ]

    success = run_cli_command(cmd)
    if success:
        print(f"✓ COCO可视化成功，结果保存在: {output_dir}")
        image_files = list(output_dir.glob("*.jpg")) + list(output_dir.glob("*.png"))
        print(f"  生成了 {len(image_files)} 张可视化图像")
    else:
        print("✗ COCO可视化失败")
    return success

def demo_labelme_visualization():
    """演示LabelMe格式可视化"""
    print("\n" + "="*60)
    print("演示LabelMe格式可视化")
    print("="*60)

    labelme_dir = project_root / "assets" / "test_data" / "seg" / "labelme"
    output_dir = project_root / "temp_output" / "visualize" / "labelme"

    # 清理旧输出目录
    if output_dir.exists():
        import shutil
        shutil.rmtree(output_dir)

    cmd = [
        "visualize", "labelme",
        str(labelme_dir),
        "--output-dir", str(output_dir),
        "--save",
        "--verbose"
    ]

    success = run_cli_command(cmd)
    if success:
        print(f"✓ LabelMe可视化成功，结果保存在: {output_dir}")
        image_files = list(output_dir.glob("*.jpg")) + list(output_dir.glob("*.png"))
        print(f"  生成了 {len(image_files)} 张可视化图像")
    else:
        print("✗ LabelMe可视化失败")
    return success

def main():
    """主函数"""
    print("DataFlow-CV CLI可视化功能演示")
    print("="*60)

    # 检查测试数据是否存在
    test_data_root = project_root / "assets" / "test_data" / "seg"
    if not test_data_root.exists():
        print(f"错误: 测试数据目录不存在: {test_data_root}")
        print("请确保在项目根目录运行此脚本")
        return 1

    # 创建临时输出目录
    temp_root = project_root / "temp_output" / "visualize"
    temp_root.mkdir(parents=True, exist_ok=True)

    successes = []

    # 演示三种格式的可视化
    successes.append(demo_yolo_visualization())
    successes.append(demo_coco_visualization())
    successes.append(demo_labelme_visualization())

    # 总结
    print("\n" + "="*60)
    print("演示总结")
    print("="*60)
    total = len(successes)
    passed = sum(successes)
    print(f"总共演示了 {total} 种格式的可视化")
    print(f"成功: {passed}, 失败: {total - passed}")

    if all(successes):
        print("✓ 所有演示都成功完成!")
        print(f"可视化结果保存在: {project_root / 'temp_output' / 'visualize'}")
        print("\n您可以使用以下命令手动测试:")
        print("  dataflow-cv visualize yolo assets/test_data/seg/yolo --class-file assets/test_data/seg/yolo/classes.txt --output-dir ./output --save")
        print("  dataflow-cv visualize coco assets/test_data/seg/coco/annotations.json --image-dir assets/test_data/seg/coco/images --output-dir ./output --save")
        print("  dataflow-cv visualize labelme assets/test_data/seg/labelme --output-dir ./output --save")
        return 0
    else:
        print("✗ 部分演示失败，请检查错误信息")
        return 1

if __name__ == "__main__":
    sys.exit(main())