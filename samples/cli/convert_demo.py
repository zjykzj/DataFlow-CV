#!/usr/bin/env python3
"""
CLI格式转换功能示例脚本

展示如何使用dataflow-cv convert命令进行六种标签格式转换：
1. YOLO → COCO
2. YOLO → LabelMe
3. COCO → YOLO
4. COCO → LabelMe
5. LabelMe → YOLO
6. LabelMe → COCO

使用方法：
1. 在项目根目录运行：python samples/cli/convert_demo.py
2. 或者直接运行：./samples/cli/convert_demo.py
"""

import subprocess
import sys
import json
from pathlib import Path

# 添加项目根目录到PATH，以便导入dataflow模块
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def run_cli_command(cmd_args):
    """运行CLI命令并返回结果"""
    import sys
    cmd = [sys.executable, "-m", "dataflow.cli.main"] + cmd_args
    print(f"执行命令: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd,
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
        print("错误: 未找到Python解释器")
        return False
    except Exception as e:
        print(f"错误: 执行命令时发生异常: {e}")
        return False

def demo_yolo_to_coco():
    """演示YOLO转COCO格式"""
    print("\n" + "="*60)
    print("演示YOLO转COCO格式")
    print("="*60)

    yolo_dir = project_root / "assets" / "test_data" / "seg" / "yolo"
    class_file = yolo_dir / "classes.txt"
    image_dir = project_root / "assets" / "test_data" / "seg" / "yolo" / "images"
    output_file = project_root / "temp_output" / "convert" / "yolo_to_coco.json"

    # 清理旧输出文件
    if output_file.exists():
        output_file.unlink()

    cmd = [
        "convert", "yolo2coco",
        str(yolo_dir),
        str(output_file),
        "--class-file", str(class_file),
        "--image-dir", str(image_dir),
        "--verbose"
    ]

    success = run_cli_command(cmd)
    if success:
        print(f"✓ YOLO→COCO转换成功，结果保存在: {output_file}")
        # 验证输出文件
        if output_file.exists():
            with open(output_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                image_count = len(data.get("images", []))
                annotation_count = len(data.get("annotations", []))
                category_count = len(data.get("categories", []))
                print(f"  包含 {image_count} 张图像, {annotation_count} 个标注, {category_count} 个类别")
        return True
    else:
        print("✗ YOLO→COCO转换失败")
        return False

def demo_yolo_to_labelme():
    """演示YOLO转LabelMe格式"""
    print("\n" + "="*60)
    print("演示YOLO转LabelMe格式")
    print("="*60)

    yolo_dir = project_root / "assets" / "test_data" / "seg" / "yolo"
    class_file = yolo_dir / "classes.txt"
    image_dir = project_root / "assets" / "test_data" / "seg" / "yolo" / "images"
    output_dir = project_root / "temp_output" / "convert" / "yolo_to_labelme"

    # 清理旧输出目录
    if output_dir.exists():
        import shutil
        shutil.rmtree(output_dir)

    cmd = [
        "convert", "yolo2labelme",
        str(yolo_dir),
        str(output_dir),
        "--class-file", str(class_file),
        "--image-dir", str(image_dir),
        "--verbose"
    ]

    success = run_cli_command(cmd)
    if success:
        print(f"✓ YOLO→LabelMe转换成功，结果保存在: {output_dir}")
        # 统计生成的文件
        json_files = list(output_dir.glob("*.json"))
        print(f"  生成了 {len(json_files)} 个LabelMe标注文件")
        return True
    else:
        print("✗ YOLO→LabelMe转换失败")
        return False

def demo_coco_to_yolo():
    """演示COCO转YOLO格式"""
    print("\n" + "="*60)
    print("演示COCO转YOLO格式")
    print("="*60)

    coco_file = project_root / "assets" / "test_data" / "seg" / "coco" / "annotations.json"
    image_dir = project_root / "assets" / "test_data" / "seg" / "coco" / "images"
    output_dir = project_root / "temp_output" / "convert" / "coco_to_yolo"

    # 清理旧输出目录
    if output_dir.exists():
        import shutil
        shutil.rmtree(output_dir)

    cmd = [
        "convert", "coco2yolo",
        str(coco_file),
        str(output_dir),
        "--image-dir", str(image_dir),
        "--verbose"
    ]

    success = run_cli_command(cmd)
    if success:
        print(f"✓ COCO→YOLO转换成功，结果保存在: {output_dir}")
        # 统计生成的文件
        txt_files = list(output_dir.glob("*.txt"))
        print(f"  生成了 {len(txt_files)} 个YOLO标注文件")
        return True
    else:
        print("✗ COCO→YOLO转换失败")
        return False

def demo_coco_to_labelme():
    """演示COCO转LabelMe格式"""
    print("\n" + "="*60)
    print("演示COCO转LabelMe格式")
    print("="*60)

    coco_file = project_root / "assets" / "test_data" / "seg" / "coco" / "annotations.json"
    image_dir = project_root / "assets" / "test_data" / "seg" / "coco" / "images"
    output_dir = project_root / "temp_output" / "convert" / "coco_to_labelme"

    # 清理旧输出目录
    if output_dir.exists():
        import shutil
        shutil.rmtree(output_dir)

    cmd = [
        "convert", "coco2labelme",
        str(coco_file),
        str(output_dir),
        "--image-dir", str(image_dir),
        "--verbose"
    ]

    success = run_cli_command(cmd)
    if success:
        print(f"✓ COCO→LabelMe转换成功，结果保存在: {output_dir}")
        # 统计生成的文件
        json_files = list(output_dir.glob("*.json"))
        print(f"  生成了 {len(json_files)} 个LabelMe标注文件")
        return True
    else:
        print("✗ COCO→LabelMe转换失败")
        return False

def demo_labelme_to_yolo():
    """演示LabelMe转YOLO格式"""
    print("\n" + "="*60)
    print("演示LabelMe转YOLO格式")
    print("="*60)

    labelme_dir = project_root / "assets" / "test_data" / "seg" / "labelme"
    output_dir = project_root / "temp_output" / "convert" / "labelme_to_yolo"

    # 清理旧输出目录
    if output_dir.exists():
        import shutil
        shutil.rmtree(output_dir)

    cmd = [
        "convert", "labelme2yolo",
        str(labelme_dir),
        str(output_dir),
        "--verbose"
    ]

    success = run_cli_command(cmd)
    if success:
        print(f"✓ LabelMe→YOLO转换成功，结果保存在: {output_dir}")
        # 统计生成的文件
        txt_files = list(output_dir.glob("*.txt"))
        print(f"  生成了 {len(txt_files)} 个YOLO标注文件")
        return True
    else:
        print("✗ LabelMe→YOLO转换失败")
        return False

def demo_labelme_to_coco():
    """演示LabelMe转COCO格式"""
    print("\n" + "="*60)
    print("演示LabelMe转COCO格式")
    print("="*60)

    labelme_dir = project_root / "assets" / "test_data" / "seg" / "labelme"
    output_file = project_root / "temp_output" / "convert" / "labelme_to_coco.json"

    # 清理旧输出文件
    if output_file.exists():
        output_file.unlink()

    cmd = [
        "convert", "labelme2coco",
        str(labelme_dir),
        str(output_file),
        "--verbose"
    ]

    success = run_cli_command(cmd)
    if success:
        print(f"✓ LabelMe→COCO转换成功，结果保存在: {output_file}")
        # 验证输出文件
        if output_file.exists():
            with open(output_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                image_count = len(data.get("images", []))
                annotation_count = len(data.get("annotations", []))
                category_count = len(data.get("categories", []))
                print(f"  包含 {image_count} 张图像, {annotation_count} 个标注, {category_count} 个类别")
        return True
    else:
        print("✗ LabelMe→COCO转换失败")
        return False

def main():
    """主函数"""
    print("DataFlow-CV CLI格式转换功能演示")
    print("="*60)

    # 检查测试数据是否存在
    test_data_root = project_root / "assets" / "test_data" / "seg"
    if not test_data_root.exists():
        print(f"错误: 测试数据目录不存在: {test_data_root}")
        print("请确保在项目根目录运行此脚本")
        return 1

    # 创建临时输出目录
    temp_root = project_root / "temp_output" / "convert"
    temp_root.mkdir(parents=True, exist_ok=True)

    successes = []

    # 演示六种格式转换
    successes.append(demo_yolo_to_coco())
    successes.append(demo_yolo_to_labelme())
    successes.append(demo_coco_to_yolo())
    successes.append(demo_coco_to_labelme())
    successes.append(demo_labelme_to_yolo())
    successes.append(demo_labelme_to_coco())

    # 总结
    print("\n" + "="*60)
    print("演示总结")
    print("="*60)
    total = len(successes)
    passed = sum(successes)
    print(f"总共演示了 {total} 种格式转换")
    print(f"成功: {passed}, 失败: {total - passed}")

    if all(successes):
        print("✓ 所有演示都成功完成!")
        print(f"转换结果保存在: {project_root / 'temp_output' / 'convert'}")
        print("\n您可以使用以下命令手动测试:")
        print("  dataflow-cv convert yolo2coco assets/test_data/seg/yolo ./output.json --class-file assets/test_data/seg/yolo/classes.txt --image-dir assets/test_data/seg/yolo/images")
        print("  dataflow-cv convert yolo2labelme assets/test_data/seg/yolo ./output --class-file assets/test_data/seg/yolo/classes.txt --image-dir assets/test_data/seg/yolo/images")
        print("  dataflow-cv convert coco2yolo assets/test_data/seg/coco/annotations.json ./output --image-dir assets/test_data/seg/coco/images")
        print("  dataflow-cv convert coco2labelme assets/test_data/seg/coco/annotations.json ./output --image-dir assets/test_data/seg/coco/images")
        print("  dataflow-cv convert labelme2yolo assets/test_data/seg/labelme ./output")
        print("  dataflow-cv convert labelme2coco assets/test_data/seg/labelme ./output.json")
        return 0
    else:
        print("✗ 部分演示失败，请检查错误信息")
        return 1

if __name__ == "__main__":
    sys.exit(main())