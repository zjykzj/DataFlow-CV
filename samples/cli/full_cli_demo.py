#!/usr/bin/env python3
"""
CLI完整工作流示例脚本

展示完整的计算机视觉数据处理工作流：
1. 可视化原始YOLO标签
2. 将YOLO标签转换为COCO格式
3. 可视化转换后的COCO标签
4. 将COCO标签转换回YOLO格式
5. 验证转换的准确性

使用方法：
1. 在项目根目录运行：python samples/cli/full_cli_demo.py
2. 或者直接运行：./samples/cli/full_cli_demo.py
"""

import subprocess
import sys
import json
import shutil
from pathlib import Path

# 添加项目根目录到PATH，以便导入dataflow模块
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def run_cli_command(cmd_args, check=True):
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

        if check and result.returncode != 0:
            print(f"命令执行失败，退出码: {result.returncode}")
            return False
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

def cleanup_directory(dir_path):
    """清理目录"""
    if dir_path.exists():
        shutil.rmtree(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)

def step1_visualize_yolo():
    """步骤1: 可视化原始YOLO标签"""
    print("\n" + "="*60)
    print("步骤1: 可视化原始YOLO标签")
    print("="*60)

    yolo_dir = project_root / "assets" / "test_data" / "seg" / "yolo"
    class_file = yolo_dir / "classes.txt"
    image_dir = yolo_dir / "images"
    output_dir = project_root / "temp_output" / "full_demo" / "step1_yolo_visualization"

    cleanup_directory(output_dir)

    cmd = [
        "visualize", "yolo",
        str(image_dir),  # image_dir (positional)
        str(yolo_dir / "labels"),  # label_dir (positional)
        str(class_file),  # class_file (positional)
        "--save",
        str(output_dir),  # output directory for --save
        "--verbose"
    ]

    success = run_cli_command(cmd)
    if success:
        image_files = list(output_dir.glob("*.jpg")) + list(output_dir.glob("*.png"))
        print(f"✓ 步骤1完成: 生成了 {len(image_files)} 张YOLO可视化图像")
        return True, output_dir
    else:
        print("✗ 步骤1失败")
        return False, None

def step2_yolo_to_coco():
    """步骤2: YOLO转COCO格式"""
    print("\n" + "="*60)
    print("步骤2: YOLO转COCO格式")
    print("="*60)

    yolo_dir = project_root / "assets" / "test_data" / "seg" / "yolo"
    class_file = yolo_dir / "classes.txt"
    image_dir = yolo_dir / "images"
    output_file = project_root / "temp_output" / "full_demo" / "step2_coco_annotations.json"

    if output_file.exists():
        output_file.unlink()

    cmd = [
        "convert", "yolo2coco",
        str(yolo_dir / "labels"),  # YOLO label directory
        str(output_file),
        "--class-file", str(class_file),
        "--image-dir", str(image_dir),
        "--verbose"
    ]

    success = run_cli_command(cmd)
    if success:
        # 验证输出文件
        if output_file.exists():
            with open(output_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                image_count = len(data.get("images", []))
                annotation_count = len(data.get("annotations", []))
                category_count = len(data.get("categories", []))
                print(f"✓ 步骤2完成: 转换成功，包含 {image_count} 张图像, {annotation_count} 个标注, {category_count} 个类别")
        return True, output_file
    else:
        print("✗ 步骤2失败")
        return False, None

def step3_visualize_coco(coco_file):
    """步骤3: 可视化COCO标签"""
    print("\n" + "="*60)
    print("步骤3: 可视化COCO标签")
    print("="*60)

    image_dir = project_root / "assets" / "test_data" / "seg" / "coco" / "images"
    output_dir = project_root / "temp_output" / "full_demo" / "step3_coco_visualization"

    cleanup_directory(output_dir)

    cmd = [
        "visualize", "coco",
        str(image_dir),  # image_dir (positional)
        str(coco_file),  # coco_file (positional)
        "--save",
        str(output_dir),  # output directory for --save
        "--verbose"
    ]

    success = run_cli_command(cmd)
    if success:
        image_files = list(output_dir.glob("*.jpg")) + list(output_dir.glob("*.png"))
        print(f"✓ 步骤3完成: 生成了 {len(image_files)} 张COCO可视化图像")
        return True, output_dir
    else:
        print("✗ 步骤3失败")
        return False, None

def step4_coco_to_yolo(coco_file):
    """步骤4: COCO转YOLO格式"""
    print("\n" + "="*60)
    print("步骤4: COCO转YOLO格式")
    print("="*60)

    image_dir = project_root / "assets" / "test_data" / "seg" / "coco" / "images"
    output_dir = project_root / "temp_output" / "full_demo" / "step4_yolo_converted"

    cleanup_directory(output_dir)

    cmd = [
        "convert", "coco2yolo",
        str(coco_file),
        str(output_dir),
        "--image-dir", str(image_dir),
        "--verbose"
    ]

    success = run_cli_command(cmd)
    if success:
        # YOLO标签文件可能在labels子目录中
        labels_dir = output_dir / "labels"
        if labels_dir.exists():
            txt_files = list(labels_dir.rglob("*.txt"))
        else:
            txt_files = list(output_dir.rglob("*.txt"))
        # 过滤掉classes.txt文件（不是标注文件）
        label_files = [f for f in txt_files if f.name != "classes.txt"]
        print(f"✓ 步骤4完成: 生成了 {len(label_files)} 个YOLO标注文件")
        return True, output_dir
    else:
        print("✗ 步骤4失败")
        return False, None

def step5_compare_yolo_dirs(original_dir, converted_dir):
    """步骤5: 比较原始YOLO和转换后的YOLO目录"""
    print("\n" + "="*60)
    print("步骤5: 验证转换准确性")
    print("="*60)

    # YOLO标签文件可能在labels子目录中
    original_labels_dir = original_dir
    converted_labels_dir = converted_dir / "labels" if (converted_dir / "labels").exists() else converted_dir

    # 查找所有.txt文件（递归）
    original_files = list(original_labels_dir.rglob("*.txt"))
    converted_files = list(converted_labels_dir.rglob("*.txt"))

    # 过滤掉classes.txt文件（不是标注文件）
    original_label_files = [f for f in original_files if f.name != "classes.txt"]
    converted_label_files = [f for f in converted_files if f.name != "classes.txt"]

    print(f"原始YOLO目录: {len(original_label_files)} 个标注文件")
    print(f"转换后YOLO目录: {len(converted_label_files)} 个标注文件")

    if len(original_label_files) == len(converted_label_files):
        print("✓ 文件数量匹配，转换基本准确")
        # 可以添加更详细的比较，如内容对比
        return True
    else:
        print("✗ 文件数量不匹配，转换可能有问题")
        return False

def main():
    """主函数"""
    print("DataFlow-CV CLI完整工作流演示")
    print("="*60)
    print("本演示将展示完整的标签处理工作流：")
    print("1. 可视化原始YOLO标签")
    print("2. 将YOLO标签转换为COCO格式")
    print("3. 可视化转换后的COCO标签")
    print("4. 将COCO标签转换回YOLO格式")
    print("5. 验证转换的准确性")
    print("="*60)

    # 检查测试数据是否存在
    test_data_root = project_root / "assets" / "test_data" / "seg"
    if not test_data_root.exists():
        print(f"错误: 测试数据目录不存在: {test_data_root}")
        print("请确保在项目根目录运行此脚本")
        return 1

    # 创建临时输出目录
    temp_root = project_root / "temp_output" / "full_demo"
    cleanup_directory(temp_root)

    all_success = True
    results = {}

    # 步骤1: 可视化原始YOLO标签
    success1, yolo_viz_dir = step1_visualize_yolo()
    all_success = all_success and success1
    results["step1"] = success1

    # 步骤2: YOLO转COCO格式
    success2, coco_file = step2_yolo_to_coco()
    all_success = all_success and success2
    results["step2"] = success2

    # 步骤3: 可视化COCO标签
    if success2:
        success3, coco_viz_dir = step3_visualize_coco(coco_file)
        all_success = all_success and success3
        results["step3"] = success3
    else:
        results["step3"] = False

    # 步骤4: COCO转YOLO格式
    if success2:
        success4, yolo_converted_dir = step4_coco_to_yolo(coco_file)
        all_success = all_success and success4
        results["step4"] = success4
    else:
        results["step4"] = False

    # 步骤5: 比较原始YOLO和转换后的YOLO
    if success1 and success4:
        original_yolo_dir = project_root / "assets" / "test_data" / "seg" / "yolo" / "labels"
        success5 = step5_compare_yolo_dirs(original_yolo_dir, yolo_converted_dir)
        all_success = all_success and success5
        results["step5"] = success5
    else:
        results["step5"] = False

    # 总结
    print("\n" + "="*60)
    print("工作流演示总结")
    print("="*60)
    print("步骤完成情况:")
    for i, (step, success) in enumerate(results.items(), 1):
        status = "✓" if success else "✗"
        step_name = {
            "step1": "可视化原始YOLO标签",
            "step2": "YOLO转COCO格式",
            "step3": "可视化COCO标签",
            "step4": "COCO转YOLO格式",
            "step5": "验证转换准确性",
        }.get(step, step)
        print(f"  步骤{i}: {step_name} {status}")

    if all_success:
        print("\n✓ 完整工作流演示成功完成!")
        print(f"所有结果保存在: {project_root / 'temp_output' / 'full_demo'}")
        print("\n您已成功体验了DataFlow-CV CLI的核心功能:")
        print("  - 可视化YOLO和COCO格式标签")
        print("  - 在YOLO和COCO格式间进行双向转换")
        print("  - 验证转换的准确性")
        return 0
    else:
        print("\n✗ 工作流演示部分失败")
        print("请检查错误信息，或尝试单独运行各个步骤")
        return 1

if __name__ == "__main__":
    sys.exit(main())