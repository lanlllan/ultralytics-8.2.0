"""
批量图片数据增强脚本
功能：对原始图片进行亮度、饱和度、色温调整，生成训练集变体
      同时自动复制同名 .txt / .json 标注文件，保持标注与图片一一对应
      JSON 标注文件会自动更新 imagePath 字段指向新图片
用法：python cmd/batch_image_augmentation.py
"""

import os
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import json
import shutil
from PIL import Image, ImageEnhance
import numpy as np
from pathlib import Path


def adjust_brightness(image, factor):
    """
    调整图片亮度
    :param image: PIL Image对象
    :param factor: 亮度因子，-100到100的值会被转换为合适的增强因子
    :return: 调整后的图片
    """
    # 将-100到100的范围转换为0到2的增强因子
    # -100 -> 0 (完全变暗)
    # 0 -> 1 (原始亮度)
    # 100 -> 2 (亮度翻倍)
    enhancer_factor = 1 + (factor / 100)
    enhancer_factor = max(0, min(2, enhancer_factor))
    
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(enhancer_factor)


def adjust_saturation(image, factor):
    """
    调整图片饱和度
    :param image: PIL Image对象
    :param factor: 饱和度因子，-100到100
    :return: 调整后的图片
    """
    # 将-100到100的范围转换为0到2的增强因子
    enhancer_factor = 1 + (factor / 100)
    enhancer_factor = max(0, min(2, enhancer_factor))
    
    enhancer = ImageEnhance.Color(image)
    return enhancer.enhance(enhancer_factor)


def adjust_color_temperature(image, factor):
    """
    调整图片色温
    :param image: PIL Image对象
    :param factor: 色温因子，-100到100
                  正值：增加暖色调（增加红色，减少蓝色）
                  负值：增加冷色调（减少红色，增加蓝色）
    :return: 调整后的图片
    """
    # 转换为RGB numpy数组
    img_array = np.array(image, dtype=np.float32)
    
    # 根据factor调整色温
    # factor范围：-100到100
    # 正值：暖色调，增强红色通道，减弱蓝色通道
    # 负值：冷色调，减弱红色通道，增强蓝色通道
    
    red_adjust = 1 + (factor / 200)  # -100: 0.5, 0: 1.0, 100: 1.5
    blue_adjust = 1 - (factor / 200)  # -100: 1.5, 0: 1.0, 100: 0.5
    
    img_array[:, :, 0] = np.clip(img_array[:, :, 0] * red_adjust, 0, 255)  # Red
    img_array[:, :, 2] = np.clip(img_array[:, :, 2] * blue_adjust, 0, 255)  # Blue
    
    return Image.fromarray(img_array.astype(np.uint8))


def copy_label_file(src_label_path, dst_label_path):
    """
    复制标注文件（.txt 直接复制）
    :param src_label_path: 源标注文件路径
    :param dst_label_path: 目标标注文件路径
    :return: 是否复制成功
    """
    if src_label_path.exists():
        shutil.copy2(src_label_path, dst_label_path)
        return True
    return False


def copy_json_label(src_json_path, dst_json_path, new_image_name, image_output_dir=None):
    """
    复制 JSON 标注文件，并更新 imagePath 指向新图片
    :param src_json_path: 源 JSON 文件路径
    :param dst_json_path: 目标 JSON 文件路径
    :param new_image_name: 新图片的文件名（如 001-1.jpg）
    :param image_output_dir: 图片输出目录，若指定则 imagePath 指向该目录下的新图片；否则沿用原路径的父目录
    :return: 是否复制成功
    """
    if not src_json_path.exists():
        return False
    
    with open(src_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if "imagePath" in data:
        if image_output_dir is not None:
            data["imagePath"] = str(Path(image_output_dir) / new_image_name)
        else:
            old_path = Path(data["imagePath"])
            data["imagePath"] = str(old_path.parent / new_image_name)
    
    with open(dst_json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    return True


def process_single_image(input_path, output_dir=None, txt_dir=None, json_dir=None, 
                         output_txt_dir=None, output_json_dir=None):
    """
    处理单张图片，生成6个变体，同时复制对应的标注文件
    :param input_path: 输入图片路径（如：001-0-0.jpg）
    :param output_dir: 图片输出目录，如果为None则与输入图片同目录
    :param txt_dir: TXT标注文件源目录，如果为None则在图片同目录查找
    :param json_dir: JSON标注文件源目录，如果为None则在图片同目录查找
    :param output_txt_dir: TXT标注文件输出目录，如果为None则与output_dir相同
    :param output_json_dir: JSON标注文件输出目录，如果为None则与output_dir相同
    :return: (处理的图片列表, 复制的标注数量)
    """
    input_path = Path(input_path)
    
    if not input_path.exists():
        print(f"  file not found: {input_path}")
        return [], 0
    
    if output_dir is None:
        output_dir = input_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    if txt_dir is None:
        txt_dir = input_path.parent
    else:
        txt_dir = Path(txt_dir)
    
    if json_dir is None:
        json_dir = input_path.parent
    else:
        json_dir = Path(json_dir)
    
    if output_txt_dir is None:
        output_txt_dir = output_dir
    else:
        output_txt_dir = Path(output_txt_dir)
        output_txt_dir.mkdir(parents=True, exist_ok=True)
    
    if output_json_dir is None:
        output_json_dir = output_dir
    else:
        output_json_dir = Path(output_json_dir)
        output_json_dir.mkdir(parents=True, exist_ok=True)
    
    stem = input_path.stem
    ext = input_path.suffix
    
    if stem.endswith('-0'):
        base_name = stem[:-2]
    else:
        base_name = stem
    
    src_txt = txt_dir / f"{stem}.txt"
    src_json = json_dir / f"{stem}.json"
    has_txt = src_txt.exists()
    has_json = src_json.exists()
    
    label_info = []
    if has_txt:
        label_info.append(src_txt.name)
    if has_json:
        label_info.append(src_json.name)
    label_str = f"  (labels: {', '.join(label_info)})" if label_info else "  (no label)"
    print(f"\n  process: {input_path.name}{label_str}")
    
    try:
        img = Image.open(input_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        augmentations = [
            (1, "brightness -50", adjust_brightness, -50),
            (2, "brightness +50", adjust_brightness, 50),
            (3, "saturation +100", adjust_saturation, 100),
            (4, "saturation -100", adjust_saturation, -100),
            (5, "temperature +100", adjust_color_temperature, 100),
            (6, "temperature -100", adjust_color_temperature, -100),
        ]
        
        processed_files = []
        label_count = 0
        
        for idx, desc, func, factor in augmentations:
            img_out = output_dir / f"{base_name}-{idx}{ext}"
            img_adjusted = func(img, factor)
            img_adjusted.save(img_out, quality=95)
            processed_files.append(img_out)
            
            copied = []
            if has_txt:
                copy_label_file(src_txt, output_txt_dir / f"{base_name}-{idx}.txt")
                copied.append(".txt")
                label_count += 1
            if has_json:
                new_img_name = f"{base_name}-{idx}{ext}"
                copy_json_label(src_json, output_json_dir / f"{base_name}-{idx}.json", new_img_name, output_dir)
                copied.append(".json")
                label_count += 1
            
            suffix = f" + {', '.join(copied)}" if copied else ""
            print(f"   -> {img_out.name} ({desc}){suffix}")
        
        print(f"  done: {len(processed_files)} images, {label_count} label files copied")
        return processed_files, label_count
        
    except Exception as e:
        print(f"  failed: {e}")
        return [], 0


def batch_process_directory(input_dir, pattern="*-0.jpg", output_dir=None, txt_dir=None, json_dir=None,
                            output_txt_dir=None, output_json_dir=None):
    """
    批量处理目录中的图片，同时复制对应的标注文件
    :param input_dir: 输入目录路径
    :param pattern: 文件匹配模式，默认为 *-0.jpg
    :param output_dir: 图片输出目录，如果为None则与输入图片同目录
    :param txt_dir: TXT标注文件源目录，如果为None则在图片同目录查找
    :param json_dir: JSON标注文件源目录，如果为None则在图片同目录查找
    :param output_txt_dir: TXT标注文件输出目录，如果为None则与output_dir相同
    :param output_json_dir: JSON标注文件输出目录，如果为None则与output_dir相同
    :return: 处理的文件总数
    """
    input_dir = Path(input_dir)
    
    if not input_dir.exists():
        print(f"  dir not found: {input_dir}")
        return 0
    
    image_files = sorted(input_dir.glob(pattern))
    
    if not image_files:
        print(f"  no files matching '{pattern}' in {input_dir}")
        return 0
    
    txt_search_dir = Path(txt_dir) if txt_dir else input_dir
    json_search_dir = Path(json_dir) if json_dir else input_dir
    txt_count = len(list(txt_search_dir.glob("*.txt"))) if txt_search_dir.exists() else 0
    json_count = len(list(json_search_dir.glob("*.json"))) if json_search_dir.exists() else 0
    
    print(f"\n  found {len(image_files)} images, {txt_count} txt files, {json_count} json files")
    print(f"  input images:   {input_dir}")
    if txt_dir:
        print(f"  txt (source):  {txt_dir}")
    if json_dir:
        print(f"  json (source): {json_dir}")
    if output_dir:
        print(f"  output images: {output_dir}")
    if output_txt_dir:
        print(f"  output txt:    {output_txt_dir}")
    if output_json_dir:
        print(f"  output json:   {output_json_dir}")
    print("-" * 60)
    
    total_images = 0
    total_labels = 0
    for img_path in image_files:
        images, labels = process_single_image(
            img_path, output_dir, txt_dir, json_dir,
            output_txt_dir=output_txt_dir, output_json_dir=output_json_dir
        )
        total_images += len(images)
        total_labels += labels
    
    print("\n" + "=" * 60)
    print(f"  batch done!")
    print(f"  source images: {len(image_files)}")
    print(f"  generated images: {total_images}")
    print(f"  copied labels:   {total_labels}")
    print("=" * 60)
    
    return total_images


if __name__ == "__main__":
    import sys
    
    # 默认处理目录（输入）
    default_input_dir = "./datasets/bvn/images/val"
    default_txt_dir = "./datasets/bvn/labels/val"
    default_json_dir = "./datasets/x-AngLabel-output"
    # 默认输出目录（留空则与输入/图片输出同目录）
    default_output_dir = None
    default_output_txt_dir = None
    default_output_json_dir = None
    
    print("=" * 60)
    print("  图片数据增强工具")
    print("  支持亮度、饱和度、色温调整")
    print("=" * 60)
    
    if len(sys.argv) > 1:
        if sys.argv[1].endswith(('.jpg', '.jpeg', '.png')):
            input_file = sys.argv[1]
            output_dir = sys.argv[2] if len(sys.argv) > 2 else None
            txt_dir = sys.argv[3] if len(sys.argv) > 3 else None
            json_dir = sys.argv[4] if len(sys.argv) > 4 else None
            output_txt_dir = sys.argv[5] if len(sys.argv) > 5 else None
            output_json_dir = sys.argv[6] if len(sys.argv) > 6 else None
            process_single_image(input_file, output_dir, txt_dir, json_dir, output_txt_dir, output_json_dir)
        else:
            input_dir = sys.argv[1]
            pattern = sys.argv[2] if len(sys.argv) > 2 else "*-0.jpg"
            output_dir = sys.argv[3] if len(sys.argv) > 3 else None
            txt_dir = sys.argv[4] if len(sys.argv) > 4 else None
            json_dir = sys.argv[5] if len(sys.argv) > 5 else None
            output_txt_dir = sys.argv[6] if len(sys.argv) > 6 else None
            output_json_dir = sys.argv[7] if len(sys.argv) > 7 else None
            batch_process_directory(input_dir, pattern, output_dir, txt_dir, json_dir, output_txt_dir, output_json_dir)
    else:
        print("\n1. batch (default directories)")
        print("2. single file")
        print("3. custom directory")
        
        choice = input("\nchoice (1/2/3, enter=1): ").strip() or "1"
        
        if choice == "1":
            print(f"\n  using defaults:")
            print(f"  - input images: {default_input_dir}")
            print(f"  - input txt:    {default_txt_dir}")
            print(f"  - input json:   {default_json_dir}")
            print(f"  - output (all): 同输入目录，或下面分别指定")
            output_dir = input("  output images dir (enter=same as input): ").strip() or default_output_dir
            output_txt_dir = input("  output txt dir (enter=same as images output): ").strip() or None
            output_json_dir = input("  output json dir (enter=same as images output): ").strip() or None
            confirm = input("\n  continue? (y/n, enter=y): ").strip().lower()
            if confirm in ("", "y", "yes"):
                batch_process_directory(
                    default_input_dir,
                    "*-0.jpg",
                    output_dir=output_dir or None,
                    txt_dir=default_txt_dir,
                    json_dir=default_json_dir,
                    output_txt_dir=output_txt_dir or None,
                    output_json_dir=output_json_dir or None,
                )
            else:
                print("  cancelled")
        elif choice == "2":
            file_path = input("image path: ").strip()
            if file_path:
                output_dir = input("output images dir (enter=same as image): ").strip() or None
                txt_dir = input("txt source dir (enter=same as image): ").strip() or None
                json_dir = input("json source dir (enter=same as image): ").strip() or None
                output_txt_dir = input("output txt dir (enter=same as images output): ").strip() or None
                output_json_dir = input("output json dir (enter=same as images output): ").strip() or None
                process_single_image(file_path, output_dir, txt_dir, json_dir, output_txt_dir, output_json_dir)
            else:
                print("  no path given")
        elif choice == "3":
            dir_path = input("image directory: ").strip()
            if dir_path:
                pattern = input("pattern (default: *-0.jpg): ").strip() or "*-0.jpg"
                output_dir = input("output images dir (enter=same as input): ").strip() or None
                txt_dir = input("txt source dir (enter=same as input): ").strip() or None
                json_dir = input("json source dir (enter=same as input): ").strip() or None
                output_txt_dir = input("output txt dir (enter=same as images output): ").strip() or None
                output_json_dir = input("output json dir (enter=same as images output): ").strip() or None
                batch_process_directory(
                    dir_path, pattern, output_dir, txt_dir, json_dir,
                    output_txt_dir or None, output_json_dir or None
                )
            else:
                print("  no path given")
        else:
            print("  invalid choice")
