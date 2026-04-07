"""
批量文件重命名脚本
规则: 前缀 + 序号 + 后缀 + 扩展名
序号位数根据文件总数自动确定（如 01, 001, 0001）
用法: python cmd/batch_rename.py
"""

import os
import sys
from pathlib import Path


def get_files(directory, extensions):
    """获取目录下指定扩展名的文件，按名称排序"""
    ext_set = {e.lower().lstrip(".") for e in extensions}
    files = [
        f for f in sorted(Path(directory).iterdir())
        if f.is_file() and f.suffix.lower().lstrip(".") in ext_set
    ]
    return files


def preview_rename(files, prefix, suffix, start_num, pad_width):
    """预览重命名结果"""
    results = []
    for i, f in enumerate(files):
        num = start_num + i
        new_name = f"{prefix}{str(num).zfill(pad_width)}{suffix}{f.suffix}"
        results.append((f.name, new_name))
    return results


def execute_rename(directory, files, prefix, suffix, start_num, pad_width):
    """执行重命名，使用临时名避免冲突"""
    temp_names = []
    for i, f in enumerate(files):
        temp = f.parent / f"__temp_rename_{i}___{f.suffix}"
        f.rename(temp)
        temp_names.append(temp)

    renamed = 0
    for i, temp in enumerate(temp_names):
        num = start_num + i
        new_name = f"{prefix}{str(num).zfill(pad_width)}{suffix}{temp.suffix}"
        new_path = Path(directory) / new_name
        temp.rename(new_path)
        renamed += 1

    return renamed


def main():
    print("=" * 50)
    print("  batch rename")
    print("=" * 50)

    directory = input("\ndirectory path: ").strip().strip('"')
    if not os.path.isdir(directory):
        print(f"  not a valid directory: {directory}")
        return

    ext_input = input("file extensions (e.g. jpg,png,txt): ").strip()
    if not ext_input:
        print("  no extensions specified")
        return
    extensions = [e.strip() for e in ext_input.split(",")]

    files = get_files(directory, extensions)
    if not files:
        print(f"  no files found with extensions: {extensions}")
        return

    print(f"\n  found {len(files)} files")

    prefix = input("prefix (e.g. img-): ").strip()
    suffix = input("suffix (e.g. -raw, press Enter to skip): ").strip()
    start_input = input("start number (default 1): ").strip()
    start_num = int(start_input) if start_input else 1

    total = start_num + len(files) - 1
    pad_width = len(str(total))

    print(f"\n  digit width: {pad_width} (max number: {total})")
    print("-" * 50)
    print("  preview:")

    results = preview_rename(files, prefix, suffix, start_num, pad_width)
    show_count = min(5, len(results))
    for old, new in results[:show_count]:
        print(f"    {old}  ->  {new}")
    if len(results) > show_count:
        print(f"    ... ({len(results) - show_count} more)")
    if len(results) > 1:
        old, new = results[-1]
        print(f"    {old}  ->  {new}")

    print("-" * 50)
    confirm = input("  proceed? (y/n): ").strip().lower()
    if confirm != "y":
        print("  cancelled")
        return

    renamed = execute_rename(directory, files, prefix, suffix, start_num, pad_width)
    print(f"\n  done: {renamed} files renamed")


if __name__ == "__main__":
    main()
