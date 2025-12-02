import os
import re
import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--root_dir", type=str, required=True)
args = parser.parse_args()
root_dir = args.root_dir

# 获取所有 model-*.pt 文件
files = glob.glob(os.path.join(root_dir, "model-*.pt"))

# 提取编号并排序
def extract_number(f):
    match = re.search(r'model-(\d+)\.pt$', f)
    return int(match.group(1)) if match else -1

sorted_files = sorted(files, key=extract_number)

# 保留每10个（编号 % 10 == 0），同时保留第一个和最后一个
to_keep = set()
for f in sorted_files:
    num = extract_number(f)
    if num % 10 == 0 or num == extract_number(sorted_files[0]) or num == extract_number(sorted_files[-1]):
        to_keep.add(f)

# 打印保留的 checkpoints
print("=" * 60)
print(f"Total checkpoints found: {len(sorted_files)}")
print(f"Checkpoints to KEEP ({len(to_keep)}):")
print("=" * 60)
for f in sorted(to_keep, key=extract_number):
    print(f"  ✓ {os.path.basename(f)} (model-{extract_number(f)})")

# 收集要删除的文件
to_remove = [f for f in sorted_files if f not in to_keep]

print("\n" + "=" * 60)
print(f"Checkpoints to REMOVE ({len(to_remove)}):")
print("=" * 60)
for f in to_remove:
    print(f"  ✗ {os.path.basename(f)} (model-{extract_number(f)})")

# 用户确认
print("\n" + "=" * 60)
if to_remove:
    confirm = input(f"Are you sure you want to delete {len(to_remove)} checkpoints? (y/n): ")
    if confirm.lower() == 'y':
        print("\nDeleting checkpoints...")
        for f in to_remove:
            print(f"  Removing: {os.path.basename(f)}")
            os.remove(f)
        print(f"\n✓ Successfully removed {len(to_remove)} checkpoints!")
    else:
        print("\n✓ Cancelled. No files were deleted.")
else:
    print("No checkpoints to remove.")