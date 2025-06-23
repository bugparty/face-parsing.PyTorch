#!/usr/bin/env python3
import os
import glob
import shutil

def main():
    mask_dir = '/home/bowman/data/CelebAMask-HQ/mask'
    test_dir = '/home/bowman/data/CelebAMask-HQ/test-img'
    os.makedirs(test_dir, exist_ok=True)

    # 获取所有 mask 文件，并按文件名中的数字索引排序
    mask_files = glob.glob(os.path.join(mask_dir, '*.png'))
    mask_files = sorted(
        mask_files,
        key=lambda p: int(os.path.splitext(os.path.basename(p))[0])
    )
    # 取最后 200 张（如果不足则移动全部）
    n_move = min(200, len(mask_files))
    to_move = mask_files[-n_move:]

    # 移动选中的文件到 test-img 目录
    for src in to_move:
        dst = os.path.join(test_dir, os.path.basename(src))
        shutil.move(src, dst)
        print(f"Moved {os.path.basename(src)}")

    print(f"Done. Moved {len(to_move)} files to {test_dir}")

if __name__ == '__main__':
    main()