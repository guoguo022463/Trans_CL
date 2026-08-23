import os
import shutil

def ignore_all_files(dir_path, contents):
    # 这个函数的核心逻辑就是：只忽略“文件”，从而把“文件夹结构”完整保留下来
    return [c for c in contents if os.path.isfile(os.path.join(dir_path, c))]

src_dir = r"D:\contrastive_transformers_ids-main"
dst_dir = r"D:\contrastive_transformers_ids-main_structure"

print("开始提取纯文件夹结构...")
# dirs_exist_ok=True 可以完美解决你刚才遇到的“文件已存在”报错
shutil.copytree(src_dir, dst_dir, ignore=ignore_all_files, dirs_exist_ok=True)
print(f"成功！现在 {dst_dir} 里面只有文件夹结构，没有任何文件。")