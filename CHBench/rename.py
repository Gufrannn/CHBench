import os
import re
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]
def rename_images(folder_path):
    # 获取文件夹中的所有文件
    files = os.listdir(folder_path)

    # 过滤出图像文件（假设图像文件扩展名为jpg、jpeg、png、gif等）
    image_files = [f for f in files if f.lower().endswith(('jpg', 'jpeg', 'png', 'gif'))]

    # 按文件名排序
    image_files.sort(key=natural_sort_key)

    # 重命名图像文件
    for index, filename in enumerate(image_files):
        group_number = (index // 2) + 1
        sub_number = index % 2
        new_filename = f"{group_number}_{sub_number}{os.path.splitext(filename)[1]}"  # 生成新的文件名
        old_file_path = os.path.join(folder_path, filename)
        new_file_path = os.path.join(folder_path, new_filename)
        os.rename(old_file_path, new_file_path)  # 重命名文件

    print(f"重命名了 {len(image_files)} 个图像文件。")


# 使用示例
folder_path = r"C:\Users\Administrator\Desktop\winnoground_type3_2 - 副本"
rename_images(folder_path)
