import os


def rename_images_in_folder(folder_path):
    # 获取文件夹中的所有文件
    files = os.listdir(folder_path)

    for file_name in files:
        # 检查文件是否为图片文件，可以根据需要扩展更多的图片格式
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
            # 提取文件名和扩展名
            base_name, extension = os.path.splitext(file_name)

            # 检查文件名是否以'image'开头并且后面跟着数字
            if base_name.startswith('image') and base_name[5:].isdigit():
                # 提取数字部分并加上127
                new_number = int(base_name[5:]) + 170

                # 创建新的文件名
                new_file_name = f'image{new_number}{extension}'

                # 获取旧文件的完整路径和新文件的完整路径
                old_file_path = os.path.join(folder_path, file_name)
                new_file_path = os.path.join(folder_path, new_file_name)

                # 重命名文件
                os.rename(old_file_path, new_file_path)
                print(f'Renamed {file_name} to {new_file_name}')


# 指定文件夹路径
folder_path = r"C:\Users\Administrator\Desktop\new_winnoground_type3\winnoground_type3_2_p+n"
rename_images_in_folder(folder_path)
