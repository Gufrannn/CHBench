import os
from PIL import Image
import re

# 自然排序函数
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

# 文件夹路径
folder_path = r"C:\Users\Administrator\Desktop\winnoground_type3_2 - 副本"

# 获取文件夹内的文件名
files = [f for f in os.listdir(folder_path) if f.endswith('.png')]

# 自然排序
files.sort(key=natural_sort_key)

# 创建图片对的字典
image_pairs = {}
for file in files:
    base_name = file.rsplit('_', 1)[0]  # 去掉_0或_1部分
    if base_name in image_pairs:
        image_pairs[base_name].append(file)
    else:
        image_pairs[base_name] = [file]

# 合并图片并保存
image_count = 0
for base_name, pair_files in image_pairs.items():
    if len(pair_files) != 2:
        print(f"警告: 找到不完整的图片对: {pair_files}")
        continue

    img0_path = os.path.join(folder_path, pair_files[0])
    img1_path = os.path.join(folder_path, pair_files[1])

    img0 = Image.open(img0_path)
    img1 = Image.open(img1_path)

    # 合并图片，img0在左，img1在右，中间有10像素的白色区域
    combined_width = img0.width + img1.width + 10
    combined_height = max(img0.height, img1.height)

    combined_img = Image.new('RGB', (combined_width, combined_height), (255, 255, 255))
    combined_img.paste(img0, (0, 0))
    combined_img.paste(img1, (img0.width + 10, 0))
    # 保存合并后的图片
    image_count += 1
    name = image_count
    new_image_name = f'image{name}.jpg'
    combined_img.save(os.path.join(folder_path, new_image_name))

    # 删除源文件
    os.remove(img0_path)
    os.remove(img1_path)

print(f"共处理了 {image_count} 对图片，并保存为 image1 到 image{image_count}")
