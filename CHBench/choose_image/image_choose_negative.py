# -*- coding: utf-8 -*-
import os
import json
from collections import Counter

# 指定文件夹路径
folder_path = "/home/liudongdong/filter_img_results/output_w3n2/"

# 获取文件夹中的所有JSON文件
json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]

# 创建一个字典来存储每个文件中回答错误的图像名
incorrect_images_dict = {}

# 遍历每个JSON文件
for json_file in json_files:
    file_path = os.path.join(folder_path, json_file)
    
    # 读取JSON文件
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    # 创建一个列表来存储回答错误的图像名
    incorrect_images = []

    # 遍历数据并检查生成的输出是否正确
    for item in data['data']:
        if item['generated_output'].strip() != item['correct_answers']:
            incorrect_images.append(item['image_name'])
    
    # 将结果存储到字典中
    incorrect_images_dict[json_file] = incorrect_images

# 获取所有错误图像名的集合
all_incorrect_images = set()
for images in incorrect_images_dict.values():
    all_incorrect_images.update(images)

# 查找在所有文件中都出现的错误图像名
common_incorrect_images = []
for image in all_incorrect_images:
    # 统计每个错误图像名在所有文件中的出现次数
    occurrence_count = sum(image in images for images in incorrect_images_dict.values())
    # 如果出现次数等于文件总数，则说明该图像在所有文件中都存在
    if occurrence_count == len(incorrect_images_dict):
        common_incorrect_images.append(image)

# 输出每个数组中都有的错误图像名
print("Common Incorrect Images in All Files:")
print(common_incorrect_images)
