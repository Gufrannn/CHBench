import json
import os
from collections import Counter
import re


# 提取原始图片名称的方法
def get_base_image_name(image_name):
    return re.sub(r'round\d+_', '', image_name)


# 处理单个文件
def process_json_file(file_path, output_folder):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)["data"]

    # 统计每张图片模型输出的选项次数
    results = {}
    for entry in data:
        base_image_name = get_base_image_name(entry["image_name"])
        generated_output = entry["generated_output"].strip()  # 提取模型的生成输出并去除前后的空白字符
        model_answer = None
        for i, opt in enumerate(entry["options"]):
            if str(i + 1) in generated_output:
                model_answer = opt
                break
            elif opt in generated_output:
                model_answer = opt
                break
        if base_image_name not in results:
            results[base_image_name] = Counter()

        results[base_image_name][model_answer] += 1

    # 确定每张图片最终的选择
    final_choices = {}
    for base_image_name, counts in results.items():
        most_common = counts.most_common()
        max_count = most_common[0][1]
        final_choices[base_image_name] = [choice for choice, count in most_common if count == max_count]

    # 只保留每张图片的一个条目，并更新generated_output字段，删除model_output字段
    unique_data = {}
    for entry in data:
        base_image_name = get_base_image_name(entry["image_name"])
        if base_image_name not in unique_data:
            entry["generated_output"] = final_choices[base_image_name]
            entry["image_name"] = base_image_name  # 更新图片名称为基础名称
            del entry["model_output"]  # 删除model_output字段
            unique_data[base_image_name] = entry

    # 创建最终的输出数据
    final_output_data = {"data": list(unique_data.values())}

    # 生成输出文件路径
    output_file_name = "final_" + os.path.basename(file_path)
    output_path = os.path.join(output_folder, output_file_name)

    # 将结果写入新的JSON文件
    with open(output_path, 'w', encoding='utf-8') as output_file:
        json.dump(final_output_data, output_file, ensure_ascii=False, indent=4)

    print(f"Processed and saved: {output_path}")


# 遍历文件夹处理所有以 model_ 开头的 JSON 文件
def process_all_files_in_folder(input_folder, output_folder):
    for file_name in os.listdir(input_folder):
        if file_name.startswith("model_") and file_name.endswith(".json"):
            file_path = os.path.join(input_folder, file_name)
            process_json_file(file_path, output_folder)


# 定义输入文件夹和输出文件夹路径
input_folder = "/home/liudongdong/filter_img_results/woct_nature_type3/"
output_folder = "/home/liudongdong/filter_img_results/woct_nature_type3/"

# 创建输出文件夹（如果不存在）
os.makedirs(output_folder, exist_ok=True)

# 处理文件夹中的所有文件
process_all_files_in_folder(input_folder, output_folder)
