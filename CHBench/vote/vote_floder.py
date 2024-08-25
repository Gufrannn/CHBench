import os
import json
from collections import Counter
import re

# 指定文件夹路径
folder_path = "/home/liudongdong/result image choos/output_winnoground_type1/"


# 提取原始图片名称的方法
def get_base_image_name(image_name):
    return re.sub(r'round\d+_', '', image_name)


# 遍历文件夹中的所有文件
for filename in os.listdir(folder_path):
    if filename.startswith('model_') and filename.endswith('.json'):
        file_path = os.path.join(folder_path, filename)

        # 读取JSON文件
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
            final_choices[base_image_name] = counts.most_common(1)[0][0]  # 选择出现次数最多的选项

        # 只保留每张图片的一个条目，并更新generated_output字段
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

        # 保存结果到新的JSON文件
        output_filename = f'vote_{filename}'
        output_path = os.path.join(folder_path, output_filename)
        with open(output_path, 'w', encoding='utf-8') as output_file:
            json.dump(final_output_data, output_file, ensure_ascii=False, indent=4)

        print(f"Results saved to {output_path}")
