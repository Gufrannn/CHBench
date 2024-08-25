import json
from collections import Counter
import re

# 读取JSON文件
file_path = "/home/liudongdong/filter_img_results/aigc_type3_pn/llava.json"

with open(file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)["data"]


# 提取原始图片名称的方法
def get_base_image_name(image_name):
    return re.sub(r'round\d+_', '', image_name)


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
        model_answer=opt
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

# 将结果写入新的JSON文件
output_path = "/home/liudongdong/filter_img_results/aigc_type3_pn/final_llava.json"
with open(output_path, 'w', encoding='utf-8') as output_file:
    json.dump(final_output_data, output_file, ensure_ascii=False, indent=4)

print(f"Results saved to {output_path}")
