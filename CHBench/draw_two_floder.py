import os
import json
import matplotlib.pyplot as plt
import numpy as np
import re

# 指定两个文件夹路径
folder_path_1 = r"C:\Users\Administrator\Desktop\result image choos\aigc_type1"
folder_path_2 = r"C:\Users\Administrator\Desktop\result image choos\wcot_tpye1_aigc"

# 获取文件夹下的所有json文件，假设两个文件夹的文件名是相同的
json_files = [f for f in os.listdir(folder_path_1) if f.endswith('.json')]

# 初始化列表存储每个文件的平均单词数
average_word_counts_1 = []
average_word_counts_2 = []


# 定义一个函数来计算平均单词数
def calculate_average_word_count(folder_path, json_file):
    file_path = os.path.join(folder_path, json_file)

    # 读取JSON文件
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # 初始化统计变量
    total_words = 0
    output_count = 0

    # 遍历data中的每个元素
    for item in data["data"]:
        # 获取model_output并移除特殊标记<s>和</s>
        model_output = item["model_output"].replace("<s>", "").replace("</s>", "").strip()

        # 移除特殊字符和多余的换行符
        model_output = re.sub(r'\u200b', '', model_output)  # 移除\u200b
        model_output = re.sub(r'\n+', ' ', model_output)  # 将多个换行符替换为空格

        # 计算单词数
        word_count = len(model_output.split())

        # 累加总单词数和计数器
        total_words += word_count
        output_count += 1

    # 计算平均单词数
    average_words = total_words / output_count if output_count > 0 else 0
    return average_words


# 计算两个文件夹中每个文件的平均单词数
for json_file in json_files:
    avg_words_1 = calculate_average_word_count(folder_path_1, json_file)
    avg_words_2 = calculate_average_word_count(folder_path_2, json_file)
    average_word_counts_1.append(avg_words_1)
    average_word_counts_2.append(avg_words_2)

# 分离文件名
file_names = [f.replace('.json', '') for f in json_files]

# 设置柱状图的宽度和X轴
bar_width = 0.35
index = np.arange(len(file_names))

# 设置字体大小
plt.rcParams.update({'font.size': 21})

# 绘制双柱柱状图
plt.figure(figsize=(12, 7))
plt.bar(index, average_word_counts_1, bar_width, color=(193/255, 18/255, 33/255), label='w/o COT')
plt.bar(index + bar_width, average_word_counts_2, bar_width, color=(0/255, 47/255, 73/255), label='w COT')

# 设置X轴和Y轴标签及标题
plt.xlabel('',fontsize=24)#LVLMs
plt.ylabel('Average Word Count',fontsize=24)
plt.xticks(index + bar_width / 2, file_names, rotation=20)  # 标签水平显示，字体大小为14
plt.yticks(fontsize=16)
plt.legend(fontsize=27)

plt.tight_layout()

# 保存柱状图到指定文件夹
output_file_path = os.path.join(folder_path_1, 'ww.pdf')
plt.savefig(output_file_path)

# 如果不需要显示图表，可以注释掉以下行
# plt.show()

print(f"柱状图已保存到: {output_file_path}")
