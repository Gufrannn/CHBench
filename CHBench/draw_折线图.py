import json
import os
import matplotlib.pyplot as plt
from collections import defaultdict

# 文件夹路径
folder_path = r"C:\Users\Administrator\Desktop\result image choos\wcot_aigc_type_1\minigpt"

# 定义模型名称和对应的文件名标识
models = ["minigpt4", "cpm", "qwen", "owl", "llava", "ins"]

# 定义模型正规化名称的映射
model_names_mapping = {
    "minigpt4": "MiniGPT-4",
    "cpm": "MiniCPM",
    "qwen": "Qwen-vl",
    "owl": "mPLUG-Owl",
    "llava": "LLaVA-1.5",
    "ins": "InstructBLIP"
}

# 定义模型的自定义颜色
model_styles = {
    "minigpt4": {"color": (64/255, 4/255, 90/255), "marker": "o"},
    "cpm": {"color": (65/255, 62/255, 133/255), "marker": "s"},
    "qwen": {"color": (48/255, 104/255, 141/255), "marker": "D"},
    "owl": {"color": (248/255, 230/255, 32/255), "marker": "^"},
    "llava": {"color": (53/255, 189/255, 119/255), "marker": "v"},
    "ins": {"color": (145/255, 213/255, 66/255), "marker": "p"}
}
# 初始化存储模型评分数据的字典
model_data = {model: defaultdict(list) for model in models}

# 将单词数量分组，每组20个单词
def get_word_count_group(word_count):
    return (word_count // 20) * 20

# 遍历文件夹中的所有文件
for file_name in os.listdir(folder_path):
    if file_name.endswith('.json'):
        # 查找当前文件属于哪个模型的评估
        for model in models:
            if f"_minigpt4_{model}_" in file_name:
                file_path = os.path.join(folder_path, file_name)
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                    # 处理每个数据条目
                    for item in data['data']:
                        word_count = len(item['model_output'].split())
                        group = get_word_count_group(word_count)
                        model_rating = int(item['model_rating'])
                        model_data[model][group].append(model_rating)

# 计算每个分组的平均评分
average_ratings_per_model = {}
for model, ratings in model_data.items():
    sorted_groups = sorted(ratings.keys())
    average_ratings = [sum(ratings[group]) / len(ratings[group]) for group in sorted_groups]
    average_ratings_per_model[model] = (sorted_groups, average_ratings)

# 绘制折线图
plt.figure(figsize=(12, 8))
for model, (groups, average_ratings) in average_ratings_per_model.items():
    range_labels = [group for group in groups]
    # 使用正规化的模型名称和自定义颜色
    plt.plot(

        range_labels,
        average_ratings,
        label=model_names_mapping[model],
        color=model_styles[model]["color"],
        marker=model_styles[model]["marker"],
        linewidth=2.5,
        markersize = 10,  # 标记大小
        markeredgewidth = 2
    )
plt.xlabel('Text Length',fontsize=22)
plt.ylabel('Average Model Rating', fontsize=22)
plt.legend(fontsize=19)
plt.grid(True)
plt.xticks(range(0, max(groups) + 20, 20),rotation=0,fontsize=19)
plt.yticks(fontsize=19)
#plt.xticks(ticks=range(0, len(range_labels), 2), labels=range_labels[::2], rotation=0)
plt.tight_layout()
output_file_path = os.path.join(folder_path, 'minigpt.png')
plt.savefig(output_file_path)