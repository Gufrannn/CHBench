import os
import json
import torch
from PIL import Image
from lavis.models import load_model_and_preprocess

# 检查设备
device = torch.device("cuda:2") if torch.cuda.is_available() else "cpu"

# 定义图像文件夹、JSON文件和模型路径


image_folder = "/home/liudongdong/all_select_images/winnoground_type3_2/"
json_file ="/home/liudongdong/all_select_images/winnoground_type3_2/winnoground_type3_negative_2.json"
model, vis_processors, _ = load_model_and_preprocess(
    name="blip2_vicuna_instruct",
    model_type="vicuna7b",
    is_eval=True,
    device=device
)

# 从JSON文件中读取数据
with open(json_file, 'r') as f:
    data = json.load(f)

# 初始化结果列表和真实标签、预测标签列表
results = []
y_true = []
y_pred = []

# 循环处理JSON数据五次
for round_num in range(1, 6):
    # 遍历每个图像问题
    for item in data['images']:
        image_name = item['image_name']
        question = item['question']
        options = item['options']
        correct_answers = item['answer']
        image_type = item['image_type']

        # 加载和处理图像
        image_path = os.path.join(image_folder, image_name)
        raw_image = Image.open(image_path).convert('RGB')
        image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

        # 构建模型输入，包括问题和选项
        options_text = "\n".join([f"{i + 1}. {opt}" for i, opt in enumerate(options)])
        prompt = f"{question}\n{options_text}Please choose the correct number."

        # 使用模型生成答案
        description = model.generate({"image": image, "prompt": prompt})[0]

        # 提取模型选择的答案
        model_answer = None
        for i, opt in enumerate(options):
            if str(i + 1) in description:
                model_answer = opt
                break
            elif opt in description:
                model_answer = opt
                break

        results.append({
            "image_name": f"round{round_num}_{image_name}",
            "question": question,
            "model_output": description,
            "options": options,
            "correct_answers": correct_answers,
            "model_answer": model_answer,
            "image_type": image_type
        })

# 准备输出数据
output_data = {"data": []}
for result in results:
    output_data["data"].append({
        "image_name": result["image_name"],
        "question": result["question"],
        "model_output": result["model_output"],
        "options": result["options"],
        "correct_answers": result["correct_answers"],
        "image_type": result["image_type"]
    })

# 将字典转换为JSON字符串
json_output = json.dumps(output_data, indent=4)

# 将JSON字符串保存到文件中
output_file = "/home/liudongdong/filter_img_results/w3n2/instructblip.json"
with open(output_file, "w") as file:
    file.write(json_output)
