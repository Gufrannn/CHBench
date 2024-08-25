import os
import json
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

# 加载模型和分词器
model = AutoModel.from_pretrained("/home/liudongdong/codes/MiniCPM-main/MiniCPM-V/", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("/home/liudongdong/codes/MiniCPM-main/MiniCPM-V/", trust_remote_code=True)
torch.cuda.empty_cache()
model.eval().cuda()

# 数据文件路径

image_folder = "/home/liudongdong/all_select_images/winnoground_type3_2/"
json_file = "/home/liudongdong/all_select_images/winnoground_type3_2/winnoground_type3_negative_2.json"
# 读取JSON文件
with open(json_file, 'r') as f:
    data = json.load(f)

results = []

# 循环处理JSON数据五次
for round_num in range(1, 6):
    for item in data['images']:
        image_name = item['image_name']
        question = item['question']
        options = item['options']
        correct_answers = item['answer']
        image_type = item['image_type']

        image_path = os.path.join(image_folder, image_name)
        image = Image.open(image_path).convert('RGB')

        # 准备问题和选项
        options_text = "\n".join([f"{i + 1}. {opt}" for i, opt in enumerate(options)])
        prompt = f"{question}\n{options_text}\nPlease select one option from above."

        # 调用模型进行推断
        msgs = [{'role': 'user', 'content': prompt}]
        res, context, _ = model.chat(
            image=image,
            msgs=msgs,
            context=None,
            tokenizer=tokenizer,
            sampling=True,
            temperature=0.7
        )

        output_text = res.strip()

        results.append({
            "image_name": f"round{round_num}_{image_name}",
            "question": question,
            "model_output": output_text,
            "options": options,
            "correct_answers": correct_answers,
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
output_file ="/home/liudongdong/filter_img_results/w3n2/minicpm.json"

with open(output_file, "w") as file:
    file.write(json_output)
