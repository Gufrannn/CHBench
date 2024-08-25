import os
import json
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

torch.manual_seed(1234)

# 加载tokenizer和模型
tokenizer = AutoTokenizer.from_pretrained("/home/liudongdong/codes/Qwen-VL-master/Qwen-VL-Chat/", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("/home/liudongdong/codes/Qwen-VL-master/Qwen-VL-Chat/", device_map="cuda:1", trust_remote_code=True, bf16=True).eval()

# 数据文件路径
image_folder = "/home/liudongdong/all_select_images/new_winnoground_type3/winnoground_type3_2_p+n/"
json_file ="/home/liudongdong/all_select_images/new_winnoground_type3/winnoground_type3_2_p+n/winoground_compare_P_YN.json"
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

        # 准备输入数据
        options_text = "\n".join([f"{i + 1}. {opt}" for i, opt in enumerate(options)])
        prompt = f"{question}\nOptions:\n{options_text}\nPlease choose the correct option number."

        query = tokenizer.from_list_format([
            {'image': image_path},
            {'text': prompt}
        ])

        # 调用模型进行推断
        output_text, history = model.chat(tokenizer, query=query, history=None)

        # 提取模型选择的选项
        model_answer = None
        for i, opt in enumerate(options):
            if str(i + 1) in output_text:
                model_answer = opt
                break

        results.append({
            "image_name": f"round{round_num}_{image_name}",
            "question": question,
            "model_output": output_text,
            "options": options,
            "correct_answers": correct_answers,
            "model_answer": model_answer,
            "image_type": image_type
        })

data_output = {"data": []}
for result in results:
    data_output["data"].append({
        "image_name": result["image_name"],
        "question": result["question"],
        "model_output": result["model_output"],
        "options": result["options"],
        "correct_answers": result["correct_answers"],
        "image_type": result["image_type"]
    })

# 将字典转换为JSON字符串
json_output = json.dumps(data_output, indent=4)

# 将JSON字符串保存到文件中
with open("/home/liudongdong/filter_img_results/new_winnoground_type3/qwen.json", "w") as file:
    file.write(json_output)
