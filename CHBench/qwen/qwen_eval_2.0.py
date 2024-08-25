import os
import json
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

torch.manual_seed(1234)

# 加载tokenizer和模型
tokenizer = AutoTokenizer.from_pretrained("/home/liudongdong/codes/Qwen-VL-master/Qwen-VL-Chat/", trust_remote_code=True)
#model = AutoModelForCausalLM.from_pretrained(
#   "/home/liudongdong/codes/Qwen-VL-master/Qwen-VL-Chat/", 
#    trust_remote_code=True, 
#    bf16=True
#).to(device).eval()
model = AutoModelForCausalLM.from_pretrained("/home/liudongdong/codes/Qwen-VL-master/Qwen-VL-Chat/", device_map="cuda:1", trust_remote_code=True, bf16=True).eval()
# 数据文件路径
image_folder = "/home/liudongdong/data/multichoice_compare_P_YN_2/"
json_file = "/home/liudongdong/data/multichoice_compare_P_YN_2/images_data.json"

# 读取JSON文件
with open(json_file, 'r') as f:
    data = json.load(f)

results = []
y_true = []
y_pred = []

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
    #inputs = tokenizer(query, return_tensors='pt')
    #inputs = {key: value.to(device) for key, value in inputs.items()}

    # 调用模型进行推断
    #pred = model.generate(**inputs, max_new_tokens=50)
    #output_text = tokenizer.decode(pred.cpu()[0], skip_special_tokens=True).strip()
    output_text, history = model.chat(tokenizer, query=query, history=None)

    results.append({
        "image_name": image_name,
        "question": question,
        "model_output": output_text,
        "options": options,
        "correct_answers": correct_answers,
    })
data = {"data": []}
for result in results:
    data["data"].append({
        "image_name": result["image_name"],
        "question": result["question"],
        "model_output": result["model_output"],
        "options": result["options"],
        "correct_answers": result["correct_answers"],
        })

    # 将字典转换为JSON字符串
json_output = json.dumps(data, indent=4)

    # 将JSON字符串保存到文件中
with open("/home/liudongdong/positive_multichoice_2/qwen_multichoice_positive_2", "w") as file:
    file.write(json_output)