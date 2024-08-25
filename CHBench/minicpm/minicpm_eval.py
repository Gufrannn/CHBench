import os
import json
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics import precision_score, recall_score, f1_score,accuracy_score

# 加载模型和分词器
model = AutoModel.from_pretrained("/home/liudongdong/codes/MiniCPM-main/MiniCPM-V/", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("/home/liudongdong/codes/MiniCPM-main/MiniCPM-V/", trust_remote_code=True)
model.eval().cuda()

# 数据文件路径
image_folder = "/home/liudongdong/data/winnoground_type3_2"
json_file = "/home/liudongdong/data/winnoground_type3_2/winnoground_type3_negative_2.json"

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

    # 提取模型选择的选项
    model_answer = None
    for i, opt in enumerate(options):
        if str(i + 1) in output_text:
            model_answer = opt
            break
        elif opt in output_text:
            model_answer = opt
            break

    results.append({
        "image_name": image_name,
        "question": question,
        "model_output": output_text,
        "options": options,
        "correct_answers": correct_answers,
        "model_answer": model_answer
    })

    # 计算评估指标
    if image_type == "compare sample positive":
        y_true.append(1)
        y_pred.append(1 if model_answer in correct_answers else 0)
    elif image_type == "single sample positive":
        y_true.append(1)
        y_pred.append(1 if model_answer in correct_answers else 0)
    elif image_type == "single positive multichoice":
        y_true.append(1)
        y_pred.append(1 if model_answer in correct_answers else 0)
    elif image_type == "compare positive multichoice":
        y_true.append(1)
        y_pred.append(1 if model_answer in correct_answers else 0)
    elif image_type == "compare sample negative":
        y_true.append(0)
        y_pred.append(0 if model_answer in correct_answers else 1)
    elif image_type == "single sample negative":
        y_true.append(0)
        y_pred.append(0 if model_answer in correct_answers else 1)
    elif image_type == "compare negative multichoice":
        y_true.append(0)
        y_pred.append(0 if model_answer in correct_answers else 1)
    elif image_type == "single negative multichoice":
        y_true.append(0)
        y_pred.append(0 if model_answer in correct_answers else 1)

# 计算精度、召回率和F1分数
precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)
accuracy = accuracy_score(y_true, y_pred)
# 打印结果
for result in results:
    print(f"Image: {result['image_name']}")
    print(f"Question: {result['question']}")
    print(f"Model Output: {result['model_output']}")
    print(f"Options: {result['options']}")
    print(f"Correct Answers: {result['correct_answers']}")
    print(f"Model Answer: {result['model_answer']}")
    print()
print(f"acc Score: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
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
with open("/home/liudongdong/filter_img_results/wnt3_2/minicpm.json", "w") as file:
    file.write(json_output)