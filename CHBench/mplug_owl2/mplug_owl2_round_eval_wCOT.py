# -*- coding: utf-8 -*-
import os
import json
import torch
from PIL import Image
from transformers import TextStreamer
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score

from mplug_owl2.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from mplug_owl2.conversation import conv_templates, SeparatorStyle
from mplug_owl2.model.builder import load_pretrained_model
from mplug_owl2.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

# 定义路径
image_folder = "/home/liudongdong/all_select_images/winnoground_type3_2/"
json_file = "/home/liudongdong/all_select_images/winnoground_type3_2/winnoground_type3_negative_2.json"

model_path = "/home/liudongdong/codes/mPLUG-Owl/mPLUG-Owl2/mplug-owl2-llama2-7b/"
model_name = get_model_name_from_path(model_path)

# 加载模型
tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name, load_8bit=False, load_4bit=False, device="cuda:3")

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
        max_edge = max(image.size)
        image = image.resize((max_edge, max_edge))
        image_tensor = process_images([image], image_processor)
        image_tensor = image_tensor.to(model.device, dtype=torch.float16)

        options_text = "\n".join([f"{i + 1}. {opt}" for i, opt in enumerate(options)])
        inp = f"{DEFAULT_IMAGE_TOKEN}{question}\n{options_text}\nLet us infer step by step and then select one option from above."
        conv = conv_templates["mplug_owl2"].copy()
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
        stop_str = conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        temperature = 0.7
        max_new_tokens = 512

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
               #streamer=streamer,
                use_cache=True,
                stopping_criteria=[stopping_criteria]
            )

        output_text = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()

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
output_file = "/home/liudongdong/filter_img_results/wnt3_2_wcot/mplug_owlw2.json"
with open(output_file, "w") as file:
    file.write(json_output)
