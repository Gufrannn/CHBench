import os
import json
import torch
from PIL import Image
from transformers import TextStreamer
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path

def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

# 参数设置
image_folder = "/home/liudongdong/data/multichoice_compare_P_YN_2/"
json_file = "/home/liudongdong/data/multichoice_compare_P_YN_2/images_data.json"
model_path = "/home/liudongdong/LLaVA-main/llava-v1.5-13b/"
model_name = get_model_name_from_path(model_path)
tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name, load_8bit=False, load_4bit=True, device="cuda:3")

# 加载测试数据
with open(json_file, 'r') as f:
    data = json.load(f)

results = []
y_true = []
y_pred = []

# 遍历每个数据项，处理图像和生成文本描述
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
    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = image_tensor.to(model.device, dtype=torch.float16)

    options_text = "\n".join([f"{i + 1}. {opt}" for i, opt in enumerate(options)])
    inp = f"Question: {question}\nOptions:\n{options_text}\nPlease choose the correct option only need number(e.g., '1')."
    conv = conv_templates["llava_v1"].copy()  # 使用 llava_v1 模板
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
    stop_str = conv.sep2
    keywords = [stop_str]
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    temperature = 1  # 调低温度以获得更确定的输出
    max_new_tokens = 100  # 减少生成的最大token数量

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=True,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            streamer=streamer,
            use_cache=True)

    output_text = tokenizer.decode(output_ids[0]).strip()
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
with open("/home/liudongdong/positive_multichoice_2/llava_multichoice_positive_2", "w") as file:
    file.write(json_output)
