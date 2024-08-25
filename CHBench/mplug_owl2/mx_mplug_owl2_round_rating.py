import os
import json
import torch
from PIL import Image
from transformers import TextStreamer
from mplug_owl2.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from mplug_owl2.conversation import conv_templates, SeparatorStyle
from mplug_owl2.model.builder import load_pretrained_model
from mplug_owl2.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

# 数据文件路径
image_folder = "/home/liudongdong/all_select_images/type1_aigc_compare/compare_sample_PN/"
json_file ="/home/liudongdong/all_select_images/type1_aigc_compare/compare_sample_PN/compare_P_YN.json"
results_json_file = "/home/liudongdong/filter_img_results/wCOT_aigc_type1/mplug_owl2.json"
# 模型路径
model_path = "/home/liudongdong/codes/mPLUG-Owl/mPLUG-Owl2/mplug-owl2-llama2-7b/"

# 加载模型和处理工具
model_name = get_model_name_from_path(model_path)
tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name, load_8bit=False, load_4bit=False, device="cuda")

results = []

# 读取JSON文件
with open(json_file, 'r') as f:
    data = json.load(f)

with open(results_json_file, 'r') as f:
    results_json_data = json.load(f)

# 循环处理JSON数据2次 IF 和 VR 模式
for round_num in range(1, 3):
    for result_item in results_json_data['data']:
        image_name = result_item['image_name']
        question = result_item['question']
        model_output = result_item['model_output']
        options = result_item['options']
        correct_answers = result_item['correct_answers']
        image_type = result_item['image_type']

        tmp_img_name = image_name.strip("round_1").strip("round_2").strip("round_3").strip("round_4").strip("round_5")
        image_path = os.path.join(image_folder, tmp_img_name)

        image = Image.open(image_path).convert('RGB')
        max_edge = max(image.size)
        image = image.resize((max_edge, max_edge))
        image_tensor = process_images([image], image_processor)
        image_tensor = image_tensor.to(model.device, dtype=torch.float16)

        options_text = "\n".join([f"{i + 1}. {opt}" for i, opt in enumerate(options)])

        # 准备输入数据
        if round_num == 1:
            evalMode = "IF_rating"
            inp = f"Please rate the answer from 1 to 10 as a scoring assistant. The more the answer meets the instructions, the higher the score. The instruction \"{question}\" and the answer \"{model_output}\", please rate the answer, give only one score."
        elif round_num == 2:
            evalMode = "VR_rating"
            inp = f"Please act as a scoring assistant and score the answers according to the content in the given picture, with a score of 1 to 10. The more the answer fits the content in the picture, the higher the score. The answer is \"{model_output}\", please rate the answer, give only one score."
        else:
            exit()
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
        max_new_tokens = 1000
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

        # 提取模型选择的选项
        rating_options = ["2","3","4","5","6","7","8","9"]

        model_rating = 1
        for i, opt in enumerate(rating_options):
            if str(i + 2) in output_text:
                model_rating = opt
                break
            elif "out of" in output_text or "/10" in output_text:
                model_rating = int(opt) if model_rating > 1 else 1
                break
            elif "10" in output_text:
                model_rating = 10

        results.append({
            "image_name": image_name,
            "question": question,
            "model_output": model_output,
            "rating_model_output": output_text,
            "options": options,
            "correct_answers": correct_answers,
            "image_type": image_type,
            "model_rating": model_rating
        })

    # 保存结果到JSON文件
    data = {"data": results}
    json_output = json.dumps(data, indent=4)
    output_file = "/home/liudongdong/filter_img_results/wCOT_aigc_type1/mplug_owl2"+ "_" + str(evalMode) + ".json"
    with open(output_path, "w") as file:
        file.write(json_output)
