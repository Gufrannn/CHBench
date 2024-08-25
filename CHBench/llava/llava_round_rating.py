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

evalMode = ""
#cot_w_VR TF
# 参数设置
#image_folder = "/home/liudongdong/all_select_images/type1_aigc_compare/compare_sample_PN/"
#json_file ="/home/liudongdong/all_select_images/type1_aigc_compare/compare_sample_PN/compare_P_YN.json"
#results_json_file = "/home/liudongdong/filter_img_results/wCOT_aigc_type1/minicpm.json"

image_folder = "/home/liudongdong/all_select_images/new_winnoground_type3/winnoground_type3_2_p+n/"
json_file ="/home/liudongdong/all_select_images/new_winnoground_type3/winnoground_type3_2_p+n/winoground_compare_P_YN.json"
results_json_file = "/home/liudongdong/filter_img_results/new_winnoground_type3_wcot/qwen.json"


#image_folder = "/home/liudongdong/all_select_images/type1_nature_compare/winoground_compare_PN/"
#json_file ="/home/liudongdong/all_select_images/type1_nature_compare/winoground_compare_PN/compare_P_YN.json"
#results_json_file = "/home/liudongdong/filter_img_results/wCOT_winnoground_type1/qwen.json"


#image_folder = "/home/liudongdong/all_select_images/type3_naturalIMG_compare_PN_YN/multichoice_compare_p+n/"
#json_file ="/home/liudongdong/all_select_images/type3_naturalIMG_compare_PN_YN/multichoice_compare_p+n/winoground_compare_P_YN.json"
#results_json_file = "/home/liudongdong/filter_img_results/winnoground_type3_wCOT/minicpm.json"

model_path = "/home/liudongdong/LLaVA-main/llava-v1.5-13b/"
model_name = get_model_name_from_path(model_path)

##rating manxin

tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name, load_8bit=False, load_4bit=True, device="cuda:0")

# 加载测试数据
with open(json_file, 'r') as f:
    data = json.load(f)
results=[]

with open(results_json_file, 'r') as f:
    results_json_data = json.load(f)
    
# 循环处理JSON数据五次
for round_num in range(1, 3):
    # 遍历每个数据项，处理图像和生成文本描述
    print(round_num)
    for result_item in results_json_data['data']:
        image_name = result_item['image_name']
        question = result_item['question']
        model_output = result_item['model_output']
        options = result_item['options']
            # image_type = item['image_type']

        correct_answers = result_item['correct_answers']
        image_type = result_item['image_type']
        #tmp_str = "round" + str(round_num) + "_"
        #print(tmp_str)
        tmp_img_name = image_name.strip("round_1").strip("round_2").strip("round_3").strip("round_4").strip("round_5")
        #print(tmp_img_name)
        image_path = os.path.join(image_folder, tmp_img_name)
        image = Image.open(image_path).convert('RGB')
        max_edge = max(image.size)
        image = image.resize((max_edge, max_edge))
        image_tensor = process_images([image], image_processor, model.config)
        image_tensor = image_tensor.to(model.device, dtype=torch.float16)

        options_text = "\n".join([f"{i + 1}. {opt}" for i, opt in enumerate(options)])
        if round_num == 1:
            evalMode = "IF_rating"
        elif round_num == 2:
            evalMode = "VR_rating"
        if evalMode == "IF_rating":
            inp = (f"Please rate the answer from 1 to 10 as a scoring assistant. The more the answer meets the instructions, the higher the score. The instruction \"{question}\" and the answer \"{model_output}\", please rate the answer, give only one score.")
                   # please give only one score to the overall answer.")
        elif evalMode == "VR_rating":
            inp = (f"Please act as a scoring assistant and score the answers according to the content in the given picture, with a score of 1 to 10. The more the answer fits the content in the picture, the higher the score. "
                   f"The answer is \"{model_output}\", please rate the answer, give only one score.")
        else:
            exit()
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
              #  streamer=streamer,
                use_cache=True)

        output_text = tokenizer.decode(output_ids[0]).strip()
        # 提取模型选择的选项
        rating_options = ["2","3","4","5","6","7","8","9"]

        model_rating = 1
        for i, opt in enumerate(rating_options):
            if str(i + 2) in output_text:
                if str(i + 2) in model_output:
                    model_rating = opt
                    continue
                else:
                    model_rating = opt
                    break
            elif str("out of") in output_text or str("/10") in output_text:
                if int(model_rating) > 1:
                    break
                else:
                    model_rating = 1
            elif str(10) in output_text:
                model_rating = 10
       # print(f"Model Answer for {image_name}: {model_rating}")
        #print(f"Model Output for {image_name}: {output_text}")
       
        results.append({
            "image_name": image_name,
          # "image_name": f"round{round_num}_{image_name}",
            "question": question,
            "model_output": model_output,
            "rating_model_output": output_text,
            "options": options,
            "correct_answers": correct_answers,
            "image_type": image_type,
            "model_rating": model_rating
        })

    data = {"data": []}
    for result in results:
        data["data"].append({
            "image_name": result["image_name"],
            "question": result["question"],
            "model_output":  result["model_output"],
            "rating_model_output": result["rating_model_output"],
            "options": result["options"],
            "correct_answers": result["correct_answers"],
            "image_type": result["image_type"],
            "model_rating": result["model_rating"]
        })

    # 将字典转换为JSON字符串
    json_output = json.dumps(data, indent=4)
    # 将JSON字符串保存到文件中
    #path_str = "/home/liudongdong/filter_img_results/wCOT_aigc_type1/aigcType1_llava_cpm"+ "_" + str(evalMode) + ".json"
    path_str = "/home/liudongdong/filter_img_results/new_winnoground_type3_wcot/realtype3_new_llava_qwen"+ "_" + str(evalMode) + ".json"
    #path_str = "/home/liudongdong/filter_img_results/wCOT_winnoground_type1/realType1_llava_qwen"+ "_" + str(evalMode) + ".json"
    #path_str = "/home/liudongdong/filter_img_results/winnoground_type3_wCOT/realType3_llava_owl"+ "_" + str(evalMode) + ".json"
    with open(path_str, "w") as file:
        file.write(json_output)
