import os
import json
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

torch.manual_seed(1234)

# 加载tokenizer和模型
tokenizer = AutoTokenizer.from_pretrained("/home/liudongdong/code/trans_from_4567/Qwen-VL-master/Qwen-VL-Chat/", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("/home/liudongdong/code/trans_from_4567/Qwen-VL-master/Qwen-VL-Chat/", device_map="cuda:0", trust_remote_code=True, bf16=True).eval()

# 数据文件路径

image_folder = "/home/liudongdong/all_select_images/type1_aigc_compare/compare_sample_PN/"
json_file ="/home/liudongdong/all_select_images/type1_aigc_compare/compare_sample_PN/compare_P_YN.json"

#image_folder = "/home/liudongdong/data/type3_naturalIMG_compare_PN_YN/multichoice_compare_p+n/"
#json_file ="/home/liudongdong/data/type3_naturalIMG_compare_PN_YN/multichoice_compare_p+n/winoground_compare_P_YN.json"

##rating manxin
results_json_file = "/home/liudongdong/filter_img_results/wCOT_aigc_type1/mplug_owl2.json"

# 读取JSON文件
with open(json_file, 'r') as f:
    data = json.load(f)

results = []


with open(results_json_file, 'r') as f:
    results_json_data = json.load(f)
    
# 循环处理JSON数据2次 IF VR
for round_num in range(1, 3):
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

        # 准备输入数据
        options_text = "\n".join([f"{i + 1}. {opt}" for i, opt in enumerate(options)])
 
        if round_num == 1:
            evalMode = "IF_rating"
        elif round_num == 2:
            evalMode = "VR_rating"
        if evalMode == "IF_rating":
            prompt = (f"Please rate the answer from 1 to 10 as a scoring assistant. The more the answer meets the instructions, the higher the score. The instruction \"{question}\" and the answer \"{model_output}\", please rate the answer, give only one score.")
                   # please give only one score to the overall answer.")
        elif evalMode == "VR_rating":
            prompt = (f"Please act as a scoring assistant and score the answers according to the content in the given picture, with a score of 1 to 10. The more the answer fits the content in the picture, the higher the score. "
                   f"The answer is \"{model_output}\", please rate the answer, give only one score.")
        else:
            exit()

        query = tokenizer.from_list_format([
            {'image': image_path},
            {'text': prompt}
        ])

        # 调用模型进行推断
        output_text, history = model.chat(tokenizer, query=query, history=None)

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
    path_str = "/home/liudongdong/filter_img_results/wCOT_aigc_type1/aigc_type1_qwen_mplug_" + str(evalMode) + ".json"
    with open(path_str, "w") as file:
        file.write(json_output)


