import os
import json
import torch
from PIL import Image
from lavis.models import load_model_and_preprocess

# 检查设备
device = torch.device("cuda:2") if torch.cuda.is_available() else "cpu"

# 加载模型和预处理函数
model, vis_processors, _ = load_model_and_preprocess(
    name="blip2_vicuna_instruct",
    model_type="vicuna7b",
    is_eval=True,
    device=device
)

# 数据文件路径
image_folder = "/home/liudongdong/all_select_images/type3_aigc_compare_PN/multichoice_compare_p+n_2/"
json_file ="/home/liudongdong/all_select_images/type3_aigc_compare_PN/multichoice_compare_p+n_2/compare_P_YN.json"
results_json_file = "/home/liudongdong/filter_img_results/wCOT_aigc_type3/mplug_owl2.json"
# 读取JSON文件
with open(json_file, 'r') as f:
    data = json.load(f)

results = []

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

        # 准备输入数据
        options_text = "\n".join([f"{i + 1}. {opt}" for i, opt in enumerate(options)])

        if round_num == 1:
            evalMode = "IF_rating"
            prompt = (f"Please rate the answer from 1 to 10 as a scoring assistant. The more the answer meets the instructions, the higher the score. The instruction \"{question}\" and the answer \"{model_output}\", please rate the answer, give only one score.")
        elif round_num == 2:
            evalMode = "VR_rating"
            prompt = (f"Please act as a scoring assistant and score the answers according to the content in the given picture, with a score of 1 to 10. The more the answer fits the content in the picture, the higher the score. "
                      f"The answer is \"{model_output}\", please rate the answer, give only one score.")
        else:
            exit()

        try:
            # 加载和处理图像
            raw_image = Image.open(image_path).convert("RGB")
            image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

            # 使用BLIP-2模型生成描述
            description = model.generate({"image": image, "prompt": prompt})
            output_text = description[0] if isinstance(description, list) else description

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

        except Exception as e:
            print(f"Error processing {image_name}: {e}")
            continue

    # 保存结果到JSON文件
    data = {"data": results}
    json_output = json.dumps(data, indent=4)
    output_path = f"/home/liudongdong/filter_img_results/wCOT_aigc_type3/realtype3_instructblip_mplug"+ "_" + str(evalMode) + ".json"
    with open(output_path, "w") as file:
        file.write(json_output)
