import os
import random
import json
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from transformers import StoppingCriteriaList
from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION_Vicuna0, CONV_VISION_LLama2, StoppingCriteriaSub
from PIL import Image

# ===== 配置路径和参数 =====
cfg_path = "/home/liudongdong/MiniGPT-4-main/eval_configs/minigpt4_eval.yaml"
image_folder = "/home/liudongdong/all_select_images/type1_aigc_compare/compare_sample_PN/"
json_file = "/home/liudongdong/all_select_images/type1_aigc_compare/compare_sample_PN/compare_P_YN.json"
results_json_file = "/home/liudongdong/filter_img_results/wCOT_aigc_type1/cpm.json"
gpu_id = 2

def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

def load_image(image_path):
    return Image.open(image_path).convert('RGB')

def main():
    conv_dict = {'pretrain_vicuna0': CONV_VISION_Vicuna0,
                 'pretrain_llama2': CONV_VISION_LLama2}

    print('Initializing Chat')

    # 加载配置
    cfg = Config(cfg_path)

    model_config = cfg.model_cfg
    model_config.device_8bit = gpu_id
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to('cuda:{}'.format(gpu_id))

    CONV_VISION = conv_dict[model_config.model_type]

    vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

    stop_words_ids = [[835], [2277, 29937]]
    stop_words_ids = [torch.tensor(ids).to(device='cuda:{}'.format(gpu_id)) for ids in stop_words_ids]
    stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

    chat = Chat(model, vis_processor, device='cuda:{}'.format(gpu_id), stopping_criteria=stopping_criteria)
    print('Initialization Finished')

    # 读取数据文件
    with open(json_file, 'r') as f:
        data = json.load(f)

    results = []
    with open(results_json_file, 'r') as f:
        results_json_data = json.load(f)

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

            # 处理图片
            if not os.path.exists(image_path):
                print(f"Image {image_path} does not exist.")
                continue

            try:
                image = load_image(image_path)
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
                continue

            chat_state = CONV_VISION.copy()
            img_list = []
            chat.upload_img(image, chat_state, img_list)
            chat.encode_img(img_list)

            if round_num == 1:
                evalMode = "IF_rating"
                prompt = (f"Please rate the answer from 1 to 10 as a scoring assistant. The more the answer meets the instructions, the higher the score. The instruction \"{question}\" and the answer \"{model_output}\", please rate the answer, give only one score.")
            elif round_num == 2:
                evalMode = "VR_rating"
                prompt = (f"Please act as a scoring assistant and score the answers according to the content in the given picture, with a score of 1 to 10. The more the answer fits the content in the picture, the higher the score. "
                          f"The answer is \"{model_output}\", please rate the answer, give only one score.")
            else:
                continue

            chat.ask(prompt, chat_state)

            llm_message = chat.answer(conv=chat_state,
                                      img_list=img_list,
                                      num_beams=1,
                                      temperature=1.0,
                                      max_new_tokens=300,
                                      max_length=2000)[0]

            # 根据模型输出提取评分
            rating_options = ["2","3","4","5","6","7","8","9"]
            model_rating = 1
            for i, opt in enumerate(rating_options):
                if str(i + 2) in llm_message:
                    if str(i + 2) in model_output:
                        model_rating = opt
                        continue
                    else:
                        model_rating = opt
                        break
                elif "out of" in llm_message or "/10" in llm_message:
                    if int(model_rating) > 1:
                        break
                    else:
                        model_rating = 1
                elif "10" in llm_message:
                    model_rating = 10

            results.append({
                "image_name": image_name,
                "question": question,
                "model_output": model_output,
                "rating_model_output": llm_message,
                "options": options,
                "correct_answers": correct_answers,
                "image_type": image_type,
                "model_rating": model_rating
            })

        # 保存结果
        data = {"data": []}
        for result in results:
            data["data"].append({
                "image_name": result["image_name"],
                "question": result["question"],
                "model_output": result["model_output"],
                "rating_model_output": result["rating_model_output"],
                "options": result["options"],
                "correct_answers": result["correct_answers"],
                "image_type": result["image_type"],
                "model_rating": result["model_rating"]
            })
        json_output = json.dumps(data, indent=4)
        output_file_path = f"/home/liudongdong/filter_img_results/wCOT_aigc_type1/minigpt4_{evalMode}.json"
        with open(output_file_path, "w") as file:
            file.write(json_output)

if __name__ == "__main__":
    main()
