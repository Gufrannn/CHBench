# -*- coding: utf-8 -*-
import os
import json
import torch
import argparse
from PIL import Image
from sklearn.metrics import f1_score, recall_score, precision_score
from transformers import StoppingCriteriaList
from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION_Vicuna0, CONV_VISION_LLama2, StoppingCriteriaSub

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--image-folder", type=str, required=True, help="folder containing images to process.")
    parser.add_argument("--json-file", type=str, required=True, help="JSON file containing image questions.")
    parser.add_argument("--output-file", type=str, required=True, help="file to save the responses.")
    parser.add_argument(
        "--options",
        nargs="+",
        default=[],  # Add default empty list for options
        help="override some settings in the used config, the key-value pair in xxx=yyy format will be merged into config file (deprecate), change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args

def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def load_image(image_path):
    return Image.open(image_path).convert('RGB')

def main():
    args = parse_args()
    cfg = Config(args)

    model_config = cfg.model_cfg
    model_config.device_8bit = args.gpu_id
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))

    conv_dict = {'pretrain_vicuna0': CONV_VISION_Vicuna0, 'pretrain_llama2': CONV_VISION_LLama2}
    CONV_VISION = conv_dict[model_config.model_type]

    vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

    stop_words_ids = [[835], [2277, 29937]]
    stop_words_ids = [torch.tensor(ids).to(device='cuda:{}'.format(args.gpu_id)) for ids in stop_words_ids]
    stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

    chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id), stopping_criteria=stopping_criteria)

    with open(args.json_file, 'r') as f:
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

            image_path = os.path.join(args.image_folder, image_name)
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

            user_message = f"{question}\n{options}\nPlease choose the correct option."
            chat.ask(user_message, chat_state)

            llm_message = chat.answer(conv=chat_state, img_list=img_list, num_beams=1, temperature=1.0, max_new_tokens=300, max_length=2000)[0]

            model_answer = None
            for i, opt in enumerate(options):
                if opt.lower() in llm_message.lower():
                    model_answer = opt
                    break

            results.append({
                "image_name": f"round{round_num}_{image_name}",
                "question": question,
                "model_output": llm_message.lower(),
                "options": options,
                "correct_answers": correct_answers,
                "model_answer": model_answer,
                "image_type": image_type
            })

    data = {"data": []}
    for result in results:
        data["data"].append({
            "image_name": result["image_name"],
            "question": result["question"],
            "model_output": result["model_output"],
            "options": result["options"],
            "correct_answers": result["correct_answers"],
            "image_type": result["image_type"]
        })

    # 将字典转换为JSON字符串
    json_output = json.dumps(data, indent=4)

    # 将JSON字符串保存到文件中
    with open("/home/liudongdong/filter_img_results/wnt3_2/minigpt4.json", "a+") as file:
        file.write(json_output)

if __name__ == "__main__":
    main()
