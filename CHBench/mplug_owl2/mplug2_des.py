import torch
from PIL import Image
from transformers import TextStreamer
from mplug_owl2.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from mplug_owl2.conversation import conv_templates, SeparatorStyle
from mplug_owl2.model.builder import load_pretrained_model
from mplug_owl2.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

model_path = "/home/liudongdong/codes/mPLUG-Owl/mPLUG-Owl2/mplug-owl2-llama2-7b/"
query = "Describe the image."

# 加载预训练模型和相关组件
model_name = get_model_name_from_path(model_path)
tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name, load_8bit=False, load_4bit=False, device="cuda")

conv_template = conv_templates["mplug_owl2"]

# 生成参数
temperature = 0.7
max_new_tokens = 512

# 打开输出文件
with open("image_descriptions.txt", "w") as output_file:
    # 循环处理每张图片
    for i in range(1, 101):
        image_file = f"/home/liudongdong/codes/mPLUG-Owl/mPLUG-Owl2/images/{i}.jpg"

        try:
            # 加载和处理图像
            image = Image.open(image_file).convert('RGB')
            max_edge = max(image.size)  # 推荐将图像调整为正方形以获得最佳性能
            image = image.resize((max_edge, max_edge))

            image_tensor = process_images([image], image_processor)
            image_tensor = image_tensor.to(model.device, dtype=torch.float16)

            # 准备对话模板和角色
            conv = conv_template.copy()
            inp = DEFAULT_IMAGE_TOKEN + query
            conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            # 生成输入IDs和停止条件
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
            stop_str = conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
            streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

            # 生成输出
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor,
                    do_sample=True,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                    streamer=streamer,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria])

            outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()

            # 写入输出文件
            output_file.write(f"Image {i}.jpg:\n{outputs}\n\n")

        except Exception as e:
            print(f"Error processing {image_file}: {e}")
