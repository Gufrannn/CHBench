import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

# 加载模型和tokenizer
model_path = "/home/liudongdong/codes/MiniCPM-main/MiniCPM-V/"
model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model.eval().cuda()

# 打开输出文件
with open("image_descriptions.txt", "w") as output_file:
    # 循环处理每张图片
    for i in range(1, 101):
        image_file = f"/home/liudongdong/images/{i}.jpg"
        
        try:
            # 加载和处理图像
            image = Image.open(image_file).convert('RGB')
            question = 'Describe the image.'
            msgs = [{'role': 'user', 'content': question}]

            # 生成描述
            res, context, _ = model.chat(
                image=image,
                msgs=msgs,
                context=None,
                tokenizer=tokenizer,
                sampling=True,
                temperature=0.7
            )

            # 写入输出文件
            output_file.write(f"Image {i}.jpg:\n{res}\n\n")

        except Exception as e:
            print(f"Error processing {image_file}: {e}")
