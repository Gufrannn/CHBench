import torch
from PIL import Image
from lavis.models import load_model_and_preprocess

# 检查设备
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

# 加载模型和预处理函数
model, vis_processors, _ = load_model_and_preprocess(
    name="blip2_vicuna_instruct",
    model_type="vicuna7b",
    is_eval=True,
    device=device
)

# 打开输出文件
with open("image_descriptions.txt", "w") as output_file:
    # 循环处理每张图片
    for i in range(1, 101):
        image_file = f"/home/liudongdong/images/{i}.jpg"

        try:
            # 加载和处理图像
            raw_image = Image.open(image_file).convert("RGB")
            image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

            # 生成描述
            prompt = "Describe the image."
            description = model.generate({"image": image, "prompt": prompt})

            # 写入输出文件
            output_file.write(f"Image {i}.jpg:\n{description}\n\n")

        except Exception as e:
            print(f"Error processing {image_file}: {e}")
