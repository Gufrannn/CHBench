import torch
from PIL import Image
from lavis.models import load_model_and_preprocess

#检查设备
device = torch.device("cuda:1") if torch.cuda.is_available() else "cpu"

#加载模型和预处理函数
model, vis_processors, _ = load_model_and_preprocess(
    name="blip2_vicuna_instruct",
    model_type="vicuna7b",
    is_eval=True,
    device=device
)

#定义图像文件路径
image_file = "/home/liudongdong/data/single_P_YN/image2.jpg"

try:
    #加载和处理图像
    raw_image = Image.open(image_file).convert("RGB")
    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

    #生成描述
    prompt = "Please infer where the person comes from?"
    description = model.generate({"image": image, "prompt": prompt})

    #打印描述
    print(f"Description for {image_file}:\n{description}")

except Exception as e:
    print(f"Error processing {image_file}: {e}")
