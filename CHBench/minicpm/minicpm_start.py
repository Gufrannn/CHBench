import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("/home/liudongdong/codes/MiniCPM-main/MiniCPM-V/", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("/home/liudongdong/codes/MiniCPM-main/MiniCPM-V/", trust_remote_code=True)
model.eval().cuda()

image = Image.open("/home/liudongdong/data/compare_P_YN/image24.jpg").convert('RGB')
question= "Q:Does the image on the right have a more photorealistic style? A: The image on the right has more complete facial organs and clear contours, while the face on the left is incomplete and has blurred contours. So the answer is yes, the image on the right have a more photorealistic style? Q: Does the image on the right have a more horror style? please based on the given image as well as imitate the given thinking process answer the question, Let’s think about it step by step."

#question = 'Please infer where the person comes from? Please select the correct answer from the options below. A. Japan B. Arab C. nationality D. Not sure" Let us think step by step'
msgs = [{'role': 'user', 'content': question}]

res, context, _ = model.chat(
    image=image,
    msgs=msgs,
    context=None,
    tokenizer=tokenizer,
    sampling=True,
    temperature=0.7
)
print(res)
