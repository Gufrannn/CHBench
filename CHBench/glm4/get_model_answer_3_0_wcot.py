import json
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

# 设备选择
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# 加载分词器和模型
tokenizer = AutoTokenizer.from_pretrained("/home/liudongdong/code/GLM-4-main/glm-4-9b-chat/", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "/home/liudongdong/code/GLM-4-main/glm-4-9b-chat/",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
).to(device).eval()


# 准备生成结果的函数
def generate_output(question, options, model_output):
    query = f"Judge the choice of the model,you only need to give out the choice:Question: {question} Options: {options} Model Output: {model_output}"

    inputs = tokenizer.apply_chat_template([{"role": "user", "content": query}],
                                           add_generation_prompt=True,
                                           tokenize=True,
                                           return_tensors="pt",
                                           return_dict=True
                                           )

    inputs = inputs.to(device)
    gen_kwargs = {"max_length": 1000, "do_sample": True, "top_k": 1}

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return response


# 文件夹路径
input_folder = "/home/liudongdong/filter_img_results/woct_nature_type3/"
output_folder = "/home/liudongdong/filter_img_results/woct_nature_type3/"

# 遍历文件夹下的所有JSON文件
for filename in os.listdir(input_folder):
    if filename.endswith(".json"):
        input_filepath = os.path.join(input_folder, filename)

        # 加载JSON文件
        with open(input_filepath, "r") as file:
            data = json.load(file)

        # 遍历数据并生成输出
        for item in data["data"]:
            question = item["question"]
            options = item["options"]
            model_output = item["model_output"]
            image_type = item['image_type']
            item["generated_output"] = generate_output(question, options, model_output)

        # 生成新的输出文件名
        output_filename = f"model_{filename}"
        output_filepath = os.path.join(output_folder, output_filename)

        # 保存处理后的JSON文件
        with open(output_filepath, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=4)
