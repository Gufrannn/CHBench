import json
import os

# 定义处理单个文件的函数
def process_json_file(file_path):
    # 初始化当前文件的计数器
    tp, fp, fn, tn = 0, 0, 0, 0  

    # 打开并读取JSON文件
    with open(file_path, 'r') as file:
        data = json.load(file)["data"]

        for entry in data:
            image_type = entry["image_type"]
            correct_answers = entry["correct_answers"]
            generated_output = entry["generated_output"]
            model_answer = None

            # 如果 generated_output 有多个答案，则直接判定为错误
            if len(generated_output) > 1:
                model_answer = "wrong"  # 强制设为错误
            else:
                # 查找模型预测的答案
                for i, opt in enumerate(entry["options"]):
                    if str(i + 1) in generated_output:
                        model_answer = opt
                        break
                    elif opt in generated_output:
                        model_answer = opt
                        break

            # 根据 image_type 更新计数器
            if image_type == "compare sample positive":
                if model_answer == correct_answers:
                    tp += 1
                else:
                    fn += 1
            elif image_type == "single sample positive":
                if model_answer == correct_answers:
                    tp += 1
                else:
                    fn += 1
            elif image_type == "single positive multichoice" or image_type == "compare positive multichoice":
                if model_answer == correct_answers:
                    tp += 1
                else:
                    fp += 1
            elif image_type == "compare sample negative":
                if model_answer == correct_answers:
                    tn += 1
                else:
                    fp += 1
            elif image_type == "single sample negative":
                if model_answer == correct_answers():
                    tn += 1
                else:
                    fp += 1
            elif image_type == "compare negative multichoice" or image_type == "single negative multichoice":
                if model_answer == correct_answers:
                    tn += 1
                else:
                    fp += 1

        # 计算当前文件的评价指标
        precision = tp / (tp + fp) if (tp + fp) != 0 else 0
        recall = tp / (tp + fn) if (tp + fn) != 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) != 0 else 0

        # 返回当前文件的结果
        return {
            "file_name": os.path.basename(file_path),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy
        }

# 主函数，处理文件夹下的所有JSON文件并将结果输出为TXT文件
def process_json_folder(folder_path, output_file):
    results = []

    # 遍历文件夹下的所有JSON文件
    for file_name in os.listdir(folder_path):
        # 只处理以 "final_" 开头的 JSON 文件
        if file_name.endswith('.json') and file_name.startswith('final_'):
            file_path = os.path.join(folder_path, file_name)
            result = process_json_file(file_path)
            results.append(result)

    # 将所有结果写入TXT文件
    with open(output_file, 'w') as output:
        for result in results:
            output.write(f"File: {result['file_name']}\n")
            output.write(f"  Precision: {result['precision']:.2f}\n")
            output.write(f"  Recall: {result['recall']:.2f}\n")
            output.write(f"  F1 Score: {result['f1']:.2f}\n")
            output.write(f"  Accuracy: {result['accuracy']:.2f}\n")
            output.write("\n")

# 使用示例
folder_path ="/home/liudongdong/filter_img_results/woct_nature_type3/"
output_file = os.path.join(folder_path, 'evaluation_results.txt')  # 输出文件路径
process_json_folder(folder_path, output_file)
