# -*- coding: utf-8 -*-
import json
import os

# �����ļ���·��
folder_path = "/home/liudongdong/filter_img_results/4_type_pn/aigc/"

# �򿪽���ļ�����д��
output_path = os.path.join(folder_path, "evaluation_results.txt")
with open(output_path, 'w', encoding='utf-8') as output_file:

    # �����ļ����е�����JSON�ļ�
    for filename in os.listdir(folder_path):
        if filename.endswith(".json") and filename.startswith('model_'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as file:
                data = json.load(file)
            
            # ��ʼ����ǰ�ļ��ļ���
            tp = fp = tn = fn = 0
            
            # ����ÿ����Ŀ������ͼ�����͸��¼���
            for entry in data['data']:
                image_type = entry['image_type']
                correct_answers = entry['correct_answers'].strip()
                generated_output = entry['generated_output'].strip()
                model_answer = None
                for i, opt in enumerate(entry["options"]):
                  if str(i + 1) in generated_output:
                      model_answer = opt
                      break
                  elif opt in generated_output:
                      model_answer=opt
                      break
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
                elif image_type == "single positive multichoice":
                    if model_answer == correct_answers:
                        tp += 1
                    else:
                        fp += 1
                elif image_type == "compare positive multichoice":
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
                    if model_answer == correct_answers:
                        tn += 1
                    else:
                        fp += 1
                elif image_type == "compare negative multichoice":
                    if model_answer == correct_answers:
                        tn += 1
                    else:
                        fp += 1
                elif image_type == "single negative multichoice":
                    if model_answer == correct_answers:
                        tn += 1
                    else:
                        fp += 1
            
            # ���㵱ǰ�ļ�������ָ��
            precision = tp / (tp + fp) if (tp + fp) != 0 else 0
            recall = tp / (tp + fn) if (tp + fn) != 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) != 0 else 0

            # ����ǰ�ļ��Ľ��д������ļ�
            output_file.write(f"Results for {filename}:\n")
            output_file.write(f"Precision: {precision:.2f}\n")
            output_file.write(f"Recall: {recall:.2f}\n")
            output_file.write(f"F1 Score: {f1:.2f}\n")
            output_file.write(f"Accuracy: {accuracy:.2f}\n\n")

print("save_in_evaluation_results.txt")
