# -*- coding: utf-8 -*-
import os
import json
from collections import Counter

# ָ���ļ���·��
folder_path = "/home/liudongdong/filter_img_results/output_w3n2/"

# ��ȡ�ļ����е�����JSON�ļ�
json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]

# ����һ���ֵ����洢ÿ���ļ��лش�����ͼ����
incorrect_images_dict = {}

# ����ÿ��JSON�ļ�
for json_file in json_files:
    file_path = os.path.join(folder_path, json_file)
    
    # ��ȡJSON�ļ�
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    # ����һ���б����洢�ش�����ͼ����
    incorrect_images = []

    # �������ݲ�������ɵ�����Ƿ���ȷ
    for item in data['data']:
        if item['generated_output'].strip() != item['correct_answers']:
            incorrect_images.append(item['image_name'])
    
    # ������洢���ֵ���
    incorrect_images_dict[json_file] = incorrect_images

# ��ȡ���д���ͼ�����ļ���
all_incorrect_images = set()
for images in incorrect_images_dict.values():
    all_incorrect_images.update(images)

# �����������ļ��ж����ֵĴ���ͼ����
common_incorrect_images = []
for image in all_incorrect_images:
    # ͳ��ÿ������ͼ�����������ļ��еĳ��ִ���
    occurrence_count = sum(image in images for images in incorrect_images_dict.values())
    # ������ִ��������ļ���������˵����ͼ���������ļ��ж�����
    if occurrence_count == len(incorrect_images_dict):
        common_incorrect_images.append(image)

# ���ÿ�������ж��еĴ���ͼ����
print("Common Incorrect Images in All Files:")
print(common_incorrect_images)
