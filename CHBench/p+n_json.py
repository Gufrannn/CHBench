import json


def rename_images_in_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    for image in data['images']:
        file_name = image['image_name']
        base_name, extension = file_name.rsplit('.', 1)

        if base_name.startswith('image') and base_name[5:].isdigit():
            new_number = int(base_name[5:]) +170
            new_file_name = f'image{new_number}.{extension}'
            image['image_name'] = new_file_name

    with open(json_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=2, ensure_ascii=False)
        print(f'Renamed images in {json_path}')


# 指定JSON文件路径
json_path = r"C:\Users\Administrator\Desktop\new_winnoground_type3\winnoground_type3_2_p+n\winoground_compare_P_YN.json"
rename_images_in_json(json_path)
