# 如何用Python统计data/iu/images/iu_2image/images目录下面所有文件夹中的图片的总的数量
# 1. os.listdir()获取目录下面的所有文件夹
# 2. 遍历所有文件夹，获取每个文件夹下面的图片数量
# 3. 统计所有文件夹下面的图片数量
path = "data/iu/images/iu_2image/images"
import os

# 1. os.listdir()获取目录下面的所有文件夹
max_number = 0
min_number = 5
dir_list = os.listdir(path)
dir_name_5=[]
dir_name_4=[]
dir_name_3=[]

# 2. 遍历所有文件夹，获取每个文件夹下面的图片数量
total = 0
for dir in dir_list:
    dir_path = os.path.join(path, dir)
    file_list = os.listdir(dir_path)
    if len(file_list) > max_number:
        max_number = len(file_list)
    if len(file_list) < min_number:
        min_number = len(file_list)
    total += len(file_list)
    if len(file_list)==5:
        dir_name_5.append(dir)
    if len(file_list)==4:
        dir_name_4.append(dir)
    if len(file_list)==3:
        dir_name_3.append(dir)
# 3. 统计所有文件夹下面的图片数量
print('总共有{}张图片'.format(total))
print("包含{}个文件夹,每个文件夹最多包含{}张图像,至少包含{}张图像".format(len(dir_list), max_number, min_number))
print('包含5张图片的文件夹有：',dir_name_5)
print('包含4张图片的文件夹有：',dir_name_4)
print('包含3张图片的文件夹有：',dir_name_3)
# 读取data/iu/images/iu_2image目录下面的annotation.json文件，

import json

# 1. 读取json文件
json_path = "data/iu/images/iu_2image/annotation.json"
# 2. 将json文件转换成python的字典
with open(json_path, 'r') as f:
    json_dict = json.load(f)
# 3. 获取字典中的数据
first_train_example = json_dict.get('train')[0]
print('id:', first_train_example.get('id'))
print('report:', first_train_example.get('report'))
print('image_path:', first_train_example.get('image_path'))
print('split:', first_train_example.get('split'))
