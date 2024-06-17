# import numpy as np
#
# # 假设 box 是一个8行3列的数组
# box = np.array([[1, 2, 3],
#                 [4, 5, 6]
#                 ])
#
# # 计算每一列的平均值
# column_means = np.mean(box, axis=0)
#
# print("每一列的平均值：")
# print(column_means)
#
#
# box=np.loadtxt('边界框/Car/easy/000002.txt')
# column_means = np.mean(box, axis=0)
import os

# 文件夹路径，将其替换为您实际的文件夹路径
folder_path = "G:\Desktop\激光数据整理\Cyclist\easy"

# 获取文件夹中所有文件的文件名列表
file_names = os.listdir(folder_path)

# 过滤出所有的txt文件
txt_files = [file_name for file_name in file_names if file_name.endswith(".txt")]

# 循环读取每个txt文件的内容
for txt_file in txt_files:
    file_path = os.path.join(folder_path, txt_file)
    try:
        with open(file_path, "r") as txt_file:
            content = txt_file.read()
            # 在这里对content进行处理，比如打印文件内容
            print(f"--- {txt_file} ---")
            print(content)
    except FileNotFoundError:
        print(f"文件 '{txt_file}' 不存在。")
