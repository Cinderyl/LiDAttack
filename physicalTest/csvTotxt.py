import os
import csv

# 指定包含CSV文件的文件夹路径
folder_path = '/home/Newdisk/liaodanxin/pcdet111/OpenPCDet/phsicalTest'

# 获取文件夹下所有CSV文件的文件名列表
csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]

# 遍历每个CSV文件
for csv_file in csv_files:
    # 构建完整的文件路径
    file_path = os.path.join(folder_path, csv_file)

    # 用于存储每列的数据
    selected_columns = [[] for _ in range(4)]

    # 打开CSV文件进行读取
    with open(file_path, 'r', newline='') as csv_file:
        csv_reader = csv.reader(csv_file)

        # 跳过表头
        next(csv_reader)

        # 遍历每一行
        for row in csv_reader:
            # 提取第9到12列的数字
            for i in range(4):
                selected_columns[i].append(float(row[8 + i]))

    # 构建保存结果的文本文件路径
    txt_file_path = os.path.splitext(file_path)[0] + '_selected_columns.txt'

    # 将提取的数字保存到文本文件中，按列保存
    with open(txt_file_path, 'w') as txt_file:
        for i in range(len(selected_columns[0])):
            txt_file.write('\t'.join(str(selected_columns[j][i]) for j in range(4)) + '\n')

    print(f"已从文件 {csv_file} 中提取并保存数字到 {txt_file_path}")
