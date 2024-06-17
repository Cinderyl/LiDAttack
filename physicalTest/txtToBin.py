# with open(os.path.join(
#         "/home/Newdisk/liaodanxin/pcdet111/OpenPCDet/tools/final_result结果/中间bin/pointpillar/1000/10个扰动点/adv2_Car/Easy",
#         f"{file_number}.bin"), 'wb') as f:
#     # 依次将每个点的四个值转换为二进制，并写入文件中
#     for point in new_points:
#         x, y, z, intensity = point
#         f.write(struct.pack('ffff', x, y, z, intensity))
import os
import struct

# 定义输入和输出文件夹路径
input_folder = '/home/Newdisk/liaodanxin/pcdet111/OpenPCDet/phsicalTest'
output_folder = '/home/Newdisk/liaodanxin/pcdet111/OpenPCDet/phsicalTest'

# 确保输出文件夹存在，如果不存在则创建
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍历输入文件夹中的所有txt文件
for file_name in os.listdir(input_folder):
    if file_name.endswith('.txt'):
        input_file_path = os.path.join(input_folder, file_name)
        output_file_path = os.path.join(output_folder, file_name.replace('.txt', '.bin'))

        # 打开文本文件进行读取
        with open(input_file_path, 'r') as txt_file:
            lines = txt_file.readlines()

        # 将文本文件中的数据转换为二进制并写入新的文件中
        with open(output_file_path, 'wb') as bin_file:
            for line in lines:
                # 假设每行数据以空格或逗号分隔
                x, y, z, intensity = map(float, line.strip().split())  # 假设数据列之间用空格分隔

                # 将四个值转换为二进制并写入二进制文件
                bin_file.write(struct.pack('ffff', x, y, z, intensity))
