import codecs
import numpy as np
# def list2txt(list, path):
#     file = open(path, 'w', encoding="utf-8")
#     for l in list:
#         l = str(l)  # 强制转换
#         if l[-1] != '\n':
#             l = l + '\n'
#         file.write(l)
#     file.close()

f = codecs.open('446.txt', mode='r')  # 打开txt文件，以‘utf-8’编码读取,box3d是从kitti_objec里面获得的
line = f.readline()   # 以行的形式进行读取文件
list1 = []
list2 = []
list3 = []
while line:
    a = line.split()
    b1 = float(a[0])   # 这是选取需要读取的位数
    b2 = float(a[1])
    b3 = float(a[2])
    list1.append(b1)  # 将其添加在列表之中
    list2.append(b2)
    list3.append(b3)
    line = f.readline()
f.close()
# print(list1)
# print(list2)
# print(list3)

f1 = codecs.open('/data0/benke/ldx/openpcdet37/OpenPCDet/data/KITTI/object/training/bintotxt/1.txt', mode='r')  # 打开txt文件，以‘utf-8’编码读取
line4 = f1.readline()   # 以行的形式进行读取文件
str1=[]
str2=[]
while line4:
    c = line4.split()
    d1 = float(c[0] )  # 这是选取需要读取的位数
    d2 = float(c[1])
    d3 = float(c[2])
    d4 = float(c[3])
    if(min(list1)<d1<max(list1) and min(list2)<d2<max(list2) and min(list3)<d3<max(list3)):
        # d1=d1+10
        str2.append([d1, d2, d3,d4])
        np.savetxt(('提取.txt'), str2, fmt="%.6f", delimiter=" ")
        # d2=d2-5


    str1.append([d1,d2,d3,d4])
    line4 = f1.readline()
f.close()
# np.savetxt(('008moved21.txt'), str1, fmt="%.6f", delimiter=" ")
# list2txt(str1, '34.txt')


