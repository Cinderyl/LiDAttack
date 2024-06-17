import matplotlib
import numpy as np
import os;
from  matplotlib import pyplot as plt
import math

#通用设置
matplotlib.rc('axes', facecolor = 'white')
matplotlib.rc('figure', figsize = (6, 4))
matplotlib.rc('axes', grid = False)
#数据及线属性
# pointrcnn
# easy
x1=[0,5,10,20,40]
# y1=[0,0.78,0.92,0.96,0.99]
# plt.plot(x1, y1,'*-r',label='PointRcnn')
# moderate
# y2=[0,0.86,0.94,0.96,1.0]
# plt.plot(x1, y2,'*-r',label='PointRcnn')
# # hard
y3=[0,0.92,1.0,1.0,1.0]
plt.plot(x1, y3,'*-r',label='PointRcnn')

# pointpillar
# easy
# y4=[0,0.5,0.88,0.95,1.0]
# plt.plot(x1, y4,'*-b',label='PointPillar')
# # moderate
# y5=[0,0.82,0.98,0.99,1.0]
# plt.plot(x1, y5,'*-b',label='PointPillar')
# # hard
y6=[0,0.95,0.98,1.0,1.0]
plt.plot(x1, y6,'*-b',label='PointPillar')

# pv
# easy
y7=[0,0.43,0.75,0.85,0.90]
plt.plot(x1, y7,'*-g',label='PV_RCNN_plusplus')
# # modeatre
# y8=[0,0.66,0.86,0.93,0.97]
# plt.plot(x1, y8,'*-g',label='PV_RCNN_plusplus')
# # hard
y9=[0,0.87,0.89,0.95,0.99]
plt.plot(x1, y9,'*-g',label='PV_RCNN_plusplus')

#标题设置
plt.title('Cyclist_Hard')
plt.xlabel('perturbation_points')
plt.ylabel('success Attack Rate')
plt.legend(loc="upper left")
plt.savefig(os.path.join(
    "/home/Newdisk/liaodanxin/pcdet111/图","Cyclist_Hard_1006.jpg"))
plt.show()
