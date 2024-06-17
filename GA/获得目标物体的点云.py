import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#加载标签文件
label_file='/data0/benke/ldx/openpcdet37/OpenPCDet/data/KITTI/object/training/label_2/000446.txt'
labels=np.loadtxt(label_file,delimiter=' ',dtype=str)

#加载点云文件
point_cloud_file='/data0/benke/ldx/openpcdet37/OpenPCDet/data/KITTI/object/training/velodyne/000446.bin'
point_cloud=np.fromfile(point_cloud_file,dtype=np.float32).reshape(-1,4)
print(point_cloud)

#遍历每个物体
for i,label in enumerate(labels):
    if label[0] =='Car':
        x,y,z,w,l,h,yaw=map(float,label[8:])
        #计算物体的8个角点
        corners=np.array([[w/2,1/2,0],
                          [-w/2,1/2,0],
                          [-w/2,-1/2,0],
                          [w/2,-1/2,0],
                          [w/2,1/2,h],
                          [-w/2,1/2,h],
                          [-w/2,-1/2,h],
                          [w/2,-1/2,h]
                          ])
        #旋转角点并平移
        rotation_matrix = np.array([[np.cos(yaw),-np.sin(yaw), 0],[np.sin(yaw),np.cos(yaw),0],[0,0,1]])
        corners=np.dot(corners,np.transpose(rotation_matrix))
        corners[:,0]+=x
        corners[:,1] +=y
        corners[:,2] += z
        #提取物体的点云
        indices=np.where((point_cloud[:,0]>x-w/2)&
                         (point_cloud[:,0]<x+w/2)&
                         (point_cloud[:,1]>y-1/2)&
                         (point_cloud[:,1]<y+1/2)&
                         (point_cloud[:,2]>z)&
                         (point_cloud[:,2]<z+h)
                         )[0]
        object_cloud=point_cloud[indices, :]
        #保存点云到txt文件中
        object_file='/data0/benke/ldx/openpcdet37/OpenPCDet/data/KITTI/object/training/提取物体的点云数据/{}.txt'.format(i)
        np.savetxt(object_file,object_cloud,delimiter=' ',fmt='%.5f')

