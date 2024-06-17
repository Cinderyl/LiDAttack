# import pickle
# with open('/data0/benke/ldx/openpcdet37/OpenPCDet/output/cfgs/kitti_models/pointrcnn/default/eval/eval_all_default/default/epoch_51/val/result.pkl', 'rb') as f:
#     datadict = pickle.load(f, encoding='latin1')
# f.close()
# print(datadict)
import pickle

path = '/data0/benke/ldx/openpcdet37/OpenPCDet/output/cfgs/kitti_models/pointrcnn/default/eval/eval_all_default/default/epoch_51/val/result.pkl'  # path='/root/……/aus_openface.pkl'   pkl文件所在路径

f = open(path, 'rb')
data = pickle.load(f)

print(data)
print(len(data))

