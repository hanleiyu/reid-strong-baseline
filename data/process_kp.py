# encoding: utf-8

import glob
import json
import os, sys
from PIL import Image
import os.path as osp

def get_json_data(json_path, h, w):
    with open(json_path, 'rb') as f:
        # 定义为只读模型，并定义名称为f
        params = json.load(f)
        # 加载json文件中的内容给params
        pose = params['People'][0]['Pose2d']
        # print("before", pose)
        # 修改内容
        for i in range(0,25):
            pose[3*i] *= h
            pose[3*i + 1] *=w
            pose[3 * i] = float('%.6f'%pose[3 * i])
            pose[3 * i + 1] = float('%.6f'%pose[3 * i + 1])
        # 打印
        # print("after", pose)
    f.close()

    return pose

def write_json_data(json_path, pose):
    # 写入json文件

    with open(json_path, 'w') as r:
        # 定义为写模式，名称定义为r
        json.dump(pose, r)
        # 将dict写入名称为r的文件中
    r.close()
    # 关闭json写模式

data_path = "/home/yhl/data/VC/"

img_paths = glob.glob(osp.join(data_path, 'train/*.jpg'))
img_paths.append(glob.glob(osp.join(data_path, 'gallery/*.jpg')))
img_paths.append(glob.glob(osp.join(data_path, 'query/*.jpg')))
for img_path in img_paths:
    img = Image.open(img_path)
    w = 128/img.size[0]
    h = 256/img.size[1]
    json_path = osp.join(data_path, 'kp', img_path[-23:-4]+'_keypoints.json')
    with open(json_path,'rb') as f:
        write_json_data(json_path, get_json_data(json_path, h, w))

# img_path = "/home/yhl/data/VC/train/0002-01-02-01.jpg"
# img = Image.open(img_path)
# w = 128/img.size[0]
# h = 256/img.size[1]
# print(img.size)
# json_path = osp.join(data_path, 'kp', img_path[-23:-4]+'_keypoints.json')
# with open(json_path,'rb') as f:
#     write_json_data(json_path, get_json_data(json_path, h, w))

