# encoding: utf-8

import glob
import json
from PIL import Image
import os.path as osp


def get_json_data(path, h, w):
    with open(path, 'rb') as f:
        params = json.load(f)
        if len(params['people']) < 1:
            pose = []
            print(json_path)
        else:
            pose = params['people'][0]['pose_keypoints_2d']
            # print("before", pose)
            for i in range(0, 25):
                pose[3 * i] *= h
                pose[3 * i + 1] *= w
                pose[3 * i] = round(pose[3 * i])
                pose[3 * i + 1] = round(pose[3 * i + 1])
            # print("after", pose)
    f.close()

    return pose


def write_json_data(path, pose):
    with open(path, 'w') as r:
        json.dump(pose, r)
    r.close()


data_path = "/home/yhl/data/VC/"

img_paths = glob.glob(osp.join(data_path, 'train/*.jpg'))
for img_path in img_paths:
    img = Image.open(img_path)
    w = 128 / img.size[0]
    h = 256 / img.size[1]
    json_path = osp.join(data_path, 'kp/train', img_path[-17:-4] + '_keypoints.json')
    with open(json_path, 'rb') as f:
        write_json_data(json_path, get_json_data(json_path, h, w))

img_paths = glob.glob(osp.join(data_path, 'gallery/*.jpg'))
for img_path in img_paths:
    img = Image.open(img_path)
    w = 128 / img.size[0]
    h = 256 / img.size[1]
    json_path = osp.join(data_path, 'kp/gallery', img_path[-17:-4] + '_keypoints.json')
    with open(json_path, 'rb') as f:
        write_json_data(json_path, get_json_data(json_path, h, w))

img_paths = glob.glob(osp.join(data_path, 'query/*.jpg'))
for img_path in img_paths:
    img = Image.open(img_path)
    w = 128 / img.size[0]
    h = 256 / img.size[1]
    json_path = osp.join(data_path, 'kp/query', img_path[-17:-4] + '_keypoints.json')
    with open(json_path, 'rb') as f:
        write_json_data(json_path, get_json_data(json_path, h, w))

# img_path = "/home/yhl/data/VC/train/0338-04-03-08.jpg"
# img = Image.open(img_path)
# w = 128/img.size[0]
# h = 256/img.size[1]
# print(img.size)
# json_path = osp.join(data_path, 'kp', img_path[-23:-4]+'_keypoints.json')
# with open(json_path,'rb') as f:
#     write_json_data(json_path, get_json_data(json_path, h, w))
