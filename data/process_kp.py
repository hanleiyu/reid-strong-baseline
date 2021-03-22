# encoding: utf-8

import glob
import json
from PIL import Image
import os.path as osp
import os
import math


def get_json_data(path, h, w):
    with open(path, 'rb') as f:
        params = json.load(f)
        if len(params['people']) < 1:
            pose = []
            print(path)
        else:
            pose = params['people'][0]['pose_keypoints_2d']
            # print("before", pose)
            for i in range(0, 25):
                pose[3 * i] *= w
                pose[3 * i + 1] *= h
                pose[3 * i] = round(pose[3 * i])
                pose[3 * i + 1] = round(pose[3 * i + 1])
            # print("after", pose)
    f.close()

    return pose


def write_json_data(path, pose):
    with open(path, 'w') as r:
        json.dump(pose, r)
    r.close()


def resize_kp(a, b, name):
    img_paths = glob.glob(osp.join(data_path, name, '*.jpg'))
    for img_path in img_paths:
        img = Image.open(img_path)
        w = a / img.size[0]
        h = b / img.size[1]
        json_path = osp.join(data_path, 'kp', name, img_path[-17:-4] + '_keypoints.json')
        with open(json_path, 'rb') as f:
            write_json_data(json_path, get_json_data(json_path, h, w))


def remove(name, threshold):
    json_paths = glob.glob(osp.join(data_path, 'kp', name, '*.json'))
    kp = [1, 2, 5, 8]
    num = 0
    for p in json_paths:
        flag = 1
        with open(p, 'rb') as f:
            params = json.load(f)
            for i in range(len(kp)):
                if params[3 * kp[i] + 2] <= threshold:
                    flag = 0
                    break
            if (params[3 * 10 + 2] <= threshold or params[3 * 11 + 2] <= threshold) and (
                    params[3 * 13 + 2] <= threshold or params[3 * 14 + 2] <= threshold):
                flag = 0
        f.close()
        if flag:
            num += 1
        else:
            # print(p)
            os.remove(p)
            os.remove(osp.join(data_path, name, p[-28:-15] + '.jpg'))
    print(num)


data_path = "/home/yhl/data/VC/"


# resize_kp(8, 16, "train")
# resize_kp(8, 16, "gallery")
# resize_kp(8, 16, "query")
# os.remove("/home/yhl/data/VC/kp/train/0338-04-03-08_keypoints.json")
os.remove("/home/yhl/data/VC/train/0338-04-03-08.jpg")

# remove("train", 0.25)
# remove("gallery", 0.25)
# remove("query", 0.25)
#
# json_paths = glob.glob(osp.join(data_path, 'kp', "train", '*.json'))
# for p in json_paths:
#     flag = 1
#     with open(p, 'rb') as f:
#         params = json.load(f)
#         for i in range(25):
#             if params[3 * i] >= 8:
#                 print(p)
#                 break
#     f.close()

