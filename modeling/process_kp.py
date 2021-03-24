# encoding: utf-8

import glob
import json
from PIL import Image
import os.path as osp
import os
import math
import numpy as np
import torch


def get_json_data(path, n1, n2):
    t1 = []
    t2 = []
    with open(path, 'rb') as f:
        params = json.load(f)
        if len(params) > 0:
            t1 = [params[3 * n1], params[3 * n1 + 1]]
            t2 = [params[3 * n2], params[3 * n2 + 1]]
    f.close()

    return t1, t2


def get_leg_data(path):
    t1 = []
    t2 = []
    t3 = []
    t4 = []
    with open(path, 'rb') as f:
        params = json.load(f)
        if len(params) > 0:
            if params[3 * 10 + 2] != 0 and params[3 * 11 + 2] != 0:
                t1 = [params[3 * 10], params[3 * 10 + 1]]
                t2 = [params[3 * 11], params[3 * 11 + 1]]
            if params[3 * 13 + 2] != 0 and params[3 * 14 + 2] != 0:
                t3 = [params[3 * 13], params[3 * 13 + 1]]
                t4 = [params[3 * 14], params[3 * 14 + 1]]
    f.close()

    return t1, t2, t3, t4


def get_arm_data(path):
    t1 = []
    t2 = []
    t3 = []
    t4 = []
    with open(path, 'rb') as f:
        params = json.load(f)
        if len(params) > 0:
            if params[3 * 10 + 2] != 0 and params[3 * 11 + 2] != 0:
                t1 = [params[3 * 10], params[3 * 10 + 1]]
                t2 = [params[3 * 11], params[3 * 11 + 1]]
            if params[3 * 13 + 2] != 0 and params[3 * 14 + 2] != 0:
                t3 = [params[3 * 13], params[3 * 13 + 1]]
                t4 = [params[3 * 14], params[3 * 14 + 1]]
    f.close()

    return t1, t2, t3, t4


def get_thigh_data(path):
    t1 = []
    t2 = []
    t3 = []
    t4 = []
    with open(path, 'rb') as f:
        params = json.load(f)
        if len(params) > 0:
            if params[3 * 8 + 2] != 0 and params[3 * 10 + 2] != 0:
                t1 = [params[3 * 8], params[3 * 8 + 1]]
                t2 = [params[3 * 10], params[3 * 10 + 1]]
            if params[3 * 8 + 2] != 0 and params[3 * 13 + 2] != 0:
                t3 = [params[3 * 8], params[3 * 8 + 1]]
                t4 = [params[3 * 13], params[3 * 13 + 1]]
    f.close()

    return t1, t2, t3, t4

def dda_line_points(pt1, pt2):
    line = []
    if len(pt1) > 0:
        x1, y1 = pt1[0], pt1[1]
        x2, y2 = pt2[0], pt2[1]

        dx = x2 - x1
        dy = y2 - y1

        if abs(dx) > abs(dy):
            steps = abs(dx)
        else:
            steps = abs(dy)

        if steps == 0 :
            return [pt1, pt2]
        else:
            xinc = dx / steps
            yinc = dy / steps
            x = x1
            y = y1

        for k in range(steps + 1):
                line.append([math.floor(x + 0.5), math.floor(y + 0.5)])
                x += xinc
                y += yinc

    return line


def resize_json_data(path, h, w):
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
                pose[3 * i] = math.floor(pose[3 * i])
                pose[3 * i + 1] = math.floor(pose[3 * i + 1])
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
            write_json_data(json_path, resize_json_data(json_path, h, w))


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


def cal_mask(p, a, b):
    if a == 0:
        pt1, pt2, pt3, pt4 = get_leg_data(p)
        line = dda_line_points(pt1, pt2) + dda_line_points(pt3, pt4)
    else:
        pt1, pt2 = get_json_data(p, a, b)
        line = dda_line_points(pt1, pt2)

    a = torch.zeros(size=(16, 8))
    for i in range(len(line)):
        if line[i][0] <= 7 and line[i][1] <= 15 :
            a[line[i][1]][line[i][0]] = 1
        else:
            print(p)
    mask = torch.unsqueeze(a, 0)
    return mask

def cal_kp(a, b):
    dictionary = {}
    json_paths = glob.glob(osp.join(data_path, 'kp', "train", '*.json'))
    for p in json_paths:
        img = p[-28:-15]
        dictionary.update({img: cal_mask(p, a, b)})
    json_paths = glob.glob(osp.join(data_path, 'kp', "gallery", '*.json'))
    for p in json_paths:
        img = p[-28:-15]
        dictionary.update({img: cal_mask(p, a, b)})
    json_paths = glob.glob(osp.join(data_path, 'kp', "query", '*.json'))
    for p in json_paths:
        img = p[-28:-15]
        dictionary.update({img: cal_mask(p, a, b)})
    return dictionary


def save_kp():
    # mask1 = cal_kp(1, 8)
    # mask2 = cal_kp(2, 5)
    # mask3 = cal_kp(8, 10)
    # mask4 = cal_kp(8, 13)
    mask5 = cal_kp(0, 0)
    mask5 = cal_kp(810, 0)
    # torch.save(mask1, 'mask1.pt')
    # torch.save(mask2, 'mask2.pt')
    # torch.save(mask3, 'mask3.pt')
    # torch.save(mask4, 'mask4.pt')
    torch.save(mask5, 'mask5.pt')


data_path = "/home/yhl/data/"
# save_kp()
# a = torch.randn(4,4)
# b = torch.randn(4,4)
# d = {"1":a}
# d.update({"2":b})
# torch.save(d, 'a.pt')
#
# b = torch.load("a.pt")
# print(b["1"])
# b = torch.load("a.pt", map_location=torch.device('cuda'))
# print(b["1"])
# resize_kp(8, 16, "train")
# resize_kp(8, 16, "gallery")
# resize_kp(8, 16, "query")
# os.remove("/home/yhl/data/VC/kp/train/0338-04-03-08_keypoints.json")
# os.remove("/home/yhl/data/VC/train/0338-04-03-08.jpg")

# remove("train", 0.25)
# remove("gallery", 0.25)
# remove("query", 0.25)

# json_paths = glob.glob(osp.join(data_path, 'kp', "train", '*.json'))
# for p in json_paths:
#     flag = 1
#     with open(p, 'rb') as f:
#         params = json.load(f)
#         for i in range(25):
#             if params[3 * i+1] > 16:
#                 print(p)
#                 break
#     f.close()

# img = Image.open("/home/yhl/data/VC/query/0065-02-01-01.jpg")
# print(img.size)
# cal_mask("/home/yhl/data/VC/kp/query/0065-02-01-01_keypoints.json", 0, 0)

# # Save
# path = ['1', '2', '3', '4']
# dictionary = {}
# a = {path[0]: {'path': 'a', 'size': '10'}, path[1]: {'path': 'b', 'size': '20'}}
# b = {path[2]: {'path': 'a', 'size': '10'}, path[3]: {'path': 'b', 'size': '20'}}
# dictionary.update(a)
# dictionary.update(b)
# dictionary['1'].update({'hello': 'a'})
# np.save('my_file.npy', dictionary)

# # Load
# read_dictionary = np.load('train.npy').item()
# print(read_dictionary)


# a = torch.randn(128, 2048, 16, 8)
# # a = torch.randn(1,1,16,8)
# print(a.shape)
# # m = nn.AdaptiveAvgPool2d((256, 128))
# m = nn.functional.interpolate(a, scale_factor=16, mode='nearest')
# print(m.shape)
# path = ["", ""]
# feature = cal_feature(input, 1, 2, path)


# json_paths = glob.glob(osp.join(data_path, 'kp', "query", '*.json'))
# num = [0 for _ in range(25)]
# for p in json_paths:
#     with open(p, 'rb') as f:
#         params = json.load(f)
#         for i in range(25):
#             if params[3 * i + 2] == 0:
#                 num[i] += 1
# print(num)

# json_paths = glob.glob(osp.join(data_path,  "train", '*.json'))
# for p in json_paths:
#     os.remove(p)

# json_paths = glob.glob(osp.join(data_path, 'kp', "query", '*.json'))
# num = 0
# for p in json_paths:
#     with open(p, 'rb') as f:
#         params = json.load(f)
#         if len(params['people'])>0:
#             pose = params['people'][0]['pose_keypoints_2d']
#             if pose[3 * 0 + 2] == 0 and pose[3 * 15 + 2] == 0 and pose[3 * 16 + 2] == 0 and pose[3 * 17 + 2] == 0 and pose[3 * 18 + 2] == 0:
#                 print(p)
#                 num += 1
# print(num)
