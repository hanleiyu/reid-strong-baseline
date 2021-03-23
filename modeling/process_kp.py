# encoding: utf-8

import glob
import json
from PIL import Image
import os.path as osp
import os
import math
import numpy as np
import torch


def get_path(p):
    path = ""
    if p.find('train') != -1:
        path = osp.join("/home/yhl/data/VC/kp/train", p[-17:-4] + '_keypoints.json')
    elif p.find('gallery') != -1:
        path = osp.join("/home/yhl/data/VC/kp/gallery", p[-17:-4] + '_keypoints.json')
    elif p.find('query') != -1:
        path = osp.join("/home/yhl/data/VC/kp/query", p[-17:-4] + '_keypoints.json')
    return path


def get_json_data(path, n1, n2):
    t1 = []
    t2 = []
    with open(path, 'rb') as f:
        params = json.load(f)
        if len(params) > 0:
            t1 = [params[3 * n1 + 1], params[3 * n1]]
            t2 = [params[3 * n2 + 1], params[3 * n2]]
    f.close()

    return t1, t2


def get_leg_data(path):
    t1 = []
    t2 = []
    with open(path, 'rb') as f:
        params = json.load(f)
        if len(params) > 0:
            num = 10
            if params[3 * 10 + 2] <= 0.25 or params[3 * 11 + 2] <= 0.25 \
                    or (params[3 * 13 + 2] + params[3 * 14 + 2]) > (params[3 * 10 + 2] + params[3 * 11 + 2]):
                num = 13
            t1 = [params[3 * num + 1], params[3 * num]]
            t2 = [params[3 * (num + 1) + 1], params[3 * (num + 1)]]
    f.close()

    return t1, t2


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
            if math.floor(x + 0.5) <= 7 and math.floor(y + 0.5) <= 15:
                line.append([math.floor(x + 0.5), math.floor(y + 0.5)])
                x += xinc
                y += yinc

    return line


def cal_feature(input, index, path, kp):
    outputs = torch.empty(size=(128, 2048))
    for k in range(128):
        line = kp[path[k]][index]
        output = torch.empty(size=(len(line), 2048))
        if len(line) > 0:
            for i in range(len(line)):
                if line[i][0] > 7 or line[i][1]> 15:
                    print(line[i])
                else:
                    output[i] = input[k, :, line[i][0], line[i][1]]
        output = output.mean(0, False)
        outputs[k] = output

    return outputs.cuda()


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


def cal_kp(name):
    dictionary = {}
    json_paths = glob.glob(osp.join(data_path, 'kp', name, '*.json'))
    for p in json_paths:
        img_path = osp.join(data_path, name, p[-28:-15] + '.jpg')
        pt1, pt2 = get_json_data(p, 1, 8)
        line = dda_line_points(pt1, pt2)
        dictionary.update({img_path: {"upper body": line}})
        pt1, pt2 = get_json_data(p, 2, 5)
        line = dda_line_points(pt1, pt2)
        dictionary[img_path].update({"shoulder": line})
        pt1, pt2 = get_json_data(p, 8, 10)
        line = dda_line_points(pt1, pt2)
        dictionary[img_path].update({"left thigh": line})
        pt1, pt2 = get_json_data(p, 8, 13)
        line = dda_line_points(pt1, pt2)
        dictionary[img_path].update({"right thigh": line})
        pt1, pt2 = get_leg_data(p)
        line = dda_line_points(pt1, pt2)
        dictionary[img_path].update({"lower leg": line})
    return dictionary


def save_kp():
    train_dic = cal_kp("train")
    gallery_dic = cal_kp("gallery")
    query_dic = cal_kp("query")
    gallery_dic.update(query_dic)
    np.save('train.npy', train_dic)
    np.save('val.npy', gallery_dic)


data_path = "/home/yhl/data/VC/"
# save_kp()

# resize_kp(8, 16, "train")
# resize_kp(8, 16, "gallery")
# resize_kp(8, 16, "query")
# os.remove("/home/yhl/data/VC/kp/train/0338-04-03-08_keypoints.json")
# os.remove("/home/yhl/data/VC/train/0338-04-03-08.jpg")
#
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

# img = Image.open("/home/yhl/data/VC/train/0286-01-02-08.jpg")
# print(img.size)

# # Save
# path = ['1', '2', '3', '4']
# dictionary = {}
# a = {path[0]: {'path': 'a', 'size': '10'}, path[1]: {'path': 'b', 'size': '20'}}
# b = {path[2]: {'path': 'a', 'size': '10'}, path[3]: {'path': 'b', 'size': '20'}}
# dictionary.update(a)
# dictionary.update(b)
# dictionary['1'].update({'hello': 'a'})
# np.save('my_file.npy', dictionary)
#
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


a = torch.randn(2, 2)
d = a
for i in range(3):
    d = torch.stack((d, a), 0)
# d.mean(1, False)

