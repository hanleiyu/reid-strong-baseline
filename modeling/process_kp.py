# encoding: utf-8

import glob
import json
from PIL import Image
import os.path as osp
import os
import math
import numpy as np
import torch
import cv2
from torchvision import transforms


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
        json_path = osp.join(data_path, 'kp', name, img_path.split("/")[-1][:-4] + '_keypoints.json')
        with open(json_path, 'rb') as f:
            write_json_data(json_path, resize_json_data(json_path, h, w))


def remove(name):
    json_paths = glob.glob(osp.join(data_path, 'kpo', name, '*.json'))
    kp = [1, 2, 5, 8]
    num = 0
    for p in json_paths:
        flag = 1
        with open(p, 'rb') as f:
            params = json.load(f)
            if len(params['people']) > 0:
                params = params['people'][0]['pose_keypoints_2d']
                if (params[3 * 10 + 2] == 0 or params[3 * 11 + 2] == 0) and (
                        params[3 * 13 + 2] == 0 or params[3 * 14 + 2] == 0):
                    flag = 0
                elif (params[3 * 2 + 2] == 0 or params[3 * 3 + 2] == 0) and (
                        params[3 * 5 + 2] == 0 or params[3 * 6 + 2] == 0):
                    flag = 0
                elif (params[3 * 3 + 2] == 0 or params[3 * 4 + 2] == 0) and (
                        params[3 * 7 + 2] == 0 or params[3 * 6 + 2] == 0):
                    flag = 0
                elif params[3 * 10 + 2] == 0  and params[3 * 13 + 2] == 0:
                    flag = 0
                else:
                    for i in range(len(kp)):
                        if params[3 * kp[i] + 2] == 0:
                            flag = 0
                            break
            else:
                flag = 0

        f.close()
        if flag:
            num += 1
        else:
            print(p)
            os.remove(p)
            # os.remove(osp.join(data_path, name, p.split("/")[-1][:-15] + '.jpg'))
    print(num)


def get_json_data(path, n1, n2):
    t1 = []
    t2 = []
    c = 0
    with open(path, 'rb') as f:
        params = json.load(f)
        params = params['people'][0]['pose_keypoints_2d']
        if len(params) > 0:
            t1 = [params[3 * n1], params[3 * n1 + 1]]
            t2 = [params[3 * n2], params[3 * n2 + 1]]
            c = (params[3 * n1 + 2] + params[3 * n2 + 2])/2
    f.close()

    return t1, t2, c


def get_part_data(path, name):
    t1 = []
    t2 = []
    t3 = []
    t4 = []
    c = 0
    c1 = 0
    c2 = 0
    if name == "leg":
        n1 = 10
        n2 = 13
    elif name == "arm":
        n1 = 2
        n2 = 5
    elif name == "hand":
        n1 = 3
        n2 = 6

    with open(path, 'rb') as f:
        params = json.load(f)
        # params = params['people'][0]['pose_keypoints_2d']
        if len(params) > 0:
            if params[3 * n1 + 2] != 0 and params[3 * (n1 + 1) + 2] != 0:
                t1 = [params[3 * n1], params[3 * n1 + 1]]
                t2 = [params[3 * (n1 + 1)], params[3 * (n1 + 1) + 1]]
                c1 = (params[3 * n1 + 2] + params[3 * (n1 + 1) + 2]) / 2
            if params[3 * n2 + 2] != 0 and params[3 * (n2 + 1) + 2] != 0:
                t3 = [params[3 * n2], params[3 * n2 + 1]]
                t4 = [params[3 * (n2 + 1)], params[3 * (n2 + 1) + 1]]
                c2 = (params[3 * n2 + 2] + params[3 * (n2 + 1) + 2]) / 2

    f.close()

    if c1 != 0 and c2 != 0:
        c = (c1 + c2) / 2
    elif c1 != 0 and c2 == 0:
        c = c1
    elif c1 == 0 and c2 != 0:
        c = c2

    return t1, t2, t3, t4, c


def get_thigh_data(path):
    t1 = []
    t2 = []
    t3 = []
    t4 = []
    c = 0
    c1 = 0
    c2 = 0
    with open(path, 'rb') as f:
        params = json.load(f)
        # params = params['people'][0]['pose_keypoints_2d']
        if len(params) > 0:
            if params[3 * 8 + 2] != 0 and params[3 * 10 + 2] != 0:
                t1 = [params[3 * 8], params[3 * 8 + 1]]
                t2 = [params[3 * 10], params[3 * 10 + 1]]
                c1 = (params[3 * 8 + 2] + params[3 * 10 + 2]) / 2
            if params[3 * 8 + 2] != 0 and params[3 * 13 + 2] != 0:
                t3 = [params[3 * 8], params[3 * 8 + 1]]
                t4 = [params[3 * 13], params[3 * 13 + 1]]
                c2 = (params[3 * 8 + 2] + params[3 * 13 + 2]) / 2
    f.close()

    if c1 != 0 and c2 != 0:
        c = (params[3 * 8 + 2] + params[3 * 10 + 2] + params[3 * 13 + 2]) / 3
    elif c1 != 0 and c2 == 0:
        c = c1
    elif c1 == 0 and c2 != 0:
        c = c2

    return t1, t2, t3, t4, c


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

        if steps == 0:
            return [pt1, pt2]
        else:
            xinc = dx / steps
            yinc = dy / steps
            x = x1
            y = y1

        for k in range(int(steps) + 1):
                line.append([math.floor(x + 0.5), math.floor(y + 0.5)])
                x += xinc
                y += yinc

    return line


def cal_mask(p, h, w, a, b=0):
    pt3 = []
    pt4 = []
    m = torch.zeros(size=(h, w))
    if a != "face" and a != "body" and a != "people":
        if a == "leg":
            pt1, pt2, pt3, pt4, c = get_part_data(p, "leg")
            # weight = 10
            # weight = 15
        elif a == "thigh":
            pt1, pt2, pt3, pt4, c = get_thigh_data(p)
        elif a == "arm":
            pt1, pt2, pt3, pt4, c = get_part_data(p, "arm")
            # weight = 8
            # weight = 10
            # weight = 12
        elif a == "hand":
            pt1, pt2, pt3, pt4, c = get_part_data(p, "hand")
            # weight = 8
            # weight = 10
            # weight = 12
        else:
            pt1, pt2, c = get_json_data(p, a, b)

        line = dda_line_points(pt1, pt2) + dda_line_points(pt3, pt4)
        # print(p)
        for i in range(len(line)):
            if line[i][0] <= w-1 and line[i][1] <= h-1:
                m[line[i][1]][line[i][0]] = 1
                # for j in range(1, weight):
                j = 1
                if line[i][0] <= w-1-j:
                    m[line[i][1]][line[i][0] + j] = 1
                # if line[i][1] <= h-1-j:
                #     m[line[i][1] + j][line[i][0]] = 1
                if line[i][0] >= j:
                    m[line[i][1]][line[i][0] - j] = 1
                    # if line[i][1] >= j:
                    #     m[line[i][1] - j][line[i][0]] = 1
            # else:
            #     print(p)
    elif a == "face":
        with open(p, 'rb') as f:
            params = json.load(f)
            # params = params['people'][0]['pose_keypoints_2d']
            if len(params) > 0:
                t = math.floor(params[3*1 + 1])
                c = params[3*1+2]
                p1 = math.floor(params[3 * 2])
                p2 = math.floor(params[3 * 5])
                # print(p)
            for i in range(t+1):
                # m[:][i] = 1
                if i<h:
                    for j in range(min(p1, p2), max(p1, p2)):
                        if j < w:
                            m[i][j] = 1
    # elif a == "body" or a == "people":
    #     pt1, pt2, pt3, pt4, c1 = get_part_data(p, "leg")
    #     line = dda_line_points(pt1, pt2) + dda_line_points(pt3, pt4)
    #     pt1, pt2, pt3, pt4, c2 = get_thigh_data(p)
    #     line += dda_line_points(pt1, pt2) + dda_line_points(pt3, pt4)
    #     pt1, pt2, pt3, pt4, c3 = get_part_data(p, "arm")
    #     line += dda_line_points(pt1, pt2) + dda_line_points(pt3, pt4)
    #     pt1, pt2, pt3, pt4, c4 = get_part_data(p, "hand")
    #     line += dda_line_points(pt1, pt2) + dda_line_points(pt3, pt4)
    #     pt1, pt2, c5 = get_json_data(p, 1, 8)
    #     pt3, pt4, c6 = get_json_data(p, 2, 5)
    #     line += dda_line_points(pt1, pt2) + dda_line_points(pt3, pt4)
    #     c = (c1 + c2 + c3 + c4 + c5 + c6)/6
    #     for i in range(len(line)):
    #         if line[i][0] < w and line[i][1] < h:
    #             m[line[i][1]][line[i][0]] = 1
    #     if a == "people":
    #         with open(p, 'rb') as f:
    #             params = json.load(f)
    #             if len(params) > 0:
    #                 t = params[3 * 1]
    #                 c = params[3 * 1 + 2]
    #             for i in range(t + 1):
    #                 m[:][i] = 1

    # new_img_PIL = transforms.ToPILImage()(np.array(m))
    # new_img_PIL.show()
    mask = torch.unsqueeze(m, 0)
    return mask, c


def cal_kp(path):
    dictionary = {}
    json_paths = glob.glob(osp.join(data_path, 'kp', path, '*.json'))
    # json_paths = glob.glob("/home/yhl/data/prcc/rgb/kpo/train/121_A_cropped_rgb052_keypoints.json")
    for p in json_paths:
        img = p.split("/")[-1][:-15]
        # imgs = Image.open(osp.join(data_path, path, img + '.jpg'))
        # m1, c1 = cal_mask(p, 1, 8)
        # m2, c2 = cal_mask(p, 2, 5)
        # m3, c3 = cal_mask(p, 8, 10)
        # m4, c4 = cal_mask(p, 8, 13)
        # m5, c5 = cal_mask(p, imgs.size[1], imgs.size[0], "leg")
        m5, c5 = cal_mask(p, 16, 8, "leg")
        # m6, c6 = cal_mask(p, "thigh")
        # m7, c7 = cal_mask(p, imgs.size[1], imgs.size[0], "arm")
        m7, c7 = cal_mask(p, 16, 8, "arm")
        # m8, c8 = cal_mask(p, imgs.size[1], imgs.size[0], "hand")
        m8, c8 = cal_mask(p, 16, 8, "hand")
        # m9, c9 = cal_mask(p, imgs.size[1], imgs.size[0], "face")
        m9, c9 = cal_mask(p, 16, 8, "face")
        # m10, c10 = cal_mask(p, "body")
        # m11, c10 = cal_mask(p, "people")
        mask = torch.stack((m5, m7, m8, m9), 0)
        # c = [c1, c2, c5, c6, c7, m8]
        # mask = torch.unsqueeze(m11, 0)
        mask = mask.numpy()
        dictionary.update({img: mask})
        # dictionary.update({img: {"mask":mask, "confidence":c}})
    return dictionary


def save_kp():
    # maskt = cal_kp("train")
    maskg = cal_kp("gallery")
    maskq = cal_kp("queryc")
    # torch.save(maskt, osp.join(data_path, 'part4n/maskt.pt'))
    torch.save(maskg, osp.join(data_path, 'part4n/maskg.pt'))
    torch.save(maskq, osp.join(data_path, 'part4n/maskq.pt'))


data_path = "/home/yhl/data/prcc/rgb"
save_kp()



# resize_kp(128, 256, "train")
# resize_kp(8, 16, "val")
# resize_kp(8, 16, "test")
# os.remove("/home/yhl/data/prcc/rgb/kp/train/170_B_cropped_rgb561_keypoints.json")
# os.remove("/home/yhl/data/VC/train/0338-04-03-08.jpg")

# remove("train")
# remove("gallery")
# remove("queryc")

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


# a = torch.randn(128, 2048, 16, 8)
# # a = torch.randn(1,1,16,8)
# print(a.shape)
# # m = nn.AdaptiveAvgPool2d((256, 128))
# m = nn.functional.interpolate(a, scale_factor=16, mode='nearest')
# print(m.shape)
# path = ["", ""]
# feature = cal_feature(input, 1, 2, path)


# json_paths = glob.glob(osp.join(data_path, 'kp', "train", '*.json'))
# num = 0
# # num = [0 for _ in range(25)]
# for p in json_paths:
#     with open(p, 'rb') as f:
#         params = json.load(f)
#         if len(params)>0:
#             # pose = params['people'][0]['pose_keypoints_2d']
#         #     for i in range(25):
#         #         if pose[3 * i + 2] == 0:
#         #             num[i] += 1
#         # else:
#         #     print(p)
#             if (params[3 * 10 + 2] == 0 or params[3 * 11 + 2] == 0)
#             and (params[3 * 13 + 2] == 0 or params[3 * 14 + 2] == 0):
#                 print(p)
#                 os.remove(p)
#                 num += 1
# print(num)

def crop(path):
    img_paths = glob.glob(os.path.join(data_path, path, "*.jpg"))
    for img in img_paths:
        image = cv2.imread(img)
        m = np.zeros((image.shape[0], image.shape[1]))

        p = os.path.join(data_path, "kpo/train", img.split("/")[-1][:-4] + '_keypoints.json')
        pt1, pt2, _ = get_json_data(p, 1, 8)
        pt3, pt4, _ = get_json_data(p, 2, 5)
        for j in range(image.shape[0]):
            for i in range(image.shape[1]):
                if min(pt3[0], pt4[0]) < i < max(pt3[0], pt4[0]) and pt1[1] < j < pt2[1]:
                    m[j][i] = 1
        image[m > 0] = 0
        cv2.imwrite(os.path.join(data_path, path+"crop2", img.split("/")[-1]), image)


# crop("queryc")
# crop("gallery")
# crop("train")

def cropnew(path):
    img_paths = glob.glob(os.path.join(data_path, path, "*.jpg"))
    for img in img_paths:
        image = cv2.imread(img)
        m = np.zeros((image.shape[0], image.shape[1]))

        p = os.path.join(data_path, "kpo/train", img.split("/")[-1][:-4] + '_keypoints.json')
        pt1, pt2, _ = get_json_data(p, 1, 8)
        pt3, pt4, _ = get_json_data(p, 2, 5)
        upmin = min(pt4[0], pt3[0])
        upmax = max(pt4[0], pt3[0])
        pt5, pt6, _, pt7, _ = get_thigh_data(p)
        if len(pt5) > 0 and len(pt6) > 0 and len(pt7) > 0:
            downleft = min(pt5[0], pt6[0], pt7[0])
            downright = max(pt5[0], pt6[0], pt7[0])
            downmax = max(pt6[1], pt7[1])
        elif len(pt5) == 0 or (len(pt6) == 0 and len(pt7) == 0):
            downleft = 0
            downright = 0
            downmax = 0
        elif len(pt6) > 0:
            downleft = min(pt5[0], pt6[0])
            downright = max(pt5[0], pt6[0])
            downmax = pt6[1]
        elif len(pt7) > 0:
            downleft = min(pt5[0], pt7[0])
            downright = max(pt5[0], pt7[0])
            downmax = pt7[1]


        for j in range(image.shape[0]):
            for i in range(image.shape[1]):
                if upmin < i < upmax and pt1[1] < j < pt2[1]:
                    m[j][i] = 1
                if downleft < i < downright and pt5[1] < j < downmax:
                    m[j][i] = 1

        pt1, pt2, pt3, pt4, c1 = get_part_data(p, "leg")
        line = dda_line_points(pt1, pt2) + dda_line_points(pt3, pt4)
        pt1, pt2, pt3, pt4, c4 = get_part_data(p, "hand")
        line += dda_line_points(pt1, pt2) + dda_line_points(pt3, pt4)
        pt1, pt2, pt3, pt4, c3 = get_part_data(p, "arm")
        line += dda_line_points(pt1, pt2) + dda_line_points(pt3, pt4)
        weight = min((downright - downleft), (upmax - upmin)) / 2
        for i in range(len(line)):
            for j in range(int(weight)):
                if line[i][0] + j in range(image.shape[1]) and line[i][1] in range(image.shape[0]):
                    m[line[i][1]][line[i][0] + j] = 0
                if line[i][0] - j in range(image.shape[1]) and line[i][1] in range(image.shape[0]):
                    m[line[i][1]][line[i][0] - j] = 0

        image[m > 0] = 0
        cv2.imwrite(os.path.join(data_path, path+"crop3", img.split("/")[-1]), image)

# cropnew("queryc")
# cropnew("gallery")
# cropnew("train")


# imgs = Image.open("/home/yhl/data/prcc/rgb/train/113_C_cropped_rgb253.jpg")
# p = "/home/yhl/data/prcc/rgb/kpo/train/113_C_cropped_rgb253_keypoints.json"
# m9, c9 = cal_mask(p, imgs.size[1], imgs.size[0], "arm")

# img_paths = glob.glob(os.path.join("/home/yhl/data/prcc/rgb/train/", "*.jpg"))
# h = 0
# w = 0
# n = 0
# for img in img_paths:
#     pic = Image.open(img)
#     h += pic.size[1]
#     w += pic.size[0]
#     n += 1
# print(h/n, w/n)