# encoding: utf-8
import torch
import math
import numpy
import json
import os.path as osp
import torch.nn as nn


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


def cal_feature(input, n1, n2, path):
    a = input.data.cpu().numpy()
    outputs = []

    for k in range(0, 128):
        output = []
        pt1, pt2 = get_json_data(get_path(path[k]), n1, n2)
        line = dda_line_points(pt1, pt2)
        if len(line) > 0:
            for j in range(0, 2048):
                b = a[k][j]
                s = 0
                for i in range(len(line)):
                    s += b[line[i][0], line[i][1]]
                s /= len(line)
                output.append(s)
        outputs.append(output)

    outputs = torch.from_numpy(numpy.array(outputs)).float().cuda()
    # print(outputs.shape)

    return outputs


# a = torch.randn(128, 2048, 16, 8)
# # a = torch.randn(1,1,16,8)
# print(a.shape)
# # m = nn.AdaptiveAvgPool2d((256, 128))
# m = nn.functional.interpolate(a, scale_factor=16, mode='nearest')
# print(m.shape)
# path = ["", ""]
# feature = cal_feature(input, 1, 2, path)

# a = [('a',22,1), ('b',21,2), ('c')]
# print(a[0][0])
# print(a[0][1])


