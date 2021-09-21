# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import os.path as osp
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import scipy.misc
import torch


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class ImageDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, _, camid = self.dataset[index]
        img = read_image(img_path)

        if self.transform is not None:
            img = self.transform(img)

        return img, pid, camid, img_path


class ImageDatasetMMT(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, _, camid = self.dataset[index]
        img = read_image(img_path)
        img2 = read_image(img_path.replace('train', 'train2D'))

        if self.transform is not None:
            img = self.transform(img)
            img2 = self.transform(img2)

        return img, img2, pid, camid, img_path


class ImageDatasetPart(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, dataset, cfg=None, transform=None):
        self.dataset = dataset
        self.transform = transform
        self.cfg = cfg

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, clothid, camid = self.dataset[index]
        img = read_image(img_path)
        if img_path.split("/")[-2] == 'train':
            path = "train3D/"
            # path = "train3S/"
        elif img_path.split("/")[-2] == 'C':
            path = "query3D/"
            # path = "query3S/"
        elif img_path.split("/")[-2] == 'A':
            path = 'gallery3D/'
            # path = 'gallery3S/'
        # pointcloud = torch.from_numpy(np.load("/home/yhl/data/prcc/rgb/" + path +
        #                      img_path.split("/")[-1][:-4] + '.npy'))
        pointcloud = "0"
        if self.transform is not None:
            img = self.transform(img)

        # if msk_path != "":
        #     msk = np.load(msk_path)  # [256, 128, 14], min=6.8e-7, max=0.99
        #     msk = torch.from_numpy(msk).permute(2, 0, 1).unsqueeze(dim=0)  # [1, 14, 256, 128]
        #     msk = torch.nn.functional.interpolate(msk, size=(256, 128), mode='bilinear', align_corners=True)
        # else:
        #     msk = torch.empty([0])
        return img, pid, clothid, camid, pointcloud

