# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import os.path as osp

import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
from ..transforms.transforms import RandomCrop, RandomHorizontalFlip, RandomErasing
import numpy as np

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
        img_path, pid, camid = self.dataset[index]
        img = read_image(img_path)

        if self.transform is not None:
            img = self.transform(img)

        return img, pid, camid, img_path


class ImageDatasetPart(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, dataset, cfg=None, transform=None):
        self.dataset = dataset
        self.transform = transform
        self.cfg = cfg

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid, mask = self.dataset[index]
        # img_path, pid, camid = self.dataset[index]
        img = read_image(img_path)

        if self.transform is not None:
            img = self.transform(img)
        else:
            masks = []
            num = len(mask[:, 0, 0, 0])
            for i in range(num):
                masks.append(Image.fromarray(np.uint8(mask[i, 0, :, :] * 255)))

            img = T.Resize(self.cfg.INPUT.SIZE_TRAIN)(img)
            for i in range(num):
                masks[i] = T.Resize(self.cfg.INPUT.SIZE_TRAIN)(masks[i])

            img, masks = RandomHorizontalFlip(p=self.cfg.INPUT.PROB)(img, masks)

            img = T.Pad(self.cfg.INPUT.PADDING)(img)
            for i in range(num):
                masks[i] = T.Pad(self.cfg.INPUT.PADDING)(masks[i])

            img, masks = RandomCrop(self.cfg.INPUT.SIZE_TRAIN)(img, masks)

            img = T.ToTensor()(img)

            for i in range(num):
                masks[i] = T.Resize([16, 8])(masks[i])

            for i in range(num):
                # masks[i] = torch.from_numpy(masks[i].transpose((2, 0, 1)))
                masks[i] = T.ToTensor()(masks[i])
                masks[i] = torch.where(masks[i] > 0, torch.ones(1), masks[i])

            mask = torch.stack((masks[0], masks[1], masks[2], masks[3]), 0)

            img = T.Normalize(mean=self.cfg.INPUT.PIXEL_MEAN, std=self.cfg.INPUT.PIXEL_STD)(img)
            img = RandomErasing(probability=self.cfg.INPUT.RE_PROB, mean=self.cfg.INPUT.PIXEL_MEAN)(img)
        # return img, pid, camid, img_path, mask
        return img, pid, camid, img_path

