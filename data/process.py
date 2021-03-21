# encoding: utf-8

import glob
import re
from PIL import Image

import os.path as osp

img_paths = glob.glob(osp.join("/home/yhl/data/VC/train", '*.jpg'))
for img_path in img_paths:
    img = Image.open(img_path)
    img = img.resize((256, 128), Image.ANTIALIAS)
    img.save("/home/yhl/data/VC/train_resize")
