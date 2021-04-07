import glob
import os
import shutil
import cv2

dataset_dir = '/home/yhl/data/prcc/rgb'
dataset_dirn = '/home/yhl/data/prcc/a'


def process_prcc(path):
    if path == "train" or path == "val":
        img_paths = glob.glob(os.path.join(dataset_dir, path, "**/*.jpg"))
        for img in img_paths:
            new_name = img.split("/")[-2] + '_' + img.split("/")[-1]
            os.rename(img, new_name)
            shutil.move(new_name, os.path.join(dataset_dirn, path))
    else:
        img_paths = glob.glob(os.path.join(dataset_dir, path, "*/**/*.jpg"))
        for img in img_paths:
            camid = img.split("/")[-3]
            new_name = img.split("/")[-2] + '_' + camid + '_' + img.split("/")[-1]
            os.rename(img, new_name)
            shutil.move(new_name, os.path.join(dataset_dirn, path))

# process_prcc("train")
# process_prcc("val")
# process_prcc("test")


def grey(path):
    img_paths = glob.glob(os.path.join(dataset_dir, path, "*.jpg"))
    for img in img_paths:
        img_grey = cv2.imread(img)
        img_grey = cv2.cvtColor(img_grey, cv2.COLOR_RGB2GRAY)
        cv2.imwrite(img, img_grey)


# grey("traingrey")
# grey("gallerygrey")
# grey("querycgrey")

