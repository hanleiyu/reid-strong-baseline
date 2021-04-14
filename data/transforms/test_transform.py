from PIL import Image
import torchvision.transforms as T
import torch
import numpy as np
import torchvision.transforms as T
from data.transforms.transforms import RandomCrop, RandomHorizontalFlip, RandomErasing

path = "/home/yhl/data/prcc/rgb/train/121_A_cropped_rgb052.jpg"
img = Image.open(path).convert('RGB')
kps = torch.load('/home/yhl/data/prcc/rgb/part4n/masktest1.pt')
mask = kps[path.split("/")[-1][:-4]]


masks = []
num = len(mask[:, 0, 0, 0])
a = np.uint8(mask[0, 0, :, :] * 255)
# print(np.unique(a))
a = Image.fromarray(a)
a.save("/home/yhl/data/prcc/rgb/0.png")
a = np.uint8(mask[1, 0, :, :] * 255)
a = Image.fromarray(a)
a.save("/home/yhl/data/prcc/rgb/1.png")
a = np.uint8(mask[2, 0, :, :] * 255)
a = Image.fromarray(a)
a.save("/home/yhl/data/prcc/rgb/2.png")
a = np.uint8(mask[3, 0, :, :] * 255)
a = Image.fromarray(a)
a.save("/home/yhl/data/prcc/rgb/3.png")
# a = np.array(a)
# print(np.unique(a))
# print("hello")

# for i in range(num):
#     masks.append(Image.fromarray(np.uint8(mask[i, 0, :, :] * 255)))
#
# img = T.Resize([256, 128])(img)
# for i in range(num):
#     masks[i] = T.Resize([256, 128])(masks[i])
# # masks[0] = np.array(masks[0])
# img, masks = RandomHorizontalFlip(0.5)(img, masks)
#
# img = T.Pad(10)(img)
# for i in range(num):
#     masks[i] = T.Pad(10)(masks[i])
#
# img, masks = RandomCrop([256, 128])(img, masks)
#
# img = T.ToTensor()(img)
#
# for i in range(num):
#     masks[i] = T.Resize([16, 8])(masks[i])
#
# a = masks[0]
# a.save("/home/yhl/data/prcc/rgb/0n.png")
# a = masks[1]
# a.save("/home/yhl/data/prcc/rgb/1n.png")
# a = masks[2]
# a.save("/home/yhl/data/prcc/rgb/2n.png")
# a = masks[3]
# a.save("/home/yhl/data/prcc/rgb/3n.png")
#
# for i in range(num):
#     # masks[i] = torch.from_numpy(masks[i].transpose((2, 0, 1)))
#     masks[i] = T.ToTensor()(masks[i])
#     # masks[i] = torch.where(masks[i] > 0, torch.ones(1), masks[i])
#
# mask = torch.stack((masks[0], masks[1], masks[2], masks[3]), 0)
#
# img = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
# img = RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])(img)

# mask = torch.from_numpy(mask[0, 0, :, :])
# mask = torch.unsqueeze(mask, 2)
# mask = mask.byte()
# mask = np.expand_dims(mask[0, 0, :, :], 2)
# mask = np.repeat(mask, 3, axis=2)
# array = np.array(img)
# array = torch.from_numpy(array)
# array = torch.mul(array, mask.byte())
# # array = img * mask
# array = array.numpy()
# array = np.transpose(array, [1, 2, 0])
# img = Image.fromarray(array, mode='RGB')
# img = img.save("/home/yhl/data/prcc/rgb/mask.jpg")

# img = T.Resize([256, 128])(img)
# img = T.RandomHorizontalFlip(p=1)(img)
# img = T.Pad(10)(img)
# imgsave = img.save("/home/yhl/data/prcc/rgb/111_A_cropped_rgb028.jpg")
# img = T.RandomCrop([256, 128])(img)
# img = img.save("/home/yhl/data/prcc/rgb/crop.jpg")
