from PIL import Image
import torchvision.transforms as T

path = "/home/yhl/data/prcc/rgb/train/111_A_cropped_rgb028.jpg"
img = Image.open(path).convert('RGB')
img = T.Resize([256, 128])(img)
img = T.RandomHorizontalFlip(p=1)(img)
img = T.Pad(10)(img)
imgsave = img.save("/home/yhl/data/prcc/rgb/111_A_cropped_rgb028.jpg")
img = T.RandomCrop([256, 128])(img)
img = img.save("/home/yhl/data/prcc/rgb/crop.jpg")
