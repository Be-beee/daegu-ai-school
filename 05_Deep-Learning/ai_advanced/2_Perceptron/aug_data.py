import os
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torchvision.utils import save_image
from PIL import Image

import cv2
import numpy as np

# image type
image_foramt = [".bmp", ".jpg", ".jpeg", ".png", "tif", "tiff", "dom"]


# transforms show function

def transformsShow(name="img", ):
    def transforms_show(img):
        cv2.imshow(name, np.array(img))
        if cv2.waitkey(0) & 0xff == ord('q'):
            exit()
        return img

    return transforms_show


# 폴더에서 파일 가져오기
def file_serach(folder_path):
    img_root = []

    for (path, dir, file) in os.walk(folder_path):
        #         print(path, dir, file)
        for file_name in file:
            # cat.jpeg -> .jpeg
            ext = os.path.splitext(file_name)[-1].lower()
            if ext in image_foramt:
                root = os.path.join(path, file_name)
                img_root.append(root)

    return img_root


img_path = "./data"
data_path = file_serach(img_path)



# custom dataset
class CustomDataset(Dataset) :

    def __init__(self, path, transforms=None):
        self.path = path
        self.transforms = transforms
        print(transforms)

    def __getitem__(self, item):
        path = self.path[item]
        img = cv2.imread(path)

        if self.transforms is not None:
            img = self.transforms(img)

        return img

    def __len__(self):
        return len(self.path)


# augmentation

my_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ColorJitter(brightness=0.5),
    transforms.Resize((255, 255)),
    transforms.CenterCrop((224, 224)),
    transforms.RandomHorizontalFlip(p=0.8),
    transforms.RandomGrayscale(p=0.4),
    transforms.RandomRotation(degrees=0.5),
    transforms.ToTensor()

])

dataset = CustomDataset(data_path, transforms=my_transforms)



img_num = 1
for _ in range(5) :
    for img in dataset:
        # 현재 디렉토리 위치에 저장 됩니다.
        save_image(img, "img"+str(img_num) + ".png")
        img_num += 1