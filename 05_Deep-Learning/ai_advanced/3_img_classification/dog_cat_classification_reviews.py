# 구글 드라이브 마운트
# from google.colab import drive
# drive.mount("/content/drive")

# 라이브러리 추가
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import glob
import os

from torchvision import datasets, models, transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader

# 1. device 설정 GPU 사용 가능 여부에 따라 device 정보 저장
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# 2. data location
data_dir = "/content/drive/MyDrive/classification/cat_dog_data"

# 3. hyper parameter
batch_size = 32
num_epochs = 10
learning_rate = 0.001

# 4. custom data set
class catdogDataset(Dataset):
  def __init__(self, data_dir, mode, transform=None):
    # /content/drive/MyDrive/classification/cat_dog_data/train/*/*
    self.all_data = sorted(glob.glob(os.path.join(data_dir, mode, "*", "*")))
    self.transform = transform


  def __getitem__(self, item):
    # 데이터를 가져오는 부분
    data_path = self.all_data[item]

    # 데이터 경로 -> image
    img = Image.open(data_path) # 읽어올 땐 BGR mode
    img = img.convert("RGB") # Image RGB mode


    # img, label
    # 2. label 처리
    # train -> folder name 라벨 네임
    label_name = os.path.basename(data_path)
    # label_name -> /cat/
    label_str = str(label_name)

    if label_str.startswith("cat") == True:
      label = 0
    else:
      label = 1

    # transform=None is not None
    if self.transform is not None:
      img = self.transform(img)

    return img, label



  def __len__(self):
    length = len(self.all_data)
    return length

# data transforms
# image size 224 224 -> vgg 논문 기준

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

data_transforms = {
    'train': transforms.Compose([
          transforms.RandomRotation(5),
          transforms.RandomHorizontalFlip(p=0.5),
          transforms.RandomVerticalFlip(p=0.5),
          transforms.Resize([224, 224]),
          transforms.ToTensor(),
          transforms.Normalize(mean, std)
    ]),
    'val': transforms.Compose([
          transforms.Resize([224, 224]),
          transforms.ToTensor(),
          transforms.Normalize(mean, std)
    ]),
    'test': transforms.Compose([
          transforms.Resize([224, 224]),
          transforms.ToTensor(),
          transforms.Normalize(mean, std)
    ])

}

# data 정의 data loader
# data 정의 -> set
train_data_set = catdogDataset(data_dir, mode="train", transform=data_transforms["train"])
val_data_set = catdogDataset(data_dir, mode="val", transform=data_transforms["val"])
test_data_set = catdogDataset(data_dir, mode="test", transform=data_transforms["test"])


# data loader
train_data_loader = DataLoader(train_data_set, batch_size=batch_size, shuffle=True, drop_last=True)
val_data_loader = DataLoader(val_data_set, batch_size=batch_size, shuffle=False, drop_last=True)
test_data_loader = DataLoader(test_data_set, batch_size=batch_size, shuffle=False, drop_last=True)

# vgg
net = models.vgg11(pretrained=True).to(device)
net.classifier[6] = nn.Linear(4096, 2)

model = net.to(device)
print(model)

# loss function, optimizer scheduler
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

def train(num_epochs, model, data_loader, criterion, optimizer, save_dir, val_every, device):

  print("Start Training ...!!!")

  best_loss = 9999

  for epoch in range(num_epochs):
    for i, (imgs, labels) in enumerate(data_loader):
      imges, labels = imgs.to(device), labels.to(device)

      outputs = model(imges)

      loss = criterion(outputs, labels)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      _, argmax = torch.max(outputs, 1)
      accuracy = (labels == argmax).float().mean()

      if (i+1) % 3 == 0:
        print("Epoch [{}/{}], step [{}/{}], loss: {:.4f}, Accuracy: {:.2f}%"
        .format(epoch+1, num_epochs, i+1, len(train_data_loader), loss.item(), accuracy.item() * 100))

    if (epoch+1) % val_every == 0:
      avrg_loss = validation(epoch+1, model, val_data_loader, criterion, device)

      if avrg_loss < best_loss:
        print("Best performance at epoch: {}".format(epoch+1))
        print("save model in ", save_dir)
        best_loss = avrg_loss
        save_model(model, save_dir)

def validation(epoch, model, data_loader, criterion, device):
  print("Start Val..")
  model.eval()

  with torch.no_grad():
    total = 0 
    correct = 0
    total_loss = 0
    cnt = 0

    for i, (imgs, labels) in enumerate(data_loader):
      images, labels = imgs.to(device), labels.to(device)
      outputs = model(images)
      loss = criterion(outputs, labels)

      total += imgs.size(0) # loss를 모두 더함 size(0)번째가 loss임
      _, argmax = torch.max(outputs, 1)
      accuracy = (labels == argmax).float().mean()
      correct += (labels == argmax).sum().item()
      total_loss += loss
      cnt += 1

    avrg_loss = total_loss / cnt
    print("Validation #{} Accuracy {:.2f}% Average Loss: {:.4f}".format(epoch, correct / total * 100, avrg_loss))

  model.train()

  return avrg_loss

def save_model(model, save_dir, file_name="best_model_vgg.pt"):
  os.makedirs(save_dir, exist_ok=True)
  output_path = os.path.join(save_dir, file_name)
  torch.save(model.state_dict(), output_path)

# train start
val_every = 1
save_dir = "/content/drive/MyDrive/practice/weights"

train(num_epochs, model, train_data_loader, criterion, optimizer, save_dir, val_every, device)

# test

def test(model, data_loader, device):
  print("Start test")

  correct = 0
  total = 0
  with torch.no_grad():

    for i, (imgs, labels) in enumerate(data_loader):
      imgs, labels = imgs.to(device), labels.to(device)

      outputs = net(imgs)
      _, argmax = torch.max(outputs, 1)
      total += imgs.size(0)
      correct += (labels == argmax).sum().item()

    print("Test Accuracy for {} images: {:.2f}%".format(total, correct / total * 100))

model_loader_path = "/content/drive/MyDrive/practice/weights/best_model_vgg.pt"
model.load_state_dict(torch.load(model_loader_path))

train_mode = False
if train_mode == True:
  train(num_epochs, model, train_data_loader, criterion, optimizer, save_dir, val_every, device)
else:
  test(model, test_data_loader, device)
