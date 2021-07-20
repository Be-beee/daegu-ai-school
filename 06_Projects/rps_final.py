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
from sklearn.model_selection import train_test_split

from torchvision import datasets, models, transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader

# 1. device 설정 GPU 사용 가능 여부에 따라 device 정보 저장
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)


# 2. data location -> data/label 분리
data_dir = "./datasets_2"
rps_data_paths = sorted(glob.glob(os.path.join(data_dir, "*", "*")))
rps_data = []
rps_labels = []

for path in rps_data_paths:
    # img
    img = Image.open(path)
    img = img.convert("RGB")
    rps_data.append(img)

    # label
    label_name = os.path.dirname(path)
    label_str = str(label_name)

    if "scissors" in label_str:
      label = 0
    elif "rock" in label_str:
      label = 1
    elif "paper" in label_str:
      label = 2
    else:
      label = -1

    rps_labels.append(label)


# data random split

x_train, x_test, y_train, y_test = train_test_split(rps_data, rps_labels, test_size=0.2, shuffle=True, stratify=rps_labels, random_state=0)



# 3. hyper parameter
batch_size = 32
num_epochs = 20
learning_rate = 0.01



# 4. custom data set
class RPSDataset(Dataset):
    def __init__(self, x_train, y_train, transform=None):
        # data
        self.x = x_train
        self.y = y_train
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]

        if self.transform is not None:
            x = self.transform(x)

        return x, y




# data transforms
# image size 224 224

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

data_transforms = {
    'train': transforms.Compose([
          transforms.RandomRotation(30),
          transforms.RandomHorizontalFlip(p=0.5),
          transforms.RandomVerticalFlip(p=0.5),
          transforms.ColorJitter(brightness=0.5),
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



# dataset, dataloaderm, 모델 정의
train_dataset = RPSDataset(x_train, y_train, transform=data_transforms['train'])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = RPSDataset(x_test, y_test, transform=data_transforms['test'])
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

net = models.resnet34(pretrained=True)
# net = models.vgg11(pretrained=True)
# net = models.mobilenet_v3_large(pretrained=False, num_classes=3)

model = net.to(device)
print(model)





# loss function, optimizer scheduler
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.01)





def train(num_epochs, model, data_loader, criterion, optimizer, save_dir, device):

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

      if (i+1) % 10 == 0:
        print("Epoch [{}/{}], step [{}/{}], loss: {:.4f}, Accuracy: {:.2f}%"
        .format(epoch+1, num_epochs, i+1, len(train_loader), loss.item(), accuracy.item() * 100))

    if epoch == 9:
      save_model(model, save_dir)


def save_model(model, save_dir, file_name="last_model_resnet34.pt"):
  os.makedirs(save_dir, exist_ok=True)
  output_path = os.path.join(save_dir, file_name)
  torch.save(model.state_dict(), output_path)



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
      # print(argmax)
      total += imgs.size(0)
      correct += (labels == argmax).sum().item()

    print("Test Accuracy for {} images: {:.2f}%".format(total, correct / total * 100))





# train start
save_dir = "./models_2"
train(num_epochs, model, train_loader, criterion, optimizer, save_dir, device)





# test
model_loader_path = "./models_2/last_model_resnet34.pt"
model.load_state_dict(torch.load(model_loader_path))
print("--------------split_test_data--------------")
test(model, test_loader, device)





# setting new test dataset

# 2. data location -> data/label 분리
test_dir = "./test_sbk"
test_rps_data_paths = sorted(glob.glob(os.path.join(test_dir, "*", "*")))
test_rps_data = []
test_rps_labels = []

for path in test_rps_data_paths:
    # img
    img = Image.open(path)
    img = img.convert("RGB")
    test_rps_data.append(img)

    # label
    label_name = os.path.dirname(path)
    label_str = str(label_name)

    if "scissors" in label_str:
      label = 0
    elif "rock" in label_str:
      label = 1
    elif "paper" in label_str:
      label = 2
    else:
      label = -1

    test_rps_labels.append(label)



new_test_dataset = RPSDataset(test_rps_data, test_rps_labels, transform=data_transforms['test'])
new_test_loader = DataLoader(new_test_dataset, batch_size=batch_size, shuffle=False)




# new_test
print("--------------my_hands--------------")
# print(test_rps_labels)
model_loader_path = "./models_2/last_model_resnet34.pt"
model.load_state_dict(torch.load(model_loader_path))



test(model, new_test_loader, device)


