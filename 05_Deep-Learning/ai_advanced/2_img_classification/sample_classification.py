import torch
import torchvision
import torchvision.transforms as transforms

# data image size 32 * 32

# 1. data augmentation
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4



# 2. dataset, dataloader
# train
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)
# test
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

# class information -> 라벨 정의
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
print(classes)

# 4. device 설정
# cuda
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device type >>", device)

# 부가 코드(train dataset이 어떻게 생겼나, 시각화코드)

import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
  img = img / 2 + 0.5  # unnormalize
  npimg = img.numpy()
  plt.imshow(np.transpose(npimg, (1, 2, 0)))
  plt.show()


# 학습용 데이터를 무작위로 가져와야함
dataiter = iter(trainloader)
images, labels = dataiter.next()

# 이미지 보여주기 위한 코드
imshow(torchvision.utils.make_grid(images))


# 정답(label) 출력
print(' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))

# 5. 모델 만들기

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(3, 6, 5)
    self.pool = nn.MaxPool2d(2, 2)
    self.conv2 = nn.Conv2d(6, 16, 5)
    self.fc1 = nn.Linear(16 * 5 * 5, 120)  # input = outchannel * filter size(h*w)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 10)


  def forward(self, x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = torch.flatten(x, 1)  # 배치를 제외한 모든 차원을 평탄화
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x

# 6. 모델을 불러오는 방법
net = Net()
net.to(device)

print(net)

# 7. optimizer, criterion
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# 8. train loop

for epoch in range(30):
  running_loss = 0.0
  for i, data in enumerate(trainloader, 0):
    # [inputs, labels]의 목록인 data로 부터 입력을 받은 후:
    inputs, labels = data[0].to(device), data[1].to(device)

    # 변화도(Gradient) 매개변수를 0으로 만들고
    optimizer.zero_grad()

    # 순전파 + 역전파 + 최적화를 한 후
    outputs = net(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    # 통계 출력
    running_loss += loss.item()
    if i % 2000 == 1999:
      print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
      running_loss = 0.0

print('Finished Training')

# PATH = './cifar_net.pth'
# torch.save(net.state_dict(), PATH)

# model 저장

PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)

# 평가
dataiter = iter(testloader)
images, labels = dataiter.next()

# 이미지 출력
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

net.load_state_dict(torch.load(PATH))
net = Net()

# 인퍼런스 cpu
outputs = net(images)

_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))

correct = 0
total = 0

with torch.no_grad():
  for data in testloader:
    images, labels = data
    outputs = net(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()
    
print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

# 각 분류(class)에 대한 예측값 계산을 위한 준비
correct_pred = { classname: 0 for classname in classes }
total_pred = { classname: 0 for classname in classes }

with torch.no_grad():
  for data in testloader:
    images, labels = data
    outputs = net(images)
    _, predictions = torch.max(outputs, 1)

    for label, prediction in zip(labels, predictions):
      if label == prediction:
        correct_pred[classes[label]] += 1
      total_pred[classes[label]] += 1

# 각 분류별 정확도(acc) 출력
for classname, correct_count in correct_pred.items():
  accuracy = 100 * float(correct_count) / total_pred[classname]
  print("Accuracy for class {:5s} is: {:.1f} %".format(classname, accuracy))

