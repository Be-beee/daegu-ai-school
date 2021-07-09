import torch
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import numpy as np
import csv

import matplotlib.pyplot as plt

# csv -> 활용 가능한 데이터
fish_data = list()
fish_names = list()
target_names = set()

with open('Fish.csv', newline='\n') as f:
    reader = csv.reader(f)
    for i, (name, wei, l1, l2, l3, h, wid) in enumerate(reader):
        if i > 0:
            target_names.add(name)
            fish_row = (wei, l1, l2, l3, h, wid)
            # fish_row = (l1, h, wid)
            fish_data.append(fish_row)
            fish_names.append(name)

fish_data = np.array(fish_data).astype(float)
target_names = sorted(target_names)
t_name_dict = {string : i for i, string in enumerate(target_names)}
print(t_name_dict)  # target_name -> number

fish_label = list()

for name in fish_names:
    fish_label.append(t_name_dict[name])
fish_label = np.array(fish_label)

print(fish_data)  # [Weight, Length1, Length2, Length3, Height, Width]
print(fish_label)  # Species




# 데이터 분석 작업

colors = {0: "yellow", 1: "black", 2: "blue", 3: "red", 4: "green", 5: "grey", 6: "violet"}
weights = [v[0] for v in fish_data]
length1 = [v[1] for v in fish_data]
length2 = [v[2] for v in fish_data]
length3 = [v[3] for v in fish_data]
heights = [v[4] for v in fish_data]
widths = [v[5] for v in fish_data]

plt.plot(fish_label, weights, 'o', label='weight')
# plt.plot(fish_label, length1, 'o', label='length1')
# plt.plot(fish_label, length2, 'o', label='length2')
# plt.plot(fish_label, length3, 'o', label='length3')
# plt.plot(fish_label, heights, 'o', label='heights')
# plt.plot(fish_label, widths, 'o', label='widths')

plt.legend()
plt.title('weight feature')
plt.show()









# 가공한 데이터 -> train_test_split

x_train, x_test, y_train, y_test = train_test_split(fish_data, fish_label, test_size=0.25, shuffle=True, stratify=fish_label, random_state=42)




# Net, CustomDatasets 정의

class Net(nn.Module):
    def __init__(self): # 모델 정의
        super(Net, self).__init__()

        self.fc1 = nn.Linear(3, 128)
        # self.fc2 = nn.Linear(128, 128)
        self.relu = nn.ReLU(128)
        self.fc3 = nn.Linear(128, 7)


    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        # x = self.fc2(x)
        x = self.fc3(x)

        return x



class CustomDataset(Dataset):
    def __init__(self, x_train, y_train, transforms=None):
        # data
        self.x = x_train
        self.y = y_train

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        x = np.array(x)
        y = np.array(y)

        return x, y





# 모델 정의
train_dataset = CustomDataset(x_train, y_train, transforms=None)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, drop_last=True)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)





# 모델 학습
# epoch = 1000
# total_loss = 0
#
# model.train() # 모델을 학습모드로 변경
# for i in range(epoch):
#     for x, y in train_loader:
#         x = x.float().to(device)
#         y = y.long().to(device)
#
#         outputs = model(x)
#
#         loss = criterion(outputs, y)
#         optimizer.zero_grad()
#         loss.backward()
#
#
#         total_loss += loss.item()
#         outputs = outputs.detach().numpy()
#
#         y = y.numpy()
#     if i == 999:
#         torch.save(model.state_dict(), f"fish_model_last.pth")
#     print(f"epoch -> {i}      loss -- > ", total_loss / len(x_train))
#     optimizer.step()
#     total_loss = 0



# test_dataset = CustomDataset(x_test, y_test, transforms=None)
# test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)
# model.eval()
# model.load_state_dict(torch.load('fish_model_last.pth'))
# err = 0
# for t_x, t_y in test_loader:
#     t_x = t_x.float().to(device)
#     t_y = t_y.numpy()
#     outputs = model(t_x)
#     top = torch.topk(outputs, 1)
#     top_index = top.indices.numpy()
#
#     for y, t in zip(t_y, top_index):
#         if y != t[0]:
#             err += 1
# print(f"test acc = {int((len(x_test) - err)/len(x_test)  * 100)}%")



# # 모델 평가
# model.eval()
# # model.load_state_dict(torch.load('fish_model_last.pth'))
# test_dataset = CustomDataset(x_test, y_test, transforms=None)
# test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
#
# accuracy = 0
# for x, y in test_loader:
#     x = x.float().to(device)
#     y = y.long().to(device)
#     outputs = model(x)
#     outputs = outputs.detach().numpy()
#     answer = y.numpy()[0]
#     max_value_idx = 0
#     for i, label in enumerate(outputs[0]):
#         if label == max(outputs[0]):
#             max_value_idx = i
#     # print(y.numpy(), outputs)
#     if answer == max_value_idx:
#         # print("----> correct!")
#         accuracy += 1
#     # else:
#         # print("----> not correct!")
#
# # print(f'accuracy: {accuracy/len(x_test) * 100}%')