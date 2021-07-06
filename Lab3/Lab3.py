#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt


# In[2]:


seed_value = 17
torch.manual_seed(seed_value)
torch.cuda.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)


# In[3]:


BATCH_SIZE = 64
LR = 1e-2
EPOCHS = 300


# In[4]:


S4b_train = np.load('S4b_train.npz')
X11b_train = np.load('X11b_train.npz')
S4b_test = np.load('S4b_test.npz')
X11b_test = np.load('X11b_test.npz')


# In[5]:


class BCI_dataset(Dataset):
    def __init__(self, data_S4b, data_X11b):
        self.data = np.concatenate((data_S4b['signal'], data_X11b['signal']), axis=0)
        self.label = np.concatenate((data_S4b['label'], data_X11b['label']), axis=0)

        self.label = self.label - 1
        self.data = np.transpose(np.expand_dims(self.data, axis=1), (0, 1, 3, 2))

        mask = np.where(np.isnan(self.data))
        self.data[mask] = np.nanmean(self.data)
       
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index], self.label[index]


# In[6]:


train_dataset = BCI_dataset(S4b_train, X11b_train)
test_dataset = BCI_dataset(S4b_test, X11b_test)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# In[7]:


class EEGNet(nn.Module):
    def __init__(self, mode):
        super(EEGNet, self).__init__()
        if mode == 'elu':
            activation = nn.ELU()
        if mode == 'relu':
            activation = nn.ReLU()
        if mode == 'leakyrelu':
            activation = nn.LeakyReLU()
        
        self.firstconv = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(1, 51), stride=(1, 1), padding=(0, 25), bias=False),
        nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))

        self.deptwiseConv = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(2, 1), stride=(1, 1), groups=16, bias=False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            activation,
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4), padding=0),
            nn.Dropout(p=0.25))

        self.separableConv = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 15), stride=(1, 1), padding=(0, 7), bias=False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            activation,
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8), padding=0),
            nn.Dropout(p=0.25))
        self.classify = nn.Sequential(nn.Linear(in_features=736, out_features=2, bias=True))
       
    def forward(self, data):
        output = self.firstconv(data)
        output = self.deptwiseConv(output)
        output = self.separableConv(output)
        output = output.view(output.size(0), -1)
        output = self.classify(output)
        
        return output        


# In[8]:


def accuracy_comparison(model, epochs, elu_train_accuracy, elu_test_accuracy, relu_train_accuracy, relu_test_accuracy, leakyrelu_train_accuracy, leakyrelu_test_accuracy):
    elu_train, = plt.plot(epochs, elu_train_accuracy)
    elu_test, = plt.plot(epochs, elu_test_accuracy)
    relu_train, = plt.plot(epochs, relu_train_accuracy)
    relu_test, = plt.plot(epochs, relu_test_accuracy)
    leakyrelu_train, = plt.plot(epochs, leakyrelu_train_accuracy)
    leakyrelu_test, = plt.plot(epochs, leakyrelu_test_accuracy)
    plt.title('Activation function comparison(' + model + ')')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy(%)')
    plt.legend([elu_train, elu_test, relu_train, relu_test, leakyrelu_train, leakyrelu_test], ['elu_train', 'elu_test', 'relu_train', 'relu_test', 'leaky_relu_train', 'leaky_relu_test'], loc='lower right')
    plt.savefig(model + '.jpg', dpi=300, transparent=True)


# In[9]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
modes = ['elu', 'relu', 'leakyrelu']
for mode in modes:
    model = EEGNet(mode)
    ce_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-2, steps_per_epoch=len(train_loader), epochs=EPOCHS
    )

    model.to(device)
    ce_loss.to(device)
    
    if mode == 'elu':
        elu_train_accuracy = []
        elu_test_accuracy = []
    if mode == 'relu':
        relu_train_accuracy = []
        relu_test_accuracy = []
    if mode == 'leakyrelu':
        leakyrelu_train_accuracy = []
        leakyrelu_test_accuracy = []
    
    epochs = []
    best_accuracy = 0
    for epoch in range(EPOCHS):
        epochs.append(epoch + 1)
        model.train()
        num_total = 0
        num_corrects = 0
        for data, label in train_loader:
            optimizer.zero_grad()
            data = data.to(device).float()
            label = label.to(device).long()

            output = model(data)
            loss = ce_loss(output, label)
            loss.backward()
            optimizer.step()
            scheduler.step()

            num_corrects += (torch.argmax(output, dim=1) == label).sum().item()
            num_total += len(label)
        accuracy = num_corrects / num_total
        if mode == 'elu':
            elu_train_accuracy.append(accuracy)
        if mode == 'relu':
            relu_train_accuracy.append(accuracy)
        if mode == 'leakyrelu':
            leakyrelu_train_accuracy.append(accuracy)

        model.eval()
        num_total = 0
        num_corrects = 0
        for data, label in test_loader:

            data = data.to(device).float()
            label = label.to(device).long()

            output = model(data)

            num_corrects += (torch.argmax(output, dim=1) == label).sum().item()
            num_total += len(label)
        accuracy = num_corrects / num_total
        if accuracy > best_accuracy:
            best_accuracy = accuracy
        if mode == 'elu':
            elu_test_accuracy.append(accuracy)
        if mode == 'relu':
            relu_test_accuracy.append(accuracy)
        if mode == 'leakyrelu':
            leakyrelu_test_accuracy.append(accuracy)
    print(mode, best_accuracy)


# In[10]:


print(model)


# In[11]:


accuracy_comparison('EEGNet', epochs, elu_train_accuracy, elu_test_accuracy, relu_train_accuracy, relu_test_accuracy, leakyrelu_train_accuracy, leakyrelu_test_accuracy)


# In[12]:


class DeepConvNet(nn.Module):
    def __init__(self, mode):
        super(DeepConvNet, self).__init__()
        if mode == 'elu':
            activation = nn.ELU()
        if mode == 'relu':
            activation = nn.ReLU()
        if mode == 'leakyrelu':
            activation = nn.LeakyReLU()
        self.conv0 = nn.Conv2d(1, 25, kernel_size=(1,5))
        self.conv1 = nn.Sequential(
                nn.Conv2d(25, 25, kernel_size=(2, 1)),
                nn.BatchNorm2d(25, eps=1e-5, momentum=0.1),
                activation,
                nn.MaxPool2d(kernel_size=(1, 2)),
                nn.Dropout(p=0.5))
        self.conv2 = nn.Sequential(
                nn.Conv2d(25, 50, kernel_size=(1, 5)),
                nn.BatchNorm2d(50, eps=1e-5, momentum=0.1),
                activation,
                nn.MaxPool2d(kernel_size=(1, 2)),
                nn.Dropout(p=0.5))
        self.conv3 = nn.Sequential(
                nn.Conv2d(50, 100, kernel_size=(1, 5)),
                nn.BatchNorm2d(100, eps=1e-5, momentum=0.1),
                activation,
                nn.MaxPool2d(kernel_size=(1, 2)),
                nn.Dropout(p=0.5))
        self.conv4 = nn.Sequential(
                nn.Conv2d(100, 200, kernel_size=(1, 5)),
                nn.BatchNorm2d(200, eps=1e-5, momentum=0.1),
                activation,
                nn.MaxPool2d(kernel_size=(1, 2)),
                nn.Dropout(p=0.5))
        self.classify = nn.Linear(8600, 2)
        
    def forward(self, data):
        output = self.conv0(data)
        output = self.conv1(output)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.conv4(output)
        output = output.view(output.size(0), -1)
        output = self.classify(output)
        return output


# In[13]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
modes = ['elu', 'relu', 'leakyrelu']
for mode in modes:
    model = DeepConvNet(mode)
    ce_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-2, steps_per_epoch=len(train_loader), epochs=EPOCHS
    )

    model.to(device)
    ce_loss.to(device)
    
    if mode == 'elu':
        elu_train_accuracy = []
        elu_test_accuracy = []
    if mode == 'relu':
        relu_train_accuracy = []
        relu_test_accuracy = []
    if mode == 'leakyrelu':
        leakyrelu_train_accuracy = []
        leakyrelu_test_accuracy = []
    
    epochs = []
    best_accuracy = 0
    for epoch in range(EPOCHS):
        epochs.append(epoch + 1)
        model.train()
        num_total = 0
        num_corrects = 0
        for data, label in train_loader:
            optimizer.zero_grad()
            data = data.to(device).float()
            label = label.to(device).long()

            output = model(data)
            loss = ce_loss(output, label)
            loss.backward()
            optimizer.step()
            scheduler.step()

            num_corrects += (torch.argmax(output, dim=1) == label).sum().item()
            num_total += len(label)
        accuracy = num_corrects / num_total
        if mode == 'elu':
            elu_train_accuracy.append(accuracy)
        if mode == 'relu':
            relu_train_accuracy.append(accuracy)
        if mode == 'leakyrelu':
            leakyrelu_train_accuracy.append(accuracy)
    #     print('Train Epochs {},  \t{}'.format(epoch + 1, num_corrects / num_total))

        model.eval()
        num_total = 0
        num_corrects = 0
        for data, label in test_loader:

            data = data.to(device).float()
            label = label.to(device).long()

            output = model(data)

            num_corrects += (torch.argmax(output, dim=1) == label).sum().item()
            num_total += len(label)
        accuracy = num_corrects / num_total
        if accuracy > best_accuracy:
            best_accuracy = accuracy
        if mode == 'elu':
            elu_test_accuracy.append(accuracy)
        if mode == 'relu':
            relu_test_accuracy.append(accuracy)
        if mode == 'leakyrelu':
            leakyrelu_test_accuracy.append(accuracy)
    print(mode, best_accuracy)


# In[14]:


print(model)


# In[15]:


accuracy_comparison('DeepConvNet', epochs, elu_train_accuracy, elu_test_accuracy, relu_train_accuracy, relu_test_accuracy, leakyrelu_train_accuracy, leakyrelu_test_accuracy)


# In[ ]:




