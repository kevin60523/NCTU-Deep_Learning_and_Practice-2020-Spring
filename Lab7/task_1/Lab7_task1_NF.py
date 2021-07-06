import torch
import torch.nn as nn
import torch.nn.utils as utils
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import random

import json
from tqdm import tqdm

import torchvision
from torchvision import transforms
import torchvision.models as models

from PIL import Image
import os
import numpy as np

seed_value = 17
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
torch.cuda.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)

EPOCHS = 1000
BATCH_SIZE = 64
LR = 1e-3
IMG_SIZE = 64
NUM_CLASSES = 24

NUM_CHANNELS = 256
NUM_LEVELS = 2
NUM_STEPS = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_iCLEVR_data(mode):
    if mode == 'train':
        data = json.load(open('train.json'))
        obj = json.load(open('objects.json'))
        img = list(data.keys())
        label = list(data.values())
        for i in range(len(label)):
            for j in range(len(label[i])):
                label[i][j] = obj[label[i][j]]
            tmp = np.zeros(len(obj))
            tmp[label[i]] = 1
            label[i] = tmp
        return np.squeeze(img), np.squeeze(label)
    
    else:
        data = json.load(open('test.json'))
        obj = json.load(open('objects.json'))
        label = data
        for i in range(len(label)):
            for j in range(len(label[i])):
                label[i][j] = obj[label[i][j]]
            tmp = np.zeros(len(obj))
            tmp[label[i]] = 1
            label[i] = tmp
        return None, label

train_data = get_iCLEVR_data('train')
test_data = get_iCLEVR_data('test')

class ICLEVRDataset(Dataset):
    def __init__(self, data, mode):
        self.img_list, self.label_list = data
        self.mode = mode
        self.transform = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)),
                                             transforms.ToTensor()])       
    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, index):
        if self.mode == 'train':
            image_path = './images/' + self.img_list[index]
            img = Image.open(image_path).convert('RGB')
            img = self.transform(img).numpy()

            return img, self.label_list[index]
        
        if self.mode == 'test':
            return self.label_list[index]   

train_dataset = ICLEVRDataset(train_data, 'train')
test_dataset = ICLEVRDataset(test_data, 'test')
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers= 4)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

def mean_dim(tensor, dim):
    if isinstance(dim, int):
        dim = [dim]
    dim = sorted(dim)
    for d in dim:
        tensor = tensor.mean(dim=d, keepdim=True)
    return tensor

def bits_per_dim(x, nll):
    dim = np.prod(x.size()[1:])
    bpd = nll / (np.log(2) * dim)

    return bpd


def clip_grad_norm(optimizer, max_norm, norm_type=2):
    for group in optimizer.param_groups:
        utils.clip_grad_norm_(group['params'], max_norm, norm_type)


class NLLLoss(nn.Module):
    def __init__(self, k=64):
        super(NLLLoss, self).__init__()
        self.k = k

    def forward(self, z, sldj):
        prior_ll = -0.5 * (z ** 2 + np.log(2 * np.pi))
        prior_ll = prior_ll.flatten(1).sum(-1) - np.log(self.k) * np.prod(z.size()[1:])
        ll = prior_ll + sldj
        nll = -ll.mean()

        return nll

class ActNorm(nn.Module):
    def __init__(self, num_features, scale=1., return_ldj=False):
        super(ActNorm, self).__init__()
        self.register_buffer('is_initialized', torch.zeros(1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.logs = nn.Parameter(torch.zeros(1, num_features, 1, 1))

        self.num_features = num_features
        self.scale = float(scale)
        self.eps = 1e-6
        self.return_ldj = return_ldj

    def initialize_parameters(self, x):
        if not self.training:
            return

        with torch.no_grad():
            bias = -mean_dim(x.clone(), dim=[0, 2, 3])
            v = mean_dim((x.clone() + bias) ** 2, dim=[0, 2, 3])
            logs = (self.scale / (v.sqrt() + self.eps)).log()
            self.bias.data.copy_(bias.data)
            self.logs.data.copy_(logs.data)
            self.is_initialized += 1.

    def _center(self, x, reverse=False):
        if reverse:
            return x - self.bias
        else:
            return x + self.bias

    def _scale(self, x, sldj, reverse=False):
        logs = self.logs
        if reverse:
            x = x * logs.mul(-1).exp()
        else:
            x = x * logs.exp()

        if sldj is not None:
            ldj = logs.sum() * x.size(2) * x.size(3)
            if reverse:
                sldj = sldj - ldj
            else:
                sldj = sldj + ldj

        return x, sldj

    def forward(self, x, x_cond, ldj=None, reverse=False):
        if not self.is_initialized:
            self.initialize_parameters(x)

        if reverse:
            x, ldj = self._scale(x, ldj, reverse)
            x = self._center(x, reverse)
        else:
            x = self._center(x, reverse)
            x, ldj = self._scale(x, ldj, reverse)
        if self.return_ldj:
            return x, ldj

        return x

class Coupling(nn.Module):
    def __init__(self, in_channels, cond_channels, mid_channels):
        super(Coupling, self).__init__()
        self.nn = NN(in_channels, cond_channels, mid_channels, 2 * in_channels)
        self.scale = nn.Parameter(torch.ones(in_channels, 1, 1))

    def forward(self, x, x_cond, level, ldj, reverse=False):
        x_change, x_id = x.chunk(2, dim=1)

        st = self.nn(x_id, x_cond, level)
        s, t = st[:, 0::2, ...], st[:, 1::2, ...]
        s = self.scale * torch.tanh(s)

        if reverse:
            x_change = x_change * s.mul(-1).exp() - t
            ldj = ldj - s.flatten(1).sum(-1)
        else:
            x_change = (x_change + t) * s.exp()
            ldj = ldj + s.flatten(1).sum(-1)

        x = torch.cat((x_change, x_id), dim=1)

        return x, ldj


class NN(nn.Module):
    def __init__(self, in_channels, cond_channels, mid_channels, out_channels,
                 use_act_norm=False):
        super(NN, self).__init__()
        norm_fn = ActNorm if use_act_norm else nn.BatchNorm2d

        self.in_norm = norm_fn(in_channels)
        self.in_conv = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        self.in_condconv = nn.Conv2d(cond_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        nn.init.normal_(self.in_conv.weight, 0., 0.05)
        nn.init.normal_(self.in_condconv.weight, 0., 0.05)

        self.mid_conv1 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        self.mid_condconv1 = nn.Conv2d(cond_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        nn.init.normal_(self.mid_conv1.weight, 0., 0.05)
        nn.init.normal_(self.mid_condconv1.weight, 0., 0.05)

        self.mid_norm = norm_fn(mid_channels)
        self.mid_conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=1, padding=0, bias=False)
        self.mid_condconv2 = nn.Conv2d(cond_channels, mid_channels, kernel_size=1, padding=0, bias=False)
        nn.init.normal_(self.mid_conv2.weight, 0., 0.05)
        nn.init.normal_(self.mid_condconv2.weight, 0., 0.05)

        self.out_norm = norm_fn(mid_channels)
        self.out_conv = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=True)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)
        self.multiatt1 = nn.MultiheadAttention(1024, 1)
        self.multiatt2 = nn.MultiheadAttention(256, 1)
        self.multiatt3 = nn.MultiheadAttention(64, 1)
    def forward(self, x, x_cond, level):

        batch_size = x.size(0)
        x = self.in_norm(x)
        x = self.in_conv(x) + self.in_condconv(x_cond)
        x = F.relu(x)

        x = self.mid_conv1(x) + self.mid_condconv1(x_cond)
        x = self.mid_norm(x)
        x = F.relu(x)

        x = self.mid_conv2(x) + self.mid_condconv2(x_cond)
        x = self.out_norm(x)
        x = F.relu(x)

        x = self.out_conv(x)

        return x

class InvConv(nn.Module):
    def __init__(self, num_channels):
        super(InvConv, self).__init__()
        self.num_channels = num_channels
        
        w_init = np.random.randn(num_channels, num_channels)
        w_init = np.linalg.qr(w_init)[0].astype(np.float32)
        self.weight = nn.Parameter(torch.from_numpy(w_init))

    def forward(self, x, x_cond, sldj, reverse=False):
        ldj = torch.slogdet(self.weight)[1] * x.size(2) * x.size(3)

        if reverse:
            weight = torch.inverse(self.weight.double()).float()
            sldj = sldj - ldj
        else:
            weight = self.weight
            sldj = sldj + ldj

        weight = weight.view(self.num_channels, self.num_channels, 1, 1)
        z = F.conv2d(x, weight)
        return z, sldj

class Glow(nn.Module):

    def __init__(self, num_channels, num_levels, num_steps):
        super(Glow, self).__init__()
        self.c2c = nn.Linear(NUM_CLASSES, IMG_SIZE * IMG_SIZE)
        self.register_buffer('bounds', torch.tensor([0.95], dtype=torch.float32))
        self.flows = _Glow(in_channels=4 * 3,  # RGB image after squeeze
                           cond_channels=4,
                           mid_channels=num_channels,
                           num_levels=num_levels,
                           num_steps=num_steps)
        self.classify = nn.Linear(3 * IMG_SIZE * IMG_SIZE, 1)
    def forward(self, x, x_cond, reverse=False):
        x_cond = self.c2c(x_cond).view(x.size(0), 1, IMG_SIZE, IMG_SIZE)
        
        if reverse:
            sldj = torch.zeros(x.size(0), device=x.device)
        else:
            x, sldj = self._pre_process(x)
        
        x = squeeze(x)
        x_cond = squeeze(x_cond)
        x, sldj = self.flows(x, x_cond, sldj, reverse)
        x = squeeze(x, reverse=True)
        if reverse:
            return x, sldj
        else:

            pred = self.classify(x.view(x.size(0), -1))
            return x, sldj, pred

    def _pre_process(self, x):
        # y = x
        y = (x * 255. + torch.rand_like(x)) / 256.
        y = (2 * y - 1) * self.bounds
        y = (y + 1) / 2
        y = y.log() - (1. - y).log()

        # Save log-determinant of Jacobian of initial transform
        ldj = F.softplus(y) + F.softplus(-y) - F.softplus((1. - self.bounds).log() - self.bounds.log())
        sldj = ldj.flatten(1).sum(-1)

        return y, sldj


class _Glow(nn.Module):
    def __init__(self, in_channels, cond_channels, mid_channels, num_levels, num_steps):
        super(_Glow, self).__init__()
        self.level = num_levels
        self.steps = nn.ModuleList([_FlowStep(in_channels=in_channels,
                                              cond_channels=cond_channels,
                                              mid_channels=mid_channels)
                                    for _ in range(num_steps)])

        if num_levels > 1:
            self.next = _Glow(in_channels=2 * in_channels,
                              cond_channels=4 * cond_channels,
                              mid_channels=mid_channels,
                              num_levels=num_levels - 1,
                              num_steps=num_steps)
        else:
            self.next = None

    def forward(self, x, x_cond, sldj, reverse=False):
        if not reverse:
            for step in self.steps:
                x, sldj = step(x, x_cond, self.level, sldj, reverse)

        if self.next is not None:
            x = squeeze(x)
            x_cond = squeeze(x_cond)
            x, x_split = x.chunk(2, dim=1)
            x, sldj = self.next(x, x_cond, sldj, reverse)
            x = torch.cat((x, x_split), dim=1)
            x = squeeze(x, reverse=True)
            x_cond = squeeze(x_cond, reverse=True)

        if reverse:
            for step in reversed(self.steps):
                x, sldj = step(x, x_cond, self.level, sldj, reverse)

        return x, sldj


class _FlowStep(nn.Module):
    def __init__(self, in_channels, cond_channels, mid_channels):
        super(_FlowStep, self).__init__()

        # Activation normalization, invertible 1x1 convolution, affine coupling
        self.norm = ActNorm(in_channels, return_ldj=True)
        self.conv = InvConv(in_channels)
        self.coup = Coupling(in_channels // 2, cond_channels, mid_channels)

    def forward(self, x, x_cond, level, sldj=None, reverse=False):
        if reverse:
            x, sldj = self.coup(x, x_cond, level, sldj, reverse)
            x, sldj = self.conv(x, x_cond, sldj, reverse)
            x, sldj = self.norm(x, x_cond, sldj, reverse)
        else:
            x, sldj = self.norm(x, x_cond, sldj, reverse)
            x, sldj = self.conv(x, x_cond, sldj, reverse)
            x, sldj = self.coup(x, x_cond, level, sldj, reverse)

        return x, sldj


def squeeze(x, reverse=False):
    b, c, h, w = x.size()
    if reverse:
        # Unsqueeze
        x = x.view(b, c // 4, 2, 2, h, w)
        x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
        x = x.view(b, c // 4, h * 2, w * 2)
    else:
        # Squeeze
        x = x.view(b, c, h // 2, 2, w // 2, 2)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        x = x.view(b, c * 2 * 2, h // 2, w // 2)

    return x

class evaluation_model():
    def __init__(self):
        #modify the path to your own path
        checkpoint = torch.load('./classifier_weight.pth')
        self.resnet18 = models.resnet18(pretrained=False)
        self.resnet18.fc = nn.Sequential(
            nn.Linear(512,24),
            nn.Sigmoid()
        )
        self.resnet18.load_state_dict(checkpoint['model'])
        self.resnet18 = self.resnet18.cuda()
        self.resnet18.eval()
        self.classnum = 24
    def compute_acc(self, out, onehot_labels):
        batch_size = out.size(0)
        acc = 0
        total = 0
        for i in range(batch_size):
            k = int(onehot_labels[i].sum().item())
            total += k
            outv, outi = out[i].topk(k)
            lv, li = onehot_labels[i].topk(k)
            for j in outi:
                if j in li:
                    acc += 1
        return acc / total
    def eval(self, images, labels):
        with torch.no_grad():
            #your image shape should be (batch, 3, 64, 64)
            out = self.resnet18(images)
            acc = self.compute_acc(out.cpu(), labels.cpu())
            return acc

model = Glow(num_channels=NUM_CHANNELS, num_levels=NUM_LEVELS, num_steps=NUM_STEPS)
model.to(device)

model_E = evaluation_model()
loss_mse = nn.MSELoss()
loss_mse.to(device)
loss_fn = NLLLoss()
loss_fn.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
best = 0

fix_noise = torch.randn(32, 3, 64, 64).to(device)
for epoch in tqdm(range(EPOCHS)):
    model.train()
    
    for imgs, labels in train_loader:
        optimizer.zero_grad()
        
        imgs = imgs.to(device).float()
        labels = labels.to(device).float()
        num_label = torch.sum(labels, 1).unsqueeze(1) - 1
        z, sldj, pred = model(imgs, labels, reverse=False)
        loss = loss_fn(z, sldj)
        loss.backward()
        optimizer.step()
  
    model.eval() 
    for labels in test_loader:
        labels = labels.to(device).float()
        z, sldj = model(fix_noise, labels, reverse=True)

        score = model_E.eval(z, labels) 
        if best < score:
            best = score
            torch.save(model.state_dict(), 'nf.pt')
        print('EPOCHS {} : {}'.format(epoch, score))
        save_image(z.data, './output_NF/%d.png' % epoch, nrow=8, normalize=True)
        print('BEST: {}'.format(best))
        print()
print(best)
