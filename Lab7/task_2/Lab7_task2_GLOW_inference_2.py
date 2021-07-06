import json

import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as utils

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image

import os
import numpy as np
import random


import torch.utils.data as data
import torchvision

from tqdm import tqdm

seed_value = 17
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
torch.cuda.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)

EPOCHS = 200
BATCH_SIZE = 32
LR = 1e-3

IMG_SIZE = 64
NUM_CHANNELS = 128
NUM_LEVELS = 3
NUM_STEPS = 8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_CelebA_data():
    img_list = os.listdir('./CelebA-HQ-img')
    label_list = []
    f = open('CelebA-HQ-attribute-anno.txt', 'r')
    num_imgs = int(f.readline()[:-1])
    attrs = f.readline()[:-1].split(' ')
    for idx in range(num_imgs):
        line = f.readline()[:-1].split(' ')
        label = line[2:]
        label = list(map(int, label))
        label_list.append(label)
    f.close()
    return img_list, label_list

train_data = get_CelebA_data() 

class CelebADataset(Dataset):
    def __init__(self, data):
        self.img_list, self.label_list = data
        self.transform = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)),
                                             transforms.ToTensor()])
    def __len__(self):
        return len(self.label_list)
    
    def __getitem__(self, index):
        image_path = './CelebA-HQ-img/' + self.img_list[index]
        img = Image.open(image_path).convert('RGB')
        img = self.transform(img).numpy()

        return img, self.label_list[index]
    
train_dataset = CelebADataset(train_data)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=12)

def mean_dim(tensor, dim=None, keepdims=False):
    if dim is None:
        return tensor.mean()
    else:
        if isinstance(dim, int):
            dim = [dim]
        dim = sorted(dim)
        for d in dim:
            tensor = tensor.mean(dim=d, keepdim=True)
        if not keepdims:
            for i, d in enumerate(dim):
                tensor.squeeze_(d-i)
        return tensor

def bits_per_dim(x, nll):
    dim = np.prod(x.size()[1:])
    bpd = nll / (np.log(2) * dim)

    return bpd


def clip_grad_norm(optimizer, max_norm, norm_type=2):
    for group in optimizer.param_groups:
        utils.clip_grad_norm_(group['params'], max_norm, norm_type)


class NLLLoss(nn.Module):
    def __init__(self, k=256):
        super(NLLLoss, self).__init__()
        self.k = k

    def forward(self, z, sldj):
        prior_ll = -0.5 * (z ** 2 + np.log(2 * np.pi))
        prior_ll = prior_ll.flatten(1).sum(-1) \
            - np.log(self.k) * np.prod(z.size()[1:])
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
            bias = -mean_dim(x.clone(), dim=[0, 2, 3], keepdims=True)
            v = mean_dim((x.clone() + bias) ** 2, dim=[0, 2, 3], keepdims=True)
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

    def forward(self, x, ldj=None, reverse=False):
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
    def __init__(self, in_channels, mid_channels):
        super(Coupling, self).__init__()
        self.nn = NN(in_channels, mid_channels, 2 * in_channels)
        self.scale = nn.Parameter(torch.ones(in_channels, 1, 1))

    def forward(self, x, ldj, reverse=False):
        x_change, x_id = x.chunk(2, dim=1)

        st = self.nn(x_id)
        s, t = st[:, 0::2, ...], st[:, 1::2, ...]
        s = self.scale * torch.tanh(s)

        # Scale and translate
        if reverse:
            x_change = x_change * s.mul(-1).exp() - t
            ldj = ldj - s.flatten(1).sum(-1)
        else:
            x_change = (x_change + t) * s.exp()
            ldj = ldj + s.flatten(1).sum(-1)

        x = torch.cat((x_change, x_id), dim=1)

        return x, ldj


class NN(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels,
                 use_act_norm=False):
        super(NN, self).__init__()
        norm_fn = ActNorm if use_act_norm else nn.BatchNorm2d

        self.in_norm = norm_fn(in_channels)
        self.in_conv = nn.Conv2d(in_channels, mid_channels,
                                 kernel_size=3, padding=1, bias=False)
        nn.init.normal_(self.in_conv.weight, 0., 0.05)

        self.mid_norm = norm_fn(mid_channels)
        self.mid_conv = nn.Conv2d(mid_channels, mid_channels,
                                  kernel_size=1, padding=0, bias=False)
        nn.init.normal_(self.mid_conv.weight, 0., 0.05)

        self.out_norm = norm_fn(mid_channels)
        self.out_conv = nn.Conv2d(mid_channels, out_channels,
                                  kernel_size=3, padding=1, bias=True)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x):
        x = self.in_norm(x)
        x = F.relu(x)
        x = self.in_conv(x)

        x = self.mid_norm(x)
        x = F.relu(x)
        x = self.mid_conv(x)

        x = self.out_norm(x)
        x = F.relu(x)
        x = self.out_conv(x)

        return x

class InvConv(nn.Module):
    def __init__(self, num_channels):
        super(InvConv, self).__init__()
        self.num_channels = num_channels

        # Initialize with a random orthogonal matrix
        w_init = np.random.randn(num_channels, num_channels)
        w_init = np.linalg.qr(w_init)[0].astype(np.float32)
        self.weight = nn.Parameter(torch.from_numpy(w_init))

    def forward(self, x, sldj, reverse=False):
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

        # Use bounds to rescale images before converting to logits, not learned
        self.register_buffer('bounds', torch.tensor([0.9], dtype=torch.float32))
        self.flows = _Glow(in_channels=4 * 3,  # RGB image after squeeze
                           mid_channels=num_channels,
                           num_levels=num_levels,
                           num_steps=num_steps)

    def forward(self, x, reverse=False):
        if reverse:
            sldj = torch.zeros(x.size(0), device=x.device)
        else:
            # Expect inputs in [0, 1]
            if x.min() < 0 or x.max() > 1:
                raise ValueError('Expected x in [0, 1], got min/max {}/{}'
                                 .format(x.min(), x.max()))

            # De-quantize and convert to logits
            x, sldj = self._pre_process(x)

        x = squeeze(x)
        x, sldj = self.flows(x, sldj, reverse)
        x = squeeze(x, reverse=True)

        return x, sldj

    def _pre_process(self, x):
        y = (x * 255. + torch.rand_like(x)) / 256.
        y = (2 * y - 1) * self.bounds
        y = (y + 1) / 2
        y = y.log() - (1. - y).log()

        # Save log-determinant of Jacobian of initial transform
        ldj = F.softplus(y) + F.softplus(-y) \
            - F.softplus((1. - self.bounds).log() - self.bounds.log())
        sldj = ldj.flatten(1).sum(-1)

        return y, sldj


class _Glow(nn.Module):
    def __init__(self, in_channels, mid_channels, num_levels, num_steps):
        super(_Glow, self).__init__()
        self.steps = nn.ModuleList([_FlowStep(in_channels=in_channels,
                                              mid_channels=mid_channels)
                                    for _ in range(num_steps)])

        if num_levels > 1:
            self.next = _Glow(in_channels=2 * in_channels,
                              mid_channels=mid_channels,
                              num_levels=num_levels - 1,
                              num_steps=num_steps)
        else:
            self.next = None

    def forward(self, x, sldj, reverse=False):
        if not reverse:
            for step in self.steps:
                x, sldj = step(x, sldj, reverse)

        if self.next is not None:
            x = squeeze(x)
            x, x_split = x.chunk(2, dim=1)
            x, sldj = self.next(x, sldj, reverse)
            x = torch.cat((x, x_split), dim=1)
            x = squeeze(x, reverse=True)

        if reverse:
            for step in reversed(self.steps):
                x, sldj = step(x, sldj, reverse)

        return x, sldj


class _FlowStep(nn.Module):
    def __init__(self, in_channels, mid_channels):
        super(_FlowStep, self).__init__()

        # Activation normalization, invertible 1x1 convolution, affine coupling
        self.norm = ActNorm(in_channels, return_ldj=True)
        self.conv = InvConv(in_channels)
        self.coup = Coupling(in_channels // 2, mid_channels)

    def forward(self, x, sldj=None, reverse=False):
        if reverse:
            x, sldj = self.coup(x, sldj, reverse)
            x, sldj = self.conv(x, sldj, reverse)
            x, sldj = self.norm(x, sldj, reverse)
        else:
            x, sldj = self.norm(x, sldj, reverse)
            x, sldj = self.conv(x, sldj, reverse)
            x, sldj = self.coup(x, sldj, reverse)

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

model = Glow(num_channels=NUM_CHANNELS, num_levels=NUM_LEVELS, num_steps=NUM_STEPS) 
model.load_state_dict(torch.load("./task2_2.pt"))
model.to(device)
model.eval()

transform = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)),
                                             transforms.ToTensor()])

a_img_name = '69.jpg' #60 900 69
b_img_name = '123.jpg'  #18 71 123

image_path = './CelebA-HQ-img/' + a_img_name
a_img = Image.open(image_path).convert('RGB')
a_img = transform(a_img)
numpy_a_img = a_img.numpy()

image_path = './CelebA-HQ-img/' + b_img_name
b_img = Image.open(image_path).convert('RGB')
b_img = transform(b_img)
numpy_b_img = b_img.numpy()

a_img = a_img.to(device).unsqueeze(0)
a_z, _ = model(a_img, reverse=False)

b_img = b_img.to(device).unsqueeze(0)
b_z, _ = model(b_img, reverse=False)

difference = (b_z.detach().cpu().numpy() - a_z.detach().cpu().numpy()).squeeze(0)

all_img = np.full((8, 3, 64, 64), a_z.cpu().detach().numpy())
for i in range(8):
    all_img[i] += difference / 8 * i
all_img = torch.from_numpy(all_img).to(device)
x, _ = model(all_img, reverse=True)
images = torch.sigmoid(x)
images_concat = torchvision.utils.make_grid(images, nrow=8, padding=2, pad_value=255)
torchvision.utils.save_image(images_concat, './output/interpolation.png')