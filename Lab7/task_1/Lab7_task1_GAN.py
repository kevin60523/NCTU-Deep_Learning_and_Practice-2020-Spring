import json
import numpy as np
from PIL import Image
import os

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Parameter
from torch import Tensor
from torch.autograd import Variable

from torchvision import transforms
import torchvision.models as models
from torchvision.utils import save_image

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable

seed_value = 17
torch.manual_seed(seed_value)
torch.cuda.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)


EPOCHS = 1000
BATCH_SIZE = 64
LR_G = 1e-4
LR_D = 4e-4
B1 = 0.0
B2 = 0.9
LATENT_DIM = 100
IMG_SIZE = 64
CHANNEL = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_iCLEVR_data(mode):
    if mode == 'train':
        data = json.load(open('../train.json'))
        obj = json.load(open('../objects.json'))
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
        data = json.load(open('../new_test.json'))
        obj = json.load(open('../objects.json'))
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
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))])       
    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, index):
        if self.mode == 'train':
            image_path = '../images/' + self.img_list[index]
            img = Image.open(image_path).convert('RGB')
            img = self.transform(img).numpy()

            return img, self.label_list[index]
        
        if self.mode == 'test':
            return self.label_list[index]            

train_dataset = ICLEVRDataset(train_data, 'train')
test_dataset = ICLEVRDataset(test_data, 'test')

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers = 8)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class evaluation_model():
    def __init__(self):
        #modify the path to your own path
        checkpoint = torch.load('../classifier_weight.pth')
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

def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)

class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)

class Self_Attn(nn.Module):
    def __init__(self,in_dim,activation):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value ,attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C , width, height)
        
        out = self.gamma * out + x
        return out

class Generator(nn.Module):
    def __init__(self, conv_dim=64):
        super(Generator, self).__init__()
        self.conditionExpand = nn.Sequential(nn.Linear(24, LATENT_DIM), nn.LeakyReLU())
       
        repeat_num = int(np.log2(IMG_SIZE)) - 3
        mult = 2 ** repeat_num
        layer1 = []
        layer1.append(SpectralNorm(nn.ConvTranspose2d(LATENT_DIM + LATENT_DIM, conv_dim * mult, 4)))
        layer1.append(nn.BatchNorm2d(conv_dim * mult))
        layer1.append(nn.ReLU())

        curr_dim = conv_dim * mult
        layer2 = []
        layer2.append(SpectralNorm(nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1)))
        layer2.append(nn.BatchNorm2d(int(curr_dim / 2)))
        layer2.append(nn.ReLU())
        
        curr_dim = int(curr_dim / 2)
        layer3 = []
        layer3.append(SpectralNorm(nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1)))
        layer3.append(nn.BatchNorm2d(int(curr_dim / 2)))
        layer3.append(nn.ReLU())

        curr_dim = int(curr_dim / 2)
        layer4 = []        
        layer4.append(SpectralNorm(nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1)))
        layer4.append(nn.BatchNorm2d(int(curr_dim / 2)))
        layer4.append(nn.ReLU())        
        
        curr_dim = int(curr_dim / 2)
        last = []
        last.append(nn.ConvTranspose2d(curr_dim, 3, 4, 2, 1))
        last.append(nn.Tanh())

        self.l1 = nn.Sequential(*layer1)
        self.l2 = nn.Sequential(*layer2)
        self.l3 = nn.Sequential(*layer3)
        self.l4 = nn.Sequential(*layer4)
        self.last = nn.Sequential(*last)
        
        self.attn1 = Self_Attn(128, 'relu')
        self.attn2 = Self_Attn(64,  'relu')
        
    def forward(self, z, c):
        c = self.conditionExpand(c).view(-1, LATENT_DIM)
        z = torch.cat((z, c), dim=1)        
        z = z.view(z.size(0), z.size(1), 1, 1)
        out = self.l1(z)
        out = self.l2(out)
        out = self.l3(out)
        out = self.attn1(out)
        out = self.l4(out)
        out = self.attn2(out)
        out = self.last(out)
        return out


class Discriminator(nn.Module):
    def __init__(self, conv_dim=64):
        super(Discriminator, self).__init__()
        self.conditionExpand = nn.Sequential(nn.Linear(24, IMG_SIZE * IMG_SIZE), nn.LeakyReLU())

        layer1 = []
        layer1.append(SpectralNorm(nn.Conv2d(3 + 1, conv_dim, 4, 2, 1)))
        layer1.append(nn.LeakyReLU(0.1))

        curr_dim = conv_dim
        layer2 = []
        layer2.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
        layer2.append(nn.LeakyReLU(0.1))

        curr_dim = curr_dim * 2
        layer3 = []
        layer3.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
        layer3.append(nn.LeakyReLU(0.1))

        curr_dim = curr_dim * 2
        layer4 = []
        layer4.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
        layer4.append(nn.LeakyReLU(0.1))
        self.l4 = nn.Sequential(*layer4)

        curr_dim = curr_dim * 2
        last = []
        last.append(nn.Conv2d(curr_dim, 1, 4))        
        
        self.l1 = nn.Sequential(*layer1)
        self.l2 = nn.Sequential(*layer2)
        self.l3 = nn.Sequential(*layer3)  
        self.last = nn.Sequential(*last)     

        self.attn1 = Self_Attn(256, 'relu')
        self.attn2 = Self_Attn(512, 'relu')        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, c):
        c = self.conditionExpand(c).view(-1, 1, IMG_SIZE, IMG_SIZE)
        x = torch.cat((x, c), dim=1)
        out = self.l1(x)
        out = self.l2(out)
        out = self.l3(out)
        out = self.attn1(out)
        out = self.l4(out)
        out = self.attn2(out)
        out = self.last(out)
        return self.sigmoid(out.squeeze(3).squeeze(2))

adversarial_loss = torch.nn.BCELoss()

model_G = Generator()
model_D = Discriminator()
model_E = evaluation_model()

optimizer_G = torch.optim.Adam(model_G.parameters(), lr=LR_G, betas=(B1, B2))
optimizer_D = torch.optim.Adam(model_D.parameters(), lr=LR_D, betas=(B1, B2))

model_G.to(device)
model_D.to(device)
adversarial_loss.to(device)

best = 0
fix_z = torch.randn(32, LATENT_DIM).to(device)
for epoch in tqdm(range(EPOCHS)):
    model_G.train()
    model_D.train()
    for imgs, labels in train_loader:
        imgs = imgs.to(device).float()
        labels =  labels.to(device).float()

        real = torch.full((imgs.size(0), 1), 1., requires_grad=False, device=device)
        fake = torch.full((imgs.size(0), 1), 0., requires_grad=False, device=device)
        
        # Train Discriminator with real image
        optimizer_D.zero_grad()
        pred = model_D(imgs, labels)        
        real_loss = adversarial_loss(pred, real)
        real_loss.backward()
        optimizer_D.step()

        # Train Discriminator with fake image
        optimizer_D.zero_grad()
        z = torch.randn(imgs.size(0), LATENT_DIM).to(device)
        gen_imgs = model_G(z, labels)
        pred = model_D(gen_imgs.detach(), labels)
        fake_loss = adversarial_loss(pred, fake)
        fake_loss.backward()
        optimizer_D.step()
    
        # Train Generator with fake image
        for _ in range(2):
            optimizer_G.zero_grad()
            z = torch.randn(imgs.size(0), LATENT_DIM).to(device)
            gen_imgs = model_G(z, labels)
            predicts = model_D(gen_imgs, labels)
            loss_g = adversarial_loss(predicts, real)
            loss_g.backward()
            optimizer_G.step()
                
    print(real_loss.item() + fake_loss.item())

    model_G.eval()
    model_D.eval()
    with torch.no_grad():
        for labels in test_loader:
            labels = labels.to(device).float()
            gen_imgs = model_G(fix_z, labels)
            score = model_E.eval(gen_imgs, labels) 
            if best < score:
                best = score
                torch.save(model_G.state_dict(), 'generator.pt')
            save_image(gen_imgs[0].data, '../output/%d.png' % epoch, nrow=8, normalize=True)    
            print('EPOCHS {} : {}'.format(epoch, score))
            print(best)
            print()
print(best)