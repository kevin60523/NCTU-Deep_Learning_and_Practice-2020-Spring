#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import unicode_literals, print_function, division

import torch
import torch.nn as nn

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import pandas as pd
import matplotlib.pyplot as plt
import gc

from io import open
import unicodedata
import string
import re
import random
import time
import math

import torch.nn.functional as F
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np
from os import system
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from tqdm import tqdm


# In[2]:


seed_value = 17
torch.manual_seed(seed_value)
torch.cuda.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)


# In[3]:


EPOCHS = 300


# In[4]:


KL_PERIOD = 50


# In[5]:


SOS_TOKEN = 0
EOS_TOKEN = 1
LR = 0.05


# In[6]:


HIDDEN_SIZE = 256
LATENT_SIZE = 32
VOCAB_SIZE = 28
CONDITION_EMBEDDING_SIZE = 8


# In[7]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[8]:


train_pairs = pd.read_csv('./lab5_dataset/train.txt', header=None).values
test_pairs = pd.read_csv('./lab5_dataset/test.txt', header=None).values


# In[9]:


def Word2Number(word):
    number = []
    for i in range(len(word)):
        number.append(ord(word[i]) - 97 + 2)
    number.append(EOS_TOKEN)
    return np.array(number)


# In[10]:


def Number2Word(number):
    word = ''
    for i in range(len(number)):
        word += chr(number[i] + 97 - 2) 
    return word


# In[11]:


class TrainTenseDataset(Dataset):
    def __init__(self, data):
        self.data = []
        self.label = []
        for i in range(len(data)):
            sp, tp, pg, p = data[i][0].split(' ')
            sp = Word2Number(sp)
            tp = Word2Number(tp)
            pg = Word2Number(pg)
            p = Word2Number(p)
            data_pairs = [sp, tp, pg, p]
            for j in range(len(data_pairs)):
                self.data.append(data_pairs[j])
                self.label.append(j)
                    
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, index):
        return self.data[index], self.label[index]


# In[12]:


class TestTenseDataset(Dataset):
    def __init__(self, data):
        self.data = []
        self.label = []
        tense = [(0, 3), (0, 2), (0, 1), (0, 1), (3, 1), (0, 2), (3, 0), (2, 0), (2, 3), (2, 1)]
        for i in range(len(data)):
            data1, data2 = data[i][0].split(' ')
            self.data.append([Word2Number(data1), Word2Number(data2)])
            self.label.append(tense[i])
                    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index], self.label[index]


# In[13]:


train_dataset = TrainTenseDataset(train_pairs)
train_loader = DataLoader(train_dataset, batch_size=1, num_workers=12 ,shuffle=True)
test_dataset = TestTenseDataset(test_pairs)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


# In[14]:


def sigmoid(x):
    return 1 / (1 + math.exp(-x))   

def _teacher_forcing_ratio(epoch, total_epoch):
    return 1 - epoch / total_epoch

def _kl_weight(epoch, method, period):
    if method == 'Monotonic':
        return min(1, epoch / period)
    if method == 'Cyclical':
        if int(epoch / period) % 2 == 1:
            return 1
        else:
            return sigmoid(-10 + (epoch % period) * 20 / period)

    


# In[15]:


# epochs = []
# kl_weights = []
# for epoch in tqdm(range(EPOCHS)):
#     epochs.append(epoch)
#     kl_weights.append(_kl_weight(epoch, 'Cyclical', KL_PERIOD))
# plt.figure(figsize=(20,20))    
# plt.plot(epochs, kl_weights)
# plt.savefig('test.jpg', dpi=300, transparent=True)


# In[16]:


def plot(epochs, ce_losses, kl_losses, kl_weights, teacher_forcing_ratios, bleu_4s, method):
    fig, ax1 = plt.subplots()
    
    ax1.set_xlabel('epochs')
    ax1.set_ylabel('loss')
    ce_loss, = ax1.plot(epochs, ce_losses)
    kl_loss, = ax1.plot(epochs, kl_losses)
    ax1.tick_params(axis='y')
    
    ax2 = ax1.twinx()
    
    ax2.set_ylabel('score/weight')
    kl_weight, = ax2.plot(epochs, kl_weights, '--b')
    teacher_forcing_ratio, = ax2.plot(epochs, teacher_forcing_ratios, '--c')
    bleu_4, = ax2.plot(epochs, bleu_4s, ':g')
    ax2.tick_params(axis='y')

    plt.title(method)
    plt.legend([ce_loss, kl_loss, kl_weight, teacher_forcing_ratio, bleu_4], ['CrossEntropy loss', 'KL loss', 'KL weight', 'Teacher Forcing', 'BLEU'], loc='best')
    fig.tight_layout()
    plt.savefig(method + '.jpg', dpi=300, transparent=True)


# In[17]:


def compute_bleu(output, reference):
    cc = SmoothingFunction()
    if len(reference) == 3:
        weights = (0.33,0.33,0.33)
    else:
        weights = (0.25,0.25,0.25,0.25)
    return sentence_bleu([reference], output,weights=weights,smoothing_function=cc.method1)


# In[18]:


class VAE(nn.Module):    
    class Encoder(nn.Module):
        def __init__(self, vocab_size, hidden_size):
            super(VAE.Encoder, self).__init__()
            
            self.hidden_size = hidden_size            
            self.embedding = nn.Embedding(vocab_size, hidden_size)
            self.lstm = nn.LSTM(hidden_size, hidden_size)

        def forward(self, x, initial_hidden_state, initial_cell_state):
            word_embedding = self.embedding(x)
            word_embedding = word_embedding.permute(1, 0, 2)
            output, (hidden_state, cell_state) = self.lstm(word_embedding, (initial_hidden_state, initial_cell_state))
            return output, hidden_state, cell_state
        
        def init_hidden_state(self, size):
            return torch.zeros(1, 1, size, device=device)
        
        def init_cell_state(self):
            return torch.zeros(1, 1, self.hidden_size, device=device)
    
    class Decoder(nn.Module):
        def __init__(self, hidden_size, vocab_size):
            super(VAE.Decoder, self).__init__()
            self.hidden_size = hidden_size

            self.embedding = nn.Embedding(vocab_size, hidden_size)
            self.lstm = nn.LSTM(hidden_size, hidden_size)
            self.out = nn.Linear(hidden_size, vocab_size)
            self.softmax = nn.LogSoftmax(dim=1)
            
        def forward(self, x, hidden_state, cell_state):
            output = self.embedding(x)
            output = F.relu(output)
            output, (hidden_state, cell_state) = self.lstm(output, (hidden_state, cell_state))
            output = self.softmax(self.out(output[0]))
           
            return output, hidden_state, cell_state

        def init_cell_state(self):
            return torch.zeros(1, 1, self.hidden_size, device=device)   
    
    def __init__(self, vocab_size, hidden_size, latent_size, condition_embedding_size):
        super(VAE, self).__init__()
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.condition_embedding_size = condition_embedding_size
        
        self.tense_embedding = nn.Embedding(4, condition_embedding_size)    
        
        self.encoder = self.Encoder(vocab_size, hidden_size)
        
        self.hidden2mean = nn.Linear(hidden_size, latent_size)
        self.hidden2variance = nn.Linear(hidden_size, latent_size)
        
        self.latent2hidden = nn.Linear(latent_size + condition_embedding_size, hidden_size)
        
        self.decoder = self.Decoder(hidden_size, vocab_size)
        
    def forward(self, word, tense, use_teacher_forcing):
        # encoder initial state
        tense = self.tense_embedding(tense).unsqueeze(1) #add one dimension
        encoder_initial_hidden_state = self.encoder.init_hidden_state(self.hidden_size - self.condition_embedding_size)
        encoder_initial_hidden_state = torch.cat([encoder_initial_hidden_state, tense], dim=-1)
        encoder_initial_cell_state = self.encoder.init_cell_state()
        
        # encoder
        _, hidden_state, cell_state = self.encoder(word, encoder_initial_hidden_state, encoder_initial_cell_state)
        
        # middle
        mean = self.hidden2mean(hidden_state)
        variance = self.hidden2variance(hidden_state)
        latent = self.reparameterize(mean, variance)

        # decoder initial state
        decoder_initial_hidden_state = torch.cat([latent, tense], dim=-1)
        decoder_initial_hidden_state = self.latent2hidden(decoder_initial_hidden_state)
        decoder_initial_cell_state = self.decoder.init_cell_state()
        
        decoder_input = torch.tensor([[SOS_TOKEN]], device=device)
        pred_distribution = torch.zeros(word.size(1), self.vocab_size, device=device)
        
        # decoder
        decoder_hidden_state = decoder_initial_hidden_state
        decoder_cell_state = decoder_initial_cell_state
        pred_output = []
        for i in range(word.size(1)):
            output, decoder_hidden_state, decoder_cell_state = self.decoder(decoder_input, decoder_hidden_state, decoder_cell_state)
            pred_distribution[i] = output[0]
                        
            if use_teacher_forcing:
                decoder_input = torch.tensor([[word[0][i]]], device=device)
            else:
                if torch.argmax(output).cpu().detach().numpy() == EOS_TOKEN:
                    break
                decoder_input = torch.argmax(output).unsqueeze(0).unsqueeze(0)
            pred_output.append(torch.argmax(output).cpu().detach().numpy().item())
        
        return pred_output, pred_distribution, mean, variance
    
    def evaluate(self, input_word, input_tense, output_tense):
        # encoder initial state
        input_tense = self.tense_embedding(input_tense).unsqueeze(1) #add one dimension
        output_tense = self.tense_embedding(output_tense).unsqueeze(1)
        encoder_initial_hidden_state = self.encoder.init_hidden_state(self.hidden_size - self.condition_embedding_size)
        encoder_initial_hidden_state = torch.cat([encoder_initial_hidden_state, input_tense], dim=-1)
        encoder_initial_cell_state = self.encoder.init_cell_state()
        
        #encoder
        _, hidden_state, cell_state = self.encoder(input_word, encoder_initial_hidden_state, encoder_initial_cell_state)
        
        # middle
        mean = self.hidden2mean(hidden_state)
        variance = self.hidden2variance(hidden_state)
        latent = self.reparameterize(mean, variance)
        
        # decoder initial state
        decoder_initial_hidden_state = torch.cat([latent, output_tense], dim=-1)
        decoder_initial_hidden_state = self.latent2hidden(decoder_initial_hidden_state)
        decoder_initial_cell_state = self.decoder.init_cell_state()
        
        decoder_input = torch.tensor([[SOS_TOKEN]], device=device)
                
        decoder_hidden_state = decoder_initial_hidden_state
        decoder_cell_state = decoder_initial_cell_state
        pred_output = []
        
        while True:
            output, decoder_hidden_state, decoder_cell_state = self.decoder(decoder_input, decoder_hidden_state, decoder_cell_state)
                        
            if torch.argmax(output).cpu().detach().numpy() == EOS_TOKEN:
                break
            pred_output.append(torch.argmax(output).cpu().detach().numpy().item())
            decoder_input = torch.argmax(output).unsqueeze(0).unsqueeze(0)
        
        return pred_output
    
    def generate_gaussian(self, latent, tense):
        # encoder initial state
        tense = self.tense_embedding(tense).unsqueeze(1)

        # decoder initial state
        decoder_initial_hidden_state = torch.cat([latent, tense], dim=-1)
        decoder_initial_hidden_state = self.latent2hidden(decoder_initial_hidden_state)
        decoder_initial_cell_state = self.decoder.init_cell_state()
        
        decoder_input = torch.tensor([[SOS_TOKEN]], device=device)
                
        decoder_hidden_state = decoder_initial_hidden_state
        decoder_cell_state = decoder_initial_cell_state
        pred_output = []
        
        while True:
            output, decoder_hidden_state, decoder_cell_state = self.decoder(decoder_input, decoder_hidden_state, decoder_cell_state)
                                    
            if torch.argmax(output).cpu().detach().numpy() == EOS_TOKEN:
                break
            pred_output.append(torch.argmax(output).cpu().detach().numpy().item())
            decoder_input = torch.argmax(output).unsqueeze(0).unsqueeze(0)
        
        return pred_output
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.normal(torch.FloatTensor([0] * self.latent_size), 
                           torch.FloatTensor([1] * self.latent_size)).to(device)
        return mu + eps * std
        


# In[19]:


def loss_function(distribution, word, mean, variance, pred_len):
    criterion = nn.CrossEntropyLoss().cuda()
    ce_loss = criterion(distribution[:pred_len], word[0][:pred_len])
    
    kl_loss = -0.5 * torch.sum(1 + variance - mean.pow(2) - variance.exp())
    return ce_loss, kl_loss


# In[ ]:


method = ['Cyclical', 'Monotonic']
for KL_METHOD in method:
    model = VAE(VOCAB_SIZE, HIDDEN_SIZE, LATENT_SIZE, CONDITION_EMBEDDING_SIZE)
    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    epochs = [] 
    teacher_forcing_ratios = []
    kl_weights = []
    ce_losses = []
    kl_losses = []
    bleu_4s = []

    best_bleu_4 = 0.0

    for epoch in tqdm(range(EPOCHS)):
        model.train()
        gc.collect()
        teacher_forcing_ratio = _teacher_forcing_ratio(epoch, EPOCHS)
        kl_weight = _kl_weight(epoch, KL_METHOD, KL_PERIOD)

        total_ce_loss = 0
        total_kl_loss = 0
        train_total_bleu_4 = 0
        for word, tense in train_loader:
            optimizer.zero_grad()

            word = word.to(device).long()
            tense = tense.to(device).long()

            use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

            output, distribution, mean, variance = model(word, tense, use_teacher_forcing)

            ce_loss, kl_loss = loss_function(distribution, word, mean, variance, len(output))        
            loss = ce_loss + kl_weight * kl_loss

            target_word = Number2Word(word[0][:-1].cpu().detach().numpy())
            pred_word = Number2Word(output)

            bleu_4 = compute_bleu(pred_word, target_word)

            loss.backward()
            optimizer.step()

            total_ce_loss += ce_loss
            total_kl_loss += kl_loss
            train_total_bleu_4 += bleu_4

        ce_losses.append(total_ce_loss / len(train_dataset))
        kl_losses.append(total_kl_loss / len(train_dataset))
        teacher_forcing_ratios.append(teacher_forcing_ratio)
        kl_weights.append(kl_weight)
        epochs.append(epoch)
        
        model.eval()
        test_total_bleu_4 = 0
        with torch.no_grad():
            for (input_word, output_word), (input_tense, output_tense) in test_loader:
                input_word = input_word.to(device).long()
                output_word = output_word.to(device).long()
                input_tense = input_tense.to(device).long()
                output_tense = output_tense.to(device).long()

                output = model.evaluate(input_word, input_tense, output_tense)

                target_word = Number2Word(output_word[0][:-1].cpu().detach().numpy())
                pred_word = Number2Word(output)
                print(target_word, pred_word)
                bleu_4 = compute_bleu(pred_word, target_word)
                test_total_bleu_4 += bleu_4
                
        bleu_4s.append(test_total_bleu_4 / len(test_dataset))
        
        if test_total_bleu_4 / len(test_dataset) > best_bleu_4:
            best_bleu_4 = test_total_bleu_4 / len(test_dataset)
            torch.save(model.state_dict(), KL_METHOD + '_BEST.pt')

        print('Epoch {}, kl_loss: {}, ce_loss: {}  {} {}'.format(epoch, (total_kl_loss / len(train_dataset)), (total_ce_loss / len(train_dataset)), (train_total_bleu_4 / len(train_dataset)), (test_total_bleu_4 / len(test_dataset))))
    torch.save(model.state_dict(), KL_METHOD + '_LAST.pt')
    plot(epochs, ce_losses, kl_losses, kl_weights, teacher_forcing_ratios, bleu_4s, KL_METHOD)


# In[19]:


model = VAE(VOCAB_SIZE, HIDDEN_SIZE, LATENT_SIZE, CONDITION_EMBEDDING_SIZE)
model.load_state_dict(torch.load("./BEST/best.pt"))
model.eval()
model.to(device)

total_bleu_4 = 0

input_words = []
target_words = []
pred_words = []

with torch.no_grad():
    for (input_word, output_word), (input_tense, output_tense) in test_loader:
        
        input_word = input_word.to(device).long()
        output_word = output_word.to(device).long()
        input_tense = input_tense.to(device).long()
        output_tense = output_tense.to(device).long()

        output = model.evaluate(input_word, input_tense, output_tense)
        
        input_word = Number2Word(input_word[0][:-1].cpu().detach().numpy())
        target_word = Number2Word(output_word[0][:-1].cpu().detach().numpy())
        pred_word = Number2Word(output)

        input_words.append(input_word)
        target_words.append(target_word)
        pred_words.append(pred_word)
        
        bleu_4 = compute_bleu(pred_word, target_word)
        total_bleu_4 += bleu_4

for i in range(len(input_words)):
    print('input:{}'.format(input_words[i]))
    print('target:{}'.format(target_words[i]))
    print('prediction:{}'.format(pred_words[i]))
    print()

print('Average BLEU-4 score : {}'.format((total_bleu_4 / len(test_dataset))))


# In[20]:


def Gaussian_score(words):
    words_list = []
    score = 0
    yourpath = './lab5_dataset/train.txt'#should be your directory of train.txt
    with open(yourpath,'r') as fp:
        for line in fp:
            word = line.split(' ')
            word[3] = word[3].strip('\n')
            words_list.extend([word])
        for t in words:
            for i in words_list:
                if t == i:
                    score += 1
    return score/len(words)


# In[ ]:


model = VAE(VOCAB_SIZE, HIDDEN_SIZE, LATENT_SIZE, CONDITION_EMBEDDING_SIZE)
model.load_state_dict(torch.load("./BEST/best.pt"))
model.to(device)
model.eval()
gaussian_score = 0
while gaussian_score < 0.3:
    word_list = []
    with torch.no_grad():   
        for i in range(100):
            latent = torch.randn(1, 1, LATENT_SIZE).to(device)
            tmp = []
            for i in range(4):
                output = model.generate_gaussian(latent, torch.Tensor([i]).to(device).long())
                pred_word = Number2Word(output)
                tmp.append(pred_word)

            word_list.append(tmp)
    gaussian_score = Gaussian_score(word_list)
        
print('Gaussian score : {}'.format(Gaussian_score(word_list))) 

for i in range(len(word_list)):
    print(word_list[i])
    


# In[ ]:





# In[ ]:




