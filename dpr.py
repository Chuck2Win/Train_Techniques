# -*- coding: utf-8 -*-
"""DPR.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1oVTDB3wM1TKJgi56_TxFzhrer1ZZoUEC
"""

! pip install faiss-gpu
! pip install sentencepiece
! pip install transformers
! pip install 'git+https://github.com/SKTBrain/KoBERT.git#egg=kobert_tokenizer&subdirectory=kobert_hf'

from kobert_tokenizer import KoBERTTokenizer
from transformers import BertModel

tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
model_q = BertModel.from_pretrained('skt/kobert-base-v1')
model_p = BertModel.from_pretrained('skt/kobert-base-v1')

from google.colab import drive
drive.mount('/content/MyDrive')

import os
import json
import torch
from tqdm import tqdm
import faiss
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F

train_cur = '/content/MyDrive/MyDrive/프로젝트/기계독해/도서_train/도서.json'
val_cur = '/content/MyDrive/MyDrive/프로젝트/기계독해/도서_valid/도서.json'

# data
f = open(train_cur,'rb')
train_data = json.load(f)
f.close()
f = open(val_cur,'rb')
val_data = json.load(f)
f.close()

def make_data(train_data):
    Train_data = []
    for i in range(len(train_data['data'])):
        title = train_data['data'][i]['title']
        for j in range(len(train_data['data'][i]['paragraphs'])):
            context = train_data['data'][i]['paragraphs'][j]['context']
            for k in range(len(train_data['data'][i]['paragraphs'][j]['qas'])):
                question = train_data['data'][i]['paragraphs'][j]['qas'][k]['question']
                answers = train_data['data'][i]['paragraphs'][j]['qas'][k]['answers']
                Train_data.append([title,context,question,answers])
    return Train_data

Train_data = make_data(train_data)
Val_data = make_data(val_data)

T = lambda i : torch.LongTensor(i)
class My_Dataset(Dataset):
    def __init__(self, data, tokenizer): #stopword):
        self.data = data
        self.tokenizer = tokenizer 
    def __getitem__(self, index):
        output1 = self.tokenizer(self.data[index][0],self.data[index][1],padding='max_length',max_length = 512,truncation=True) # title, passages
        output2 = self.tokenizer(self.data[index][2],padding='max_length',max_length = 64, truncation = True) # title, passages
        output1.token_type_ids =T(output1.token_type_ids).masked_fill(T(output1.token_type_ids).eq(3),1)
        output2.token_type_ids =T(output2.token_type_ids).masked_fill(T(output2.token_type_ids).eq(3),0)
        return T(output1.input_ids), T(output1.attention_mask), T(output1.token_type_ids), T(output2.input_ids), T(output2.attention_mask), T(output2.token_type_ids)
    def __len__(self):
        return len(self.data)



class dpr_encoder(nn.Module):
    def __init__(self, encoder_p, encoder_q):
        super().__init__()
        self.encoder_p = encoder_p
        self.encoder_q = encoder_q
    def forward(self, passage_input_ids, passage_attention_mask, passage_token_type_ids, question_input_ids, question_attention_mask, question_token_type_ids):
        ep = self.encoder_p(passage_input_ids, passage_attention_mask, passage_token_type_ids)
        eq = self.encoder_p(question_input_ids, question_attention_mask, question_token_type_ids)
        return ep.pooler_output, eq.pooler_output

def product_sim_score(Q,P):
    score = Q.matmul(P.T) # B,B
    b = Q.size(0)
    loss_1 = F.log_softmax(score,dim=-1)
    loss_2 = F.nll_loss(loss_1,torch.arange(b,device = Q.device))
    return loss_2,loss_1

output_dir = '/content/MyDrive/MyDrive/프로젝트/기계독해/output'

import logging
logger = logging.getLogger('logger') # 적지 않으면 root로 생성

# 2. logging level 지정 - 기본 level Warning
logger.setLevel(logging.INFO)

# 3. logging formatting 설정 - 문자열 format과 유사 - 시간, logging 이름, level - messages
formatter = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s] >> %(message)s')

# 4. handler : log message를 지정된 대상으로 전달하는 역할.
# SteamHandler : steam(terminal 같은 console 창)에 log message를 보냄
# FileHandler : 특정 file에 log message를 보내 저장시킴.
# handler 정의
stream_handler = logging.StreamHandler()
# handler에 format 지정
stream_handler.setFormatter(formatter)
# logger instance에 handler 삽입
logger.addHandler(stream_handler)

file_handler = logging.FileHandler(os.path.join(output_dir,'log.txt'), encoding='utf-8')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

tokenizer.padding_side='right'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = dpr_encoder(model_p,model_q).to(device)
batch_size = 8
mydataset = My_Dataset(Train_data,tokenizer)
train_dataset = My_Dataset(Train_data,tokenizer)
val_dataset = My_Dataset(Val_data,tokenizer)
train_dataloader = DataLoader(train_dataset,batch_size = batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset,batch_size = batch_size , shuffle=True)
optimizer = torch.optim.Adam(model.parameters(),1e-5)
epoch = 1
linear_scheduler = lambda step : min(1/4000*step,1.)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = linear_scheduler)
criterion = product_sim_score

def train(model, optimizer, scheduler, criterion, epoch, train_dataloader, logger):
    # BERT
    Loss = []
    for epoch in tqdm(range(1,epoch+1)):
        model.train()
        Loss_t = 0.
        for data in tqdm(train_dataloader,mininterval=3600):#tqdm(train_dataloader):
            optimizer.zero_grad()
            
            data = (i.to(device) for i in data)
            #input_ids, attention_masks, token_type_ids, labels = data
            a,b,c,d,e,f = data
            Q,P = model.forward(a,b,c,d,e,f)
            loss,_ = criterion(Q,P)
            Loss_t+=loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()
            torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0)
        Loss.append(Loss_t/len(train_dataloader))
        logger.info('epoch : %d ----- Train_Loss : %.4f'%(epoch+1,Loss[-1]))
        print('epoch : %d ----- Train_Loss : %.4f'%(epoch+1,Loss[-1]))
        with torch.no_grad():
            model.eval()
            score = 0
            Loss_val = 0.
            for data in val_dataloader:
                data = (i.to(args['device']) for i in data)
                a,b,c,d,e,f = data
                Q,P = model.forward(a,b,c,d,e,f)
                loss,_ = criterion(Q,P)
                Loss_val+=loss.item()
                score+=(_.argmax(-1)==torch.arange(batch_size,device = Q.device)).mean()
            Loss_val = Loss_val/len(val_dataloader)
            logger.info('epoch : %d ----- Val_Loss : %.4f'%(epoch,Loss_val))
            logger.info('accuracy : %.3f'%(score))
            logger.info('='*100)
        torch.save(model,output_dir+'model%d'%epoch)
    logger.info('train end')
    
    return Loss

# BERT
Loss = []
for epoch in tqdm(range(1,epoch+1)):
    model.train()
    Loss_t = 0.
    for data in tqdm(train_dataloader,mininterval=3600):#tqdm(train_dataloader):
        optimizer.zero_grad()
        
        data = (i.to(device) for i in data)
        #input_ids, attention_masks, token_type_ids, labels = data
        a,b,c,d,e,f = data
        Q,P = model.forward(a,b,c,d,e,f)
        loss,_ = criterion(Q,P)
        Loss_t+=loss.item()
        loss.backward()
        optimizer.step()
        scheduler.step()
        torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0)
    Loss.append(Loss_t/len(train_dataloader))
    logger.info('epoch : %d ----- Train_Loss : %.4f'%(epoch+1,Loss[-1]))
    print('epoch : %d ----- Train_Loss : %.4f'%(epoch+1,Loss[-1]))
    with torch.no_grad():
        model.eval()
        score = 0
        Loss_val = 0.
        for data in val_dataloader:
            data = (i.to(args['device']) for i in data)
            a,b,c,d,e,f = data
            Q,P = model.forward(a,b,c,d,e,f)
            loss,_ = criterion(Q,P)
            Loss_val+=loss.item()
            score+=(_.argmax(-1)==torch.arange(batch_size,device = Q.device)).mean()
        Loss_val = Loss_val/len(val_dataloader)
        logger.info('epoch : %d ----- Val_Loss : %.4f'%(epoch,Loss_val))
        logger.info('accuracy : %.3f'%(score))
        logger.info('='*100)
    torch.save(model,output_dir+'model%d'%epoch)
logger.info('train end')