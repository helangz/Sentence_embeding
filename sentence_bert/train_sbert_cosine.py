#!/usr/bin/env python
# coding: utf-8
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertModel,BertTokenizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import AdamW
from eval_model.auc_eval import dev
from model_graph.sbert_cosine  import SbertModel
from data_reader.sbert_reader import SbertDataset


def train(model,train_loader,dev_loader,save_path,dev_step=1800,show_step=20):
    model.to(device)
    model.train()
    criterion = torch.nn.MSELoss(size_average=False)
 
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    #设置模型参数的权重衰减
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    #学习率的设置
    optimizer_params = {'lr': 1e-5, 'eps': 1e-6, 'correct_bias': False}
    #AdamW 这个优化器是主流优化器
    optimizer = AdamW(optimizer_grouped_parameters, **optimizer_params)
 
    #学习率调整器，检测准确率的状态，然后衰减学习率
    scheduler = ReduceLROnPlateau(optimizer,mode='max',factor=0.5,min_lr=1e-7, patience=5,verbose= True, threshold=0.0001, eps=1e-08)
 
    t_total = len(train_loader)
    total_epochs = 5
    bestAcc = 0
    correct = 0
    total = 0
    print('Training begin!')
    for epoch in range(total_epochs):
        for step, (indextokens_a,input_mask_a,indextokens_b,input_mask_b,label) in enumerate(train_loader):
            indextokens_a,input_mask_a,indextokens_b,input_mask_b,label = indextokens_a.to(device),input_mask_a.to(device),indextokens_b.to(device),input_mask_b.to(device),label.to(device)
            optimizer.zero_grad()
            out_put = model(indextokens_a,input_mask_a,indextokens_b,input_mask_b)
            loss = criterion(out_put, label.float())
            loss.backward()
            optimizer.step()
 
            if (step + 1) % show_step == 0:

                print("Train Epoch[{}/{}],step[{}/{}],,loss:{:.6f}".format(epoch + 1, total_epochs, step + 1, len(train_loader),loss.item()))
 
            if (step + 1) % dev_step == 0:

                acc = dev(model, dev_loader)
                if bestAcc < acc:
                    bestAcc = acc
                    torch.save(model, path)
                print("DEV Epoch[{}/{}],step[{}/{}],bestAcc{:.6f}%,dev_acc{:.6f} %,loss:{:.6f}".format(epoch + 1, total_epochs, step + 1, len(train_loader),bestAcc*100,acc*100,loss.item()))
        scheduler.step(bestAcc)
        
if __name__ == '__main__':
    
    import argparse
    parser = argparse.ArgumentParser(description='send args')
    parser.add_argument("--save_path", type=str, default="./sbert_base.pkl")
    parser.add_argument("--train_path", type=str, default="../data/train_data.csv")
    parser.add_argument("--dev_path", type=str, default="../data/test_data.csv")
    parser.add_argument("--data_root", type=str, default="./bert_model/bert_wwm2/")
    parser.add_argument("--dev_step", type=int, default=1800)
    parser.add_argument("--show_step", type=int, default=20)
    
    args = parser.parse_args()
    save_path=args.save_path
    train_path=args.train_path
    dev_path=args.dev_path
    data_root=args.data_root
    dev_step=args.dev_step
    show_step=args.show_step
    
    batch_size = 32
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_data = SbertDataset(train_path,data_root)
    dev_data = SbertDataset(dev_path,data_root)
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dataset=dev_data, batch_size=batch_size, shuffle=True)
    model = SbertModel()
    train(model,train_loader,dev_loader,save_path,dev_step=dev_step,show_step=show_step)







