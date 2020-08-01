import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertModel,BertTokenizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import AdamW
from eval_model.auc_eval import dev
from model_graph.sbert_triplet  import SbertModel
from data_reader.sbert_reader import SbertDataset
from data_reader.triplet_reader import TripDataset


def traintrain(model,train_loader,dev_loader,path,dev_step=1800,show_step=10):
    model.to(device)
    model.train()
    criterion = TripletLoss()
 
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
    correct = 1
    total = 1
    print('Training begin!')
    for epoch in range(total_epochs):
        for step, (indextokens_a,_,input_mask_a,indextokens_b,_,input_mask_b,
                   indextokens_c,_,input_mask_c) in enumerate(train_loader):
            indextokens_a,input_mask_a,indextokens_b,input_mask_b,indextokens_c,input_mask_c = indextokens_a.to(device),input_mask_a.to(device),indextokens_b.to(device),input_mask_b.to(device),indextokens_c.to(device),input_mask_c.to(device)
            optimizer.zero_grad()
            emba,embb,embc = model(indextokens_a,input_mask_a,indextokens_b,input_mask_b,indextokens_c,input_mask_c)
            loss = criterion(emba,embb,embc)

            loss.backward()
            optimizer.step()
 
            if (step + 1) % show_step == 0:
                print("Train Epoch[{}/{}],step[{}/{}],loss:{:.6f}".format(epoch + 1, total_epochs, step + 1, len(train_loader),loss.item()))
 
            if (step + 1) %dev_step== 0:
                acc = dev(model, dev_loader)
                if bestAcc < acc:
                    bestAcc = acc
                    torch.save(model, path)
                print("DEV Epoch[{}/{}],step[{}/{}],bestAcc{:.6f}%,dev_acc{:.6f} %,loss:{:.6f}".format(epoch + 1, total_epochs, step + 1, len(train_loader),bestAcc*100,acc*100,loss.item()))
        scheduler.step(bestAcc)

        
def predict(model,test_loader):
    pass



from sklearn import metrics
def dev(model,dev_loader):
    model.eval()
    with torch.no_grad():
        y_pred=[]
        y_true=[]
        for step, (
                indextokens_a,_, input_mask_a, indextokens_b, _,input_mask_b, label) in tqdm(enumerate(
            dev_loader),desc='Dev Itreation:'):
            indextokens_a,input_mask_a, indextokens_b, input_mask_b,  = indextokens_a.to(device), input_mask_a.to(
                device), indextokens_b.to(device), input_mask_b.to(device)
            emb_a=model.bert(indextokens_a, input_mask_a)[0]
            emb_b=model.bert(indextokens_b, input_mask_b)[0]
            emb_a=emb_a.mean(1)
            emb_b=emb_b.mean(1)
            cos=torch.cosine_similarity(emb_a, emb_b, dim=1).cpu().detach().numpy()
            y_pred.extend(list(cos))
            y_true.extend(list(label.numpy()))            
        auc = metrics.roc_auc_score(y_true,y_pred,average='macro')
        return auc


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='send args')
 
    parser.add_argument("--save_path", type=str, default="./sbert_base.pkl")
    parser.add_argument("--train_path", type=str, default='../data/trip_data.csv')
    parser.add_argument("--dev_path", type=str, default="../data/test_data.csv")
    parser.add_argument("--data_root", type=str, default="./bert_model/bert_wwm2/")
    parser.add_argument("--dev_step", type=int, default=1800)
    parser.add_argument("--show_step", type=int, default=10)
    
    args = parser.parse_args()
    save_path=args.save_path
    train_path=args.train_path
    dev_path=args.dev_path
    data_root=args.data_root
    dev_step=args.dev_step
    show_step=args.show_step
    
    batch_size = 32
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_data = TripDataset(train_path)
    train_data=train_data.dropna()
    dev_data = SbertDataset(test_path)
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dataset=dev_data, batch_size=batch_size, shuffle=True)
    model = TripbertModel()
    train(model,train_loader,dev_loader,save_path,dev_step=dev_step,show_step=show_step)
