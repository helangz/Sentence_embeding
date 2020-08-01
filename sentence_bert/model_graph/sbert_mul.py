import torch.nn as nn
import torch.nn.functional as F
import torch
from transformers import BertModel

class SbertModel(nn.Module):
    def __init__(self,data_root='./bert_model/bert_wwm2/'):
        super(SbertModel,self).__init__()
        self.data_root=data_root
        self.bert = BertModel.from_pretrained(self.data_root).cuda()
        for param in self.bert.parameters():
            param.requires_grad = True
 
        self.hide1 = nn.Linear(768*4,768)
        self.hide2 = nn.Linear(768,384)
 
        self.dropout = nn.Dropout(0.5)
        self.out = nn.Linear(384,2)
 
    def forward(self, indextokens_a,input_mask_a,indextokens_b,input_mask_b):
        embedding_a = self.bert(indextokens_a,input_mask_a)[0]
        embedding_b = self.bert(indextokens_b,input_mask_b)[0]
 
        embedding_a = torch.mean(embedding_a,1)
        embedding_b = torch.mean(embedding_b,1)
 
        abs = torch.abs(embedding_a - embedding_b)
        ## 其他的模型
        #　余弦相似度　　torch.cosine_similarity  nn.CosineSimilarity(dim=1, eps=1e-6)
        def mul(x,y):
            return  x*y/(torch.sum(x**2,dim=1,keepdim=True)+
                               torch.sum(y**2,dim=1,keepdim=True))
    

        
        mul=mul(embedding_a,embedding_b)
        
        target_span_embedding = torch.cat((embedding_a, embedding_b,abs,mul), dim=1)
 
 
        hide_1 = F.relu(self.hide1(target_span_embedding))
        hide_2 = self.dropout(hide_1)
        hide = F.relu(self.hide2(hide_2))
        out_put = self.out(hide)
        return out_put