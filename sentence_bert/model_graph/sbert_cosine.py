import torch.nn as nn
import torch.nn.functional as F
import torch
from transformers import BertModel

data_root='./bert_model/bert_wwm2/'
class SbertModel(nn.Module):
    def __init__(self):
        super(SbertModel,self).__init__()
 
        self.bert = BertModel.from_pretrained(data_root).cuda()
        for param in self.bert.parameters():
            param.requires_grad = True 
        self.out = nn.Linear(1,1)
 
    def forward(self, indextokens_a,input_mask_a,indextokens_b,input_mask_b):
        embedding_a = self.bert(indextokens_a,input_mask_a)[0]
        embedding_b = self.bert(indextokens_b,input_mask_b)[0] 
        embedding_a = torch.mean(embedding_a,1)
        embedding_b = torch.mean(embedding_b,1)
        cos = torch.cosine_similarity(embedding_a,embedding_b,dim=1, eps=1e-6)      
        return cos