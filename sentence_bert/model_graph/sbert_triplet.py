import torch.nn as nn
import torch.nn.functional as F
import torch
from transformers import BertModel
data_root='./bert_model/bert_wwm2/'
bert_config_file = data_root + 'bert_config.json'

init_checkpoint = data_root+'bert_model.ckpt'
bert_vocab_file = data_root + 'vocab.txt' 
 
class SbertModel(nn.Module):
    def __init__(self):
        super(SbertModel,self).__init__()
 
        self.bert = BertModel.from_pretrained(data_root).cuda()
        for param in self.bert.parameters():
            param.requires_grad = True
 
    def forward(self, indextokens_a,input_mask_a,indextokens_b,input_mask_b,indextokens_c,input_mask_c):
        embedding_a = self.bert(indextokens_a,input_mask_a)[0]
        embedding_b = self.bert(indextokens_b,input_mask_b)[0]
        embedding_c = self.bert(indextokens_c,input_mask_c)[0]
        embedding_a = torch.mean(embedding_a,1)
        embedding_b = torch.mean(embedding_b,1)
        embedding_c = torch.mean(embedding_c,1) 
        return embedding_a,embedding_b,embedding_c

    
import torch
from torch import Tensor
from typing import Union, Tuple, List, Iterable, Dict
import torch.nn.functional as F
from enum import Enum


class TripletDistanceMetric(Enum):
    """
    The metric for the triplet loss
    """
    COSINE = lambda x, y: 1 - F.cosine_similarity(x, y)
    EUCLIDEAN = lambda x, y: F.pairwise_distance(x, y, p=2)
    MANHATTAN = lambda x, y: F.pairwise_distance(x, y, p=1)

class TripletLoss(nn.Module):
    def __init__(self, distance_metric=TripletDistanceMetric.EUCLIDEAN, triplet_margin=1):
        super(TripletLoss, self).__init__()
        self.distance_metric = distance_metric
        self.triplet_margin = triplet_margin
    def forward(self,rep_anchor, rep_pos, rep_neg):

        distance_pos = self.distance_metric(rep_anchor, rep_pos)
        distance_neg = self.distance_metric(rep_anchor, rep_neg)
        losses = F.relu(distance_pos - distance_neg + self.triplet_margin)
        return losses.mean()