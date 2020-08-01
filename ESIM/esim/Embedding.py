import torch
import numpy as np
from esim.data_reader import data_process
from tqdm import tqdm


class Embedding(object):
    def __init__(self,model,word_to_indices,
                padding_idx=1,
                max_length=20,
                device='cuda'):
        self.model=model
        self.word_to_indices=word_to_indices
        #self.embedding_matrix=embedding_matrix
        self.padding_idx=1
        self.max_length=20
        self.device=device

#输入为 word的索引
    def prepare_data(self,words):
            length=min(len(words),self.max_length)
            out=torch.ones((self.max_length),  dtype=torch.long) * self.padding_idx
            end=min(len(words),self.max_length)
            out[:end] = torch.tensor(words)[:end]
            return torch.unsqueeze(out,0),torch.unsqueeze(torch.tensor(length),0)

    def trans_word(self,words):
        word_list=[]
        for i in words:
            try:
                word_list.append(self.word_to_indices[i])
            except:
                word_list.append(0)
                continue
        return word_list

    def encoded(self,words):
        words=self.trans_word(data_process(words))
        input_data=self.prepare_data(words)
        embedding=self.model._word_embedding(input_data[0].to(self.device))
        out=self.model._encoding(embedding,input_data[1].to(self.device))
        out=torch.mean(out,dim=1)
        return out.squeeze(0).cpu().detach().numpy()