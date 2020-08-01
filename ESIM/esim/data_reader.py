"""
Preprocessor and dataset definition for NLI.
"""
# Aurelien Coet, 2018.

import string
import torch
import numpy as np

from collections import Counter
from torch.utils.data import Dataset


import re
import jieba
def data_process(words):
    pattern=re.compile(r"[.。!！?？｡＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏.\n]")
    words=pattern.sub('',words)
    # 分词
    words=list(jieba.cut_for_search(words))
    #words=list(jieba.lcut(words, cut_all=True))
    #words=list(set(words1+words2))
    stop_word=['吗','啊','的','恩','嗯','呢','了','',' ',' ']
    words=[i for i in words if i not in stop_word]
    return words

class Preprocessor(object):
    def __init__(self):
        pass
    def seg_data(self,data):
        data['testa']=data['testa'].apply(lambda x:' '.join(data_process(x)))
        data['testb']=data['testb'].apply(lambda x:' '.join(data_process(x)))
        return data
        
    def set_data(self,data):
        texta_set = set(' '.join(data['testa']).split())
        textb_set = set(' '.join(data['testb']).split())
        word_set = texta_set | textb_set | {'pos', 'pad'}
        return word_set

    ## build embeddings_index
    def read_embedding(self,path='../model/sgns.zhihu.word'):
        embeddings_index = {}
        f = open(path,encoding='utf-8')
        i = 0
        for line in f:
            if i ==0:  
                i+=1
                continue
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()
        return embeddings_index
    def build_embedding_matrix(self,word_set,embeddings_index):
        word_to_indices = {}
        indices = 0
        word_to_indices['pad'] = 1   # pad
        indices += 1 
        word_to_indices['pos'] = 0   # no find its word means not in embedmatrix or not in texts
        indices += 1 

        for word in word_set:
            word_to_indices[word] = indices
            indices += 1

        print('Preparing embeddings matrix...')
        mean_word_vector = np.mean(list(embeddings_index.values()), axis=0)
        embedding_dim = len(list(embeddings_index.values())[0])
        num_words = len(word_to_indices)
        print(num_words)
        print(indices)
        embedding_matrix = np.zeros((num_words+2, embedding_dim))
        found_words = 0
        lost = 0
        for word in word_to_indices.keys():
            try:
                index = word_to_indices[word]
                embedding_vector = embeddings_index[word]
                embedding_matrix[index] = embedding_vector
                found_words += 1
            except:
                lost += 1
                print(word)

        embedding_matrix[word_to_indices['pad']] = np.zeros(embedding_dim)
        embedding_matrix[word_to_indices['pos']] = mean_word_vector
        print('{} words find {} lost in our vocabulary had {} vectors and appear more than the min frequency'.format(
            found_words, lost, 'w2v'))
        return embedding_matrix,word_to_indices
    
    def gen_input_data(self,data,word_to_indices,start=0):
        tokens = []
        for row in data['testa']:
            tokens.append(row.split(' '))
        texta = [[word_to_indices[i]  for i in t if i in word_to_indices]+[0] for t in tokens]
        tokens = []
        for row in data['testb']:
            tokens.append(row.split(' '))
        textb = [[word_to_indices[i] for i in t if i in word_to_indices] + [0]  for t in tokens]
        input_for_esim = {
        'premises': texta,
        'hypotheses':  textb,
        'labels': list(data['label'])
        }
        return input_for_esim

class NLIDataset(Dataset):
    """
    Dataset class for Natural Language Inference datasets.

    The class can be used to read preprocessed datasets where the premises,
    hypotheses and labels have been transformed to unique integer indices
    (this can be done with the 'preprocess_data' script in the 'scripts'
    folder of this repository).
    """
    def __init__(self,
                 data,
                 padding_idx=0,
                 max_premise_length=None,
                 max_hypothesis_length=None):

        self.premises_lengths = [len(seq) for seq in data["premises"]]
        self.max_premise_length = max_premise_length
        if self.max_premise_length is None:
            self.max_premise_length = max(self.premises_lengths)

        self.hypotheses_lengths = [len(seq) for seq in data["hypotheses"]]
        self.max_hypothesis_length = max_hypothesis_length
        if self.max_hypothesis_length is None:
            self.max_hypothesis_length = max(self.hypotheses_lengths)

        self.num_sequences = len(data["premises"])

        self.data = {
                     "premises": torch.ones((self.num_sequences,
                                             self.max_premise_length),
                                            dtype=torch.long) * padding_idx,
                     "hypotheses": torch.ones((self.num_sequences,
                                               self.max_hypothesis_length),
                                              dtype=torch.long) * padding_idx,
                     "labels": torch.tensor(data["labels"], dtype=torch.long)}

        for i, premise in enumerate(data["premises"]):

            end = min(len(premise), self.max_premise_length)
            self.data["premises"][i][:end] = torch.tensor(premise[:end])

            hypothesis = data["hypotheses"][i]
            end = min(len(hypothesis), self.max_hypothesis_length)
            self.data["hypotheses"][i][:end] = torch.tensor(hypothesis[:end])

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, index):
        return {
                "premise": self.data["premises"][index],
                "premise_length": min(self.premises_lengths[index],
                                      self.max_premise_length),
                "hypothesis": self.data["hypotheses"][index],
                "hypothesis_length": min(self.hypotheses_lengths[index],
                                         self.max_hypothesis_length),
                "label": self.data["labels"][index]}

    
    
    
class embDataset(Dataset):
    def __init__(self,
                 data,
                 padding_idx=0,
                 max_premise_length=None,
                ):
        self.premises_lengths = [len(seq) for seq in data["premises"]]
        self.max_premise_length = max_premise_length
        if self.max_premise_length is None:
            self.max_premise_length = max(self.premises_lengths)


        self.num_sequences = len(data["premises"])

        self.data = {"premises": torch.ones((self.num_sequences,
                                             self.max_premise_length),
                                            dtype=torch.long) * padding_idx}

        for i, premise in enumerate(data["premises"]):
            end = min(len(premise), self.max_premise_length)
            self.data["premises"][i][:end] = torch.tensor(premise[:end])

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, index):
        return {
                "premise": self.data["premises"][index],
                "premise_length": min(self.premises_lengths[index],
                                      self.max_premise_length)}