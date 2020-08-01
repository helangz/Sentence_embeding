import pandas as pd
import numpy as np
from util.process import data_process
import os
from tqdm import tqdm
from util.query2vec import trans_vec
from util.query2vec import get_query_matrix
import gensim

#数据读取
data=pd.read_csv('../data/test_data2.csv')
import json
with open('../data/query_index.json','r') as f:
    query_dic=json.load(f)
query_list=[key for key in query_dic]


## 读取模型
model= gensim.models.Word2Vec.load('./model/w2vmodel.model')
model_name='wv'
path=f'./data/query_matrix_{model_name}.npy'
embedding_dim=256

if os.path.exists(path):
    query_matrix=np.load(path)
else:
    query_matrix=get_query_matrix(model,list(map(data_process,query_list)),embedding_dim)   
    np.save(path,query_matrix)

def embed(x,model,embedding_dim):
    x=data_process(x)
    x=trans_vec(x,model,embedding_dim)
    return x


##测试annoy
from annoy import AnnoyIndex
import random  
#euclidean “angular”, “euclidean”, “manhattan”, “hamming”, or “dot”.
t = AnnoyIndex(query_matrix.shape[1],metric="euclidean")  
for i in range(query_matrix.shape[0]):
    t.add_item(i, query_matrix[i,:])
t.build(10)

def get_count_list(data,query_matrix,query_dic,query_list,n,t):
    count_list=[]
    for i in data.index:
        count=0
        text=data.loc[i,'query']
        ids=data.loc[i,'ids']
        index=t.get_nns_by_vector(embed(text,model,embedding_dim),n)
        ids_list=[query_dic[query_list[i]] for i in index]
        for j in ids_list:
            if j==str(ids):
                count+=1
        count_list.append(count)
    return count_list
count_list5=get_count_list(data,query_matrix,query_dic,query_list,6,t)
print(sum(count_list5)/len(count_list5)-1)
print((sum(count_list5)/len(count_list5)-1)/5)