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

import faiss
query_matrix=query_matrix.astype('float32')


def get_count(faiss_index):
    count_list=[]
    for i in data.index:
        count=0
        text=data.loc[i,'query']
        ids=data.loc[i,'ids']
        _,index=faiss_index.search(np.expand_dims(embed(text,model,embedding_dim),0).astype('float32'), 6) 
        index=index[0]
        ids_list=[query_dic[query_list[i]] for i in index]
        for j in ids_list:
            if str(int(j))==str(ids):
                count+=1
        count_list.append(count)
    return (sum(count_list)/len(count_list)-1)/5

#精确查找
dim = embedding_dim
index = faiss.IndexFlatL2(embedding_dim) 
index.train(query_matrix)
index.add(query_matrix.astype('float32'))
print(get_count(index))

## 减少内存
dim = embedding_dim # 向量维度
nlist = 100  #聚类中心的个数
m = 8        # 压缩成8bits
quantizer = faiss.IndexFlatIP(dim) # 定义量化器  
index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, 8)  # 8 specifies that each sub-vector is encoded as 8 bits
index.nprobe = 10 #查找聚类中心的个数，默认为1个，若nprobe=nlist则等同于精确查找
index.train(query_matrix) #需要训练
index.add(query_matrix)
print(get_count(index))

#快速搜索
dim = embedding_dim # 向量维度
nlist = 100  #聚类中心的个数
quantizer = faiss.IndexFlatIP(dim)  # 定义量化器
index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_L2) #也可采用向量內积               
index.nprobe = 10 #查找聚类中心的个数，默认为1个，若nprobe=nlist则等同于精确查找
index.train(query_matrix)
index.add(query_matrix)
print(get_count(index))

##降维搜索
dim = embedding_dim # 向量维度
index = faiss.index_factory(dim,"PCAR32,IVF100,SQ8")
index.train(query_matrix) #需要训练
index.add(query_matrix) # 添加训练时的样本
print(get_count(index))



