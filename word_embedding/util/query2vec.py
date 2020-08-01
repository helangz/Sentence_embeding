import numpy as np
from tqdm import tqdm


def trans_vec(sentences,model,embedding_dim):
    #sentences=data_process(query)
    vec=np.zeros(embedding_dim)
    count=0
    for word in sentences:
        if model.wv.__contains__(word):
            vec+=model.wv.__getitem__(word)
            count+=1
    if count:
        vec=vec/count
    return vec 

def get_query_matrix(model,query_list,embedding_dim):
    query_matrix=np.zeros((len(query_list),embedding_dim))
    for i,sen in tqdm(enumerate(query_list)):    
        count=0   
        vec=np.zeros(embedding_dim)
        for word in sen:
            if model.wv.__contains__(word):
                vec+=model.wv.__getitem__(word)
                count+=1
        if count:
            query_matrix[i:]=vec/count
    return query_matrix


#fasttext格式的数据读取
def trans_vec_ft(sentences,model,embedding_dim):
    vec=np.zeros(embedding_dim)
    count=0
    for word in sentences:
        if word in model:
            vec+=model[word]
            count+=1
    if count:
        vec=vec/count
    return vec

def get_query_matrix_ft(model,query_list,embedding_dim):
    query_matrix=np.zeros((len(query_list),embedding_dim))
    for i,sen in tqdm(enumerate(query_list)):       
        count=0        
        vec=np.zeros(embedding_dim)
        if model_type=='wv':
            for word in sen:
                if model.wv.__contains__(word):
                    vec+=model.wv.__getitem__(word)
                    count+=1
        else:           
            for word in sen:
                if word in model:
                    vec+=model[word]
                    count+=1
        if count:
            query_matrix[i:]=vec/count
    return query_matrix



