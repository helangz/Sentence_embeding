import pandas as pd
import numpy as np
import os
from tqdm import tqdm


# 生成正样本
def gen_pair(datai):
    sentences=list(set(datai['query']))
    sentence_pair=[]
    length=len(sentences)
    for i in range(length-1):
        for j in range(i+1,length):
            sentence_pair.append([sentences[i],sentences[j]])
            
    return sentence_pair

def get_pos_data(data):
    sentence_pairs=[]
    ids=[]
    for name,datai in data.groupby('ids'):
        if len(datai)>1:
            pairs=gen_pair(datai)
            sentence_pairs.extend(pairs)
            ids.extend([name]*len(pairs))
    pos=pd.DataFrame()
    pos['testa']=[pair[0] for pair in sentence_pairs]
    pos['testb']=[pair[1] for pair in sentence_pairs]
    pos['ids']=ids
    return pos

#生成负样本
def gen_neg(query_data,name,datai,num):

    sentence_pair=[]
    count=0
    for testa,testb in zip(datai['testa'],datai['testb']):
        try:
            sens=np.random.choice(list(query_data[query_data['ids']!=name]['query']),num,replace=False)      
            sentence_pair.extend([[testa,testb,sen] for sen in sens])
        except:
            sens=np.random.choice(list(query_data[query_data['ids']!=name]['query']),num,replace=True)      
            sentence_pair.extend([[testa,testb,sen] for sen in sens])
            count=count+1
            print(count)
    return sentence_pair
#neg
def get_neg_data(data,query_data,num=3):
    sentence_pairs=[]
    for name,datai in tqdm(data.groupby('ids')):
        sentence_pairs.extend(gen_neg(query_data,name,datai,num))
    neg=pd.DataFrame()
    neg['testa']=[pair[0] for pair in sentence_pairs]
    neg['testb']=[pair[1] for pair in sentence_pairs]
    neg['testc']=[pair[2] for pair in sentence_pairs]
    return neg

if __name__=='__main__':
    train=pd.read_csv('../data/query_data.csv')
    pos_pair=get_pos_data(train)
    trip_data=get_neg_data(pos_pair,train)
    trip_data.to_csv('../data/trip_data.csv',index=None)




