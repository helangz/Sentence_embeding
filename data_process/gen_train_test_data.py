import pandas as pd
import numpy as np
import os

# 提取信息
def concat_data(data,query_list,answer_list,type_list):
    query_set=[]
    i=0
    while i<len(data):
        if data.loc[i,'标准问'] is not np.nan:   
            query_set=[data.loc[i,'标准问']]
            answer_list.append(data.loc[i,'答案'])
            type_list.append(data.loc[i,'问题类型'])     
            i=i+1
            while i<len(data) and data.loc[i,'标准问'] is np.nan:
                query_set.append(data.loc[i,'扩展问'])
                i=i+1
        else:
            i=i+1 
        if query_set:
            query_list.append(query_set)  
    return query_list,answer_list,type_list

#合并信息
def gen_data(query_list,answer_list,type_list):
    ids=[]
    query_all=[]
    # 扩充query
    for i,query in enumerate(query_list):
        ids.extend([i]*len(query))   
        query_all.extend(query)
    query_data=pd.DataFrame()
    query_data['ids']=ids
    query_data['query']=query_all
    other_data=pd.DataFrame()
    other_data['ids']=list(range(len(answer_list)))
    other_data['answer']=answer_list
    other_data['type']=type_list
    data=query_data.merge(other_data,on='ids',how='left')
    return data



def read_data_from_path(path_list):  
    query_list,answer_list,type_list=[],[],[]
    for path in path_list:
        data=pd.read_excel(path)
        query_list,answer_list,type_list=concat_data(data,query_list,answer_list,type_list)
    data=gen_data(query_list,answer_list,type_list)
    # 剔除空值
    data=data.dropna().reset_index(drop=True)
    query_dic={}
    save_index=[]
    #剔除重复的query
    for i in data.index:
        q=data.loc[i,'query']
        ids=data.loc[i,'ids']
        if q not in query_dic:
            query_dic[q]=str(ids)
            save_index.append(i)   
    data=data.iloc[save_index,:]
    return data




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
    for name,datai in data.groupby('ids'):
        if len(datai)>1:
            sentence_pairs.extend(gen_pair(datai))
    pos=pd.DataFrame()
    pos['testa']=[pair[0] for pair in sentence_pairs]
    pos['testb']=[pair[1] for pair in sentence_pairs]
    pos['label']=1
    return pos

#生成负样本
def gen_neg(data,name,datai,num):
    sentences=list(set(datai['query']))
    sentence_pair=[]
    for query in sentences:
        sens=np.random.choice(list(data[data['ids']!=name]['query']),num,replace=False)      
        sentence_pair.extend([[sen,query] for sen in sens])
    return sentence_pair
#neg
def get_neg_data(data,num=5):
    sentence_pairs=[]
    for name,datai in data.groupby('ids'):
        sentence_pairs.extend(gen_neg(data,name,datai,num))
    neg=pd.DataFrame()
    neg['testa']=[pair[0] for pair in sentence_pairs]
    neg['testb']=[pair[1] for pair in sentence_pairs]
    neg['label']=0
    return neg

if __name__=='__main__':
        #数据读取与生成    
    path_list=['../data/raw_data/'+file for file in os.listdir('../data/raw_data')]
    train=read_data_from_path(path)
    test=read_data_from_path(['../data/query_test.csv'])
    train.to_csv('../data/query_data.csv',encoding='utf-8',index=None)
    test.to_csv('../data/query_test.csv',encoding='utf-8',index=None)
    
    ## 生成测试集
    pos_train=get_pos_data(train)
    neg_train=get_neg_data(train,8)
    train_data=pos_train.append(neg_train)  
    pos_test=get_pos_data(test)
    neg_test=get_neg_data(test,3)
    test_data=pos_test.append(neg_test)
    
    train_data.to_csv('./data/train_data.csv',index=None)
    test_data.to_csv('./data/test_data.csv',index=None)







