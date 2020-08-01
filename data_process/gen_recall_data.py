import pandas as pd
import json
def gen_recall_data():
    ## 原始数据
    data_test=pd.read_csv('./data/query_test.csv',encoding='utf-8')
    ## 准备扩充的数据
    data=pd.read_csv('./data/query_data.csv',encoding='utf-8')

    query_dic={}
    for i in data_test.index:
        q=data_test.loc[i,'query']
        ids=data_test.loc[i,'ids']
        if q not in query_dic:
            query_dic[q]=str(ids)

    ## 从data 中添加数据作为噪声
    for i in data.index:
        q=data.loc[i,'query']
        ids=9999
        if q not in query_dic:
            query_dic[q]=str(ids)

    ## 保存json
    with open('./data/query_index.json','w') as f:
        json.dump(query_dic,f)

    group_data=[]
    for name,datai in data_test.groupby('ids'):
        if len(datai)>=5:
            group_data.append(datai)
    test_data=pd.concat(group_data)
    test_data[['ids','query']].to_csv('./data/test_data2.csv',index=None)

if __name__=='__main__':
    gen_recall_data()
    
