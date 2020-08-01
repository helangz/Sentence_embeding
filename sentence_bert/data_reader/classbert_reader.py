from torch.utils.data import DataLoader,Dataset
from transformers import BertModel,BertTokenizer
import torch
from tqdm import tqdm
import time
import pandas as pd


class classDataset(Dataset):
    def __init__(self,filename,data_root='./bert_model/bert_wwm2/',repeat=1):      
        ## 更改最大句子的长度
        self.max_sentence_length = 32
        self.repeat = repeat
        self.data_root=data_root
        self.tokenizer = BertTokenizer.from_pretrained(self.data_root)
        self.data_list = self.read_file(filename)
        self.len = len(self.data_list)
        self.process_data_list = self.process_data() 
    def convert_into_indextokens_and_segment_id(self,text):
        tokeniz_text = self.tokenizer.tokenize(text)
        indextokens = self.tokenizer.convert_tokens_to_ids(tokeniz_text)
        input_mask = [1] * len(indextokens)
        pad_indextokens = [0]*(self.max_sentence_length-len(indextokens))
        indextokens.extend(pad_indextokens)
        input_mask_pad = [0]*(self.max_sentence_length-len(input_mask))
        input_mask.extend(input_mask_pad)
        segment_id = [0]*self.max_sentence_length
        return indextokens,segment_id,input_mask
    def read_file(self,filename):
        data_list = []
        df = pd.read_csv(filename)  # tsv文件
        s1,labels = df['query'],df['label']
 
        for sentence,label in tqdm(list(zip(s1, labels)),desc="加载数据集处理数据集："):
            if len(sentence)>self.max_sentence_length:
                sentence=sentence[:self.max_sentence_length]
            data_list.append((sentence,label))
        return data_list 
    def process_data(self):
        process_data_list = []
        for ele in tqdm(self.data_list,desc="处理文本信息："):
            res = self.do_process_data(ele)
            process_data_list.append(res)
        return process_data_list 
    def do_process_data(self,params):
        res = []
        sentence = params[0]
        label = params[1]
        indextokens,segment_id,input_mask = self.convert_into_indextokens_and_segment_id(sentence)
        indextokens = torch.tensor(indextokens,dtype=torch.long)
        segment_id = torch.tensor(segment_id,dtype=torch.long)
        input_mask = torch.tensor(input_mask,dtype=torch.long) 
        label = torch.tensor(int(label))
        res.append(indextokens)
        res.append(segment_id)
        res.append(input_mask) 
        res.append(label)
        return res
 
    def __getitem__(self, i):
        item = i
        indextokens = self.process_data_list[item][0]
        segment_id = self.process_data_list[item][1]
        input_mask = self.process_data_list[item][2]
        label = self.process_data_list[item][3]
        return indextokens,input_mask,label
 
    def __len__(self):
        if self.repeat == None:
            data_len = 1000
        else:
            data_len = len(self.process_data_list)
        return  data_len