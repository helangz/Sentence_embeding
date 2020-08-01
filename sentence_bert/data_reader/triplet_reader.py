from torch.utils.data import DataLoader,Dataset
from transformers import BertModel,BertTokenizer

import torch
from tqdm import tqdm
import time
import pandas as pd



class TripDataset(Dataset):
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
        s1, s2, s3= df['testa'], df['testb'], df['testc']
 
        for sentence_a, sentence_b, sentence_c in tqdm(list(zip(s1, s2, s3)),desc="加载数据集处理数据集："):
            if len(sentence_a)>self.max_sentence_length:
                sentence_a=sentence_a[:self.max_sentence_length]
            if len(sentence_b)>self.max_sentence_length:
                sentence_b=sentence_b[:self.max_sentence_length]
            if len(sentence_c)>self.max_sentence_length:
                sentence_c=sentence_c[:self.max_sentence_length]
            data_list.append((sentence_a, sentence_b,sentence_c))
        return data_list
    
    def process_data(self):
        process_data_list = []
        for ele in tqdm(self.data_list,desc="处理文本信息："):
            res = self.do_process_data(ele)
            process_data_list.append(res)
        return process_data_list
    def do_process_data(self,params):
 
        res = []
        sentence_a = params[0]
        sentence_b = params[1]
        sentence_c  = params[2]
 
        indextokens_a,segment_id_a,input_mask_a = self.convert_into_indextokens_and_segment_id(sentence_a)
        indextokens_a = torch.tensor(indextokens_a,dtype=torch.long)
        segment_id_a = torch.tensor(segment_id_a,dtype=torch.long)
        input_mask_a = torch.tensor(input_mask_a,dtype=torch.long)
 
        indextokens_b, segment_id_b, input_mask_b = self.convert_into_indextokens_and_segment_id(sentence_b)
        indextokens_b = torch.tensor(indextokens_b, dtype=torch.long)
        segment_id_b = torch.tensor(segment_id_b, dtype=torch.long)
        input_mask_b = torch.tensor(input_mask_b, dtype=torch.long)
 
        indextokens_c, segment_id_c, input_mask_c = self.convert_into_indextokens_and_segment_id(sentence_c)
        indextokens_c = torch.tensor(indextokens_c, dtype=torch.long)
        segment_id_c = torch.tensor(segment_id_c, dtype=torch.long)
        input_mask_c = torch.tensor(input_mask_c, dtype=torch.long)
 
        res.append(indextokens_a)
        res.append(segment_id_a)
        res.append(input_mask_a)
 
 
        res.append(indextokens_b)
        res.append(segment_id_b)
        res.append(input_mask_b)
         
        res.append(indextokens_c)
        res.append(segment_id_c)
        res.append(input_mask_c)
 
        return res
 
    def __getitem__(self, i):
        item = i
        indextokens_a = self.process_data_list[item][0]
        segment_id_a = self.process_data_list[item][1]
        input_mask_a = self.process_data_list[item][2]
        indextokens_b = self.process_data_list[item][3]
        segment_id_b = self.process_data_list[item][4]
        input_mask_b = self.process_data_list[item][5]
        
        indextokens_c = self.process_data_list[item][6]
        segment_id_c = self.process_data_list[item][7]
        input_mask_c = self.process_data_list[item][8] 
 
        return indextokens_a,input_mask_a,indextokens_b,input_mask_b,indextokens_c,input_mask_c
 
    def __len__(self):
        if self.repeat == None:
            data_len = 1000
        else:
            data_len = len(self.process_data_list)
        return data_len