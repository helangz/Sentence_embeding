from transformers import BertModel,BertTokenizer
import torch
class Bert_Embedding(object):
    def __init__(self,model,
                 dim=768,
                 data_root='./bert_model/bert_wwm2/',
                max_sentence_length=32):
        self.model=model
        self.dim=dim
        self.data_root=data_root
        self.max_sentence_length = max_sentence_length
        self.tokenizer = BertTokenizer.from_pretrained(self.data_root)
        
    
    #将数据转换为格式化输入   
    def convert(self,text):
        if len(text)>self.max_sentence_length:
            text=text[:self.max_sentence_length]
        tokeniz_text = self.tokenizer.tokenize(text)
        indextokens = self.tokenizer.convert_tokens_to_ids(tokeniz_text)
        input_mask = [1] * len(indextokens)
        pad_indextokens = [0]*(self.max_sentence_length-len(indextokens))
        indextokens.extend(pad_indextokens)
        input_mask_pad = [0]*(self.max_sentence_length-len(input_mask))
        input_mask.extend(input_mask_pad)
        #更改为torch 格式
        indextokens = torch.tensor(indextokens,dtype=torch.long)
        input_mask = torch.tensor(input_mask,dtype=torch.long)
        indextokens,input_mask=indextokens.to('cuda'),input_mask.to('cuda')
        return indextokens.unsqueeze(dim=0),input_mask.unsqueeze(dim=0)
    
    def embed_sentence(self,text):
        #batch 输入
        indextokens,input_mask=self.convert(text)
        emb=self.model.bert(indextokens,input_mask)[0]
        emb=emb.squeeze(0).mean(0)
        emb=emb.cpu().detach().numpy()
        return emb 
    
    def get_query_matrix(self,query_list):
        query_matrix=np.zeros((len(query_list),self.dim))
        for i,sen in tqdm(enumerate(query_list)):    
            query_matrix[i:]=self.embeded(sen)
        return query_matrix