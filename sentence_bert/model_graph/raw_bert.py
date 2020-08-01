class raw_BertModel(nn.Module):
    def __init__(self):
        super(raw_BertModel,self).__init__()
 
        self.bert = BertModel.from_pretrained(data_root).cuda()
        for param in self.bert.parameters():
            param.requires_grad = False 
        
    def forward(self, indextokens_a,input_mask_a,indextokens_b,input_mask_b):
        embedding_a = self.bert(indextokens_a,input_mask_a)[0]
        embedding_b = self.bert(indextokens_b,input_mask_b)[0]
        embedding_a = torch.mean(embedding_a,1)
        embedding_b = torch.mean(embedding_b,1)
        return embedding_a,embedding_b