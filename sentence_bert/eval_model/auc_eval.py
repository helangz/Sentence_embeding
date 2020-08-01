import torch
from sklearn import metrics
def dev(model,dev_loader):
    model.eval()
    with torch.no_grad():
        y_pred=[]
        y_true=[]
        for step, (
                indextokens_a, input_mask_a, indextokens_b,input_mask_b, label) in tqdm(enumerate(
            dev_loader),desc='Dev Itreation:'):
            indextokens_a,input_mask_a, indextokens_b, input_mask_b,  = indextokens_a.to(device), input_mask_a.to(
                device), indextokens_b.to(device), input_mask_b.to(device)
            emb_a=model.bert(indextokens_a, input_mask_a)[0]
            emb_b=model.bert(indextokens_b, input_mask_b)[0]
            emb_a=emb_a.mean(1)
            emb_b=emb_b.mean(1)
            cos=torch.cosine_similarity(emb_a, emb_b, dim=1).cpu().detach().numpy()
            y_pred.extend(list(cos))
            y_true.extend(list(label.numpy()))            
        auc = metrics.roc_auc_score(y_true,y_pred,average='macro')
        return auc
