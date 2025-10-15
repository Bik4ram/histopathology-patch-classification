import torch
from sklearn.metrics import roc_auc_score, f1_score

def compute_metrics(outputs, targets):
    probs = torch.softmax(outputs, dim=1)[:,1].detach().cpu().numpy()
    preds = outputs.argmax(dim=1).detach().cpu().numpy()
    targets = targets.cpu().numpy()
    auc = roc_auc_score(targets, probs)
    f1 = f1_score(targets, preds)
    return auc, f1
