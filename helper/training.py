import torch
import torch.utils.data
import preprocessing as pp
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score

class HateDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    f1_score_micro = f1_score(labels, preds, average='micro')
    f1_score_macro = f1_score(labels, preds, average='macro')
    f1_score_weighted = f1_score(labels, preds, average='weighted')
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'f1_micro': f1_score_micro,
        'f1_macro': f1_score_macro,
        'f1_weighted': f1_score_weighted,
    }
def compute_metrics_multiclass(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, _, _ = precision_recall_fscore_support(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    f1_score_micro = f1_score(labels, preds, average='micro')
    f1_score_macro = f1_score(labels, preds, average='macro')
    f1_score_weighted = f1_score(labels, preds, average='weighted')
    return {
        'accuracy': acc,
        'precision_macro': precision,
        'recall_macro': recall,
        'f1_micro': f1_score_micro,
        'f1_macro': f1_score_macro,
        'f1_weighted': f1_score_weighted,
    }
