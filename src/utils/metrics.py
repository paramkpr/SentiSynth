import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


def compute_metrics(p):
    """Computes metrics for HF Trainer."""
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary') # Assuming binary
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }
