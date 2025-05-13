import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


def compute_metrics(p):
    """Computes metrics for HF Trainer."""
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary') # Assuming binary
    acc = accuracy_score(labels, preds)
    
    # Calculate confusion matrix metrics
    true_positives = np.sum((preds == 1) & (labels == 1))
    false_positives = np.sum((preds == 1) & (labels == 0))
    true_negatives = np.sum((preds == 0) & (labels == 0))
    false_negatives = np.sum((preds == 0) & (labels == 1))
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'true_positives': int(true_positives),
        'false_positives': int(false_positives), 
        'true_negatives': int(true_negatives),
        'false_negatives': int(false_negatives)
    }
