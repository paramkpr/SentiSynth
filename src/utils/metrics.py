import numpy as np  
import torch
import torch
import torch.nn.functional as F
import math
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



def perplexity_metrics(eval_pred):
    """
    Hugging Face passes (logits, labels) – no loss attribute.
    We compute CE‑loss ourselves, then PPL = exp(loss).
    Padding / ignored positions are ‑100 by HF convention.
    """
    # Unpack EvalPrediction → ndarray → torch.Tensor
    logits, labels = eval_pred
    logits  = torch.as_tensor(logits,  dtype=torch.float32)
    labels  = torch.as_tensor(labels,  dtype=torch.long)

    # Shift so that token t predicts t+1 (standard LM training)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Cross‑entropy over non‑ignored tokens
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index = -100,     # HF Trainer uses -100 for padding
        reduction    = "mean"
    )

    return {
        "loss":       loss.item(),
        "perplexity": math.exp(loss.item())
    }
