# src/trainers/weighted_soft_trainer.py
import torch, torch.nn.functional as F
from torch import nn
from transformers import Trainer

class WeightedSoftTrainer(Trainer):
    def __init__(self, *args, alpha=0.5, temperature=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.T     = temperature

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        **_ignored,                    # absorbs num_items_in_batch and future args
    ):
        labels  = inputs.pop("labels")
        weights = inputs.pop("weights", None)
        soft    = inputs.pop("soft_labels", None)
        inputs.pop("is_synth", None)

        outputs = model(**inputs)
        logits  = outputs.logits

        ce = F.cross_entropy(logits, labels, reduction="none")

        if soft is not None:
            soft   = torch.tensor(soft, device=logits.device)
            kl     = F.kl_div(
                F.log_softmax(logits / self.T, dim=-1),
                F.softmax(soft / self.T, dim=-1),
                reduction="batchmean",
            ) * (self.T ** 2)
            loss = self.alpha * ce + (1 - self.alpha) * kl
        else:
            loss = ce

        if weights is not None:
            loss = loss * torch.tensor(weights, device=logits.device)

        loss = loss.mean()
        return (loss, outputs) if return_outputs else loss

