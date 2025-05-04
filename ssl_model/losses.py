from typing import Any, Dict, List
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.loss import _Loss

from ssl_model.utils import get_similarity_fn


class NoiseContrastiveEstimationLoss(_Loss):
    """
    NCE loss: for each prediction step, push positive samples ahead of negatives.

    Args:
        pred_steps: number of future steps to predict
        pred_offset: starting offset for predictions
        n_negatives: number of negatives per positive
        sim_metric_params: dict specifying the similarity metric
        reduction: 'mean' | 'sum' (averaging over steps)
    """
    __constants__ = ['pred_steps', 'n_negatives']

    def __init__(
        self,
        pred_steps: int = 1,
        pred_offset: int = 0,
        n_negatives: int = 1,
        sim_metric_params=None,
        reduction: str = 'mean',
    ) -> None:
        super().__init__(size_average=None, reduce=None, reduction=reduction)
        # compute list of time offsets
        if sim_metric_params is None:
            sim_metric_params = {'name': 'cosine'}
        self.pred_steps: List[int] = list(
            range(1 + pred_offset, 1 + pred_offset + pred_steps)
        )
        self.n_negatives = n_negatives
        self.sim_fn = get_similarity_fn(sim_metric_params)

    def forward(self, z: Tensor) -> Tensor:
        """
        Compute NCE loss given embeddings z of shape (batch, seq_len, embed_dim).

        Returns:
            loss: scalar Tensor
        """
        device = z.device
        total = torch.tensor(0.0, device=device)
        for t in self.pred_steps:
            # positive similarities
            pos = self.sim_fn(z[:, :-t], z[:, t:])  # [batch, seq_len - t]
            preds = [pos]
            # negatives: shuffle temporal dimension
            for _ in range(self.n_negatives):
                seqlen = pos.size(1)
                idx = torch.randperm(seqlen, device=device)
                neg = self.sim_fn(z[:, :-t], z[:, idx])
                preds.append(neg)

            logits = torch.stack(preds, dim=-1)  # [..., 1 + n_negatives]
            logp = F.log_softmax(logits, dim=-1)[..., 0]
            total = total + -logp.mean()

        # reduction across prediction steps
        if self.reduction == 'mean':
            return total / len(self.pred_steps)
        return total

# Example usage:
# loss_fn = NoiseContrastiveEstimationLoss(pred_steps=3, n_negatives=5, reduction='mean')
# loss = loss_fn(embeddings)  # embeddings: [batch, seq_len, dim]
