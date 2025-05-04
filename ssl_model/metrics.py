import torch
import torchmetrics
from ssl_model.utils import get_similarity_fn


class NoiseContrastiveEstimationAccuracy(torchmetrics.Metric):
    """
    Computes classification accuracy for NCE task: picking the positive among negatives.
    """
    is_differentiable = False
    higher_is_better = True
    full_state_update = False

    def __init__(
        self,
        pred_steps: int = 1,
        pred_offset: int = 0,
        n_negatives: int = 1,
        sim_metric_params=None,
        dist_sync_on_step: bool = False,
    ):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        if sim_metric_params is None:
            sim_metric_params = {'name': 'cosine'}
        steps = list(range(1 + pred_offset, 1 + pred_offset + pred_steps))
        self.pred_steps = steps
        self.pred_offset = pred_offset
        self.n_negatives = n_negatives
        self.sim_fn = get_similarity_fn(sim_metric_params)

        # states
        self.add_state('correct', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, z: torch.Tensor):
        batch_size, seq_len, _ = z.shape

        for t in self.pred_steps:
            pos = self.sim_fn(z[:, :-t], z[:, t:])
            preds = [pos]
            for _ in range(self.n_negatives):
                neg = self.sim_fn(z[:, :-t], z)
                preds.append(neg)

            logits = torch.stack(preds, dim=-1)
            # predict index of max similarity
            preds_idx = torch.argmax(logits, dim=-1)
            # correct if idx==0
            self.correct += (preds_idx == 0).sum()
            self.total += preds_idx.numel()

    def compute(self):
        """Returns overall accuracy."""
        return self.correct.float() / self.total
