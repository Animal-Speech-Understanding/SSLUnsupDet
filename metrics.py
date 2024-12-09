from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics import Metric

from utils import bounded_euclidean_similarity


class NoiseContrastiveEstimationMetric(Metric):
    def __init__(
        self,
        pred_steps: int = 1,
        pred_offset: int = 0,
        n_negatives: int = 1,
        sim_metric_params: dict[str, Any] | None = None,
        dist_sync_on_step: bool = False,
    ) -> None:
        """
        Initializes the NoiseContrastiveEstimationMetric.

        Args:
            pred_steps (int, optional): Number of prediction steps. Defaults to 1.
            pred_offset (int, optional): Offset for prediction steps. Defaults to 0.
            n_negatives (int, optional): Number of negative samples. Defaults to 1.
            sim_metric_params (Optional[Dict[str, Any]], optional): Parameters for the similarity metric. Defaults to None.
            dist_sync_on_step (bool, optional): Synchronize metric state across processes at each forward step. Defaults to False.
        """
        super(NoiseContrastiveEstimationMetric, self).__init__(dist_sync_on_step=dist_sync_on_step)

        self.pred_steps = list(range(1 + pred_offset, 1 + pred_offset + pred_steps))
        self.pred_offset = pred_offset
        self.n_negatives = n_negatives

        sim_metric_params = sim_metric_params or {"name": "cosine"}
        sim_name = sim_metric_params.get("name", "cosine").lower()
        if sim_name == "cosine":
            self.similarity_metric = lambda z1, z2: F.cosine_similarity(z1, z2, dim=-1)
        elif sim_name == "bounded_euclidean":
            self.similarity_metric = lambda z1, z2: bounded_euclidean_similarity(z1, z2, dim=-1, **sim_metric_params.get("params", {}))
        else:
            raise ValueError(f"Unsupported similarity metric: {sim_metric_params['name']}")

        # Define state variables
        self.add_state("metric_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def similarity(self, z: torch.Tensor, z_shift: torch.Tensor) -> torch.Tensor:
        """
        Computes similarity between z and shifted z.

        Args:
            z (torch.Tensor): Original embeddings.
            z_shift (torch.Tensor): Shifted embeddings.

        Returns:
            torch.Tensor: Similarity scores.
        """
        return self.similarity_metric(z, z_shift)

    def compute_preds(self, z: torch.Tensor) -> dict[int, list[torch.Tensor]]:
        """
        Computes predictions for each prediction step.

        Args:
            z (torch.Tensor): Input embeddings of shape (batch_size, sequence_length, embedding_dim).

        Returns:
            Dict[int, List[torch.Tensor]]: Dictionary mapping prediction step to list of predictions (positives and negatives).
        """
        preds: dict[int, list[torch.Tensor]] = {}
        for t in self.pred_steps:
            positive_pred = self.similarity(z[:, :-t], z[:, t:])
            preds[t] = [positive_pred]
            for _ in range(self.n_negatives):
                # Shuffle the future embeddings within the batch to create negatives
                time_reorder = torch.arange(positive_pred.shape[1], device=z.device)
                negative_pred = self.similarity(z[:, :-t], z[:, time_reorder])
                preds[t].append(negative_pred)
        return preds

    def metric(self, preds: dict[int, list[torch.Tensor]]) -> torch.Tensor:
        """
        Computes the metric from predictions.

        Args:
            preds (Dict[int, List[torch.Tensor]]): Predictions for each step.

        Returns:
            torch.Tensor: Computed metric.
        """
        m = 0.0
        for t, t_preds in preds.items():
            # Stack predictions: first is positive, rest are negatives
            out = torch.stack(t_preds, dim=-1)  # Shape: (batch_size, seq_length - t, n_negatives + 1)
            out = F.log_softmax(out, dim=-1)
            # The first entry is the positive sample
            positive_log_prob = out[..., 0]
            m += -positive_log_prob.mean()
        return m

    def update(self, z: torch.Tensor) -> None:
        """
        Updates the internal state with a new batch of embeddings.

        Args:
            z (torch.Tensor): Input embeddings of shape (batch_size, sequence_length, embedding_dim).
        """
        preds = self.compute_preds(z)
        loss = self.metric(preds)
        self.metric_sum += loss
        self.count += z.size(0)

    def compute(self) -> torch.Tensor:
        """
        Computes the final metric value.

        Returns:
            torch.Tensor: Final metric value.
        """
        if self.count == 0:
            return torch.tensor(0.0, device=self.metric_sum.device)
        return self.metric_sum / self.count


class PRF1Metric(object):
    def __init__(self,
                 tolerance):
        self.tolerance = tolerance
        
    def recall(self, true_positives, false_negatives):
        return true_positives / (true_positives + false_negatives)
    
    def precision(self, true_positives, false_positives):
        return true_positives / (true_positives + false_positives)
    
    def f1(self, true_positives, false_negatives, false_positives):
        return 2 * true_positives / (2 * true_positives + false_positives + false_negatives)
    
    def prob_detection(self, true_positives, trues):
        return true_positives / len(trues)
    
    def prob_false_alarm(self, true_positives, false_positives):
        return false_positives / (false_positives + true_positives)
    
    def classify_predictions(self, trues, preds):
        true_positives = 0
        false_negatives = 0
        for t in trues:
            condition = np.logical_and(t - self.tolerance <= preds, preds <= t + self.tolerance)
            if len(preds[condition]) > 0:
                true_positives += 1
            else:
                false_negatives += 1
                
        false_positives = 0
        for p in preds:
            condition = np.logical_and(p - self.tolerance <= trues, trues <= p + self.tolerance)
            if len(trues[condition]) > 0:
                pass
            else:
                false_positives += 1
                
        return true_positives, false_negatives, false_positives
    
    def r_value(self, R, P):
        os = R / P - 1
        r1 = np.sqrt((1 - R) ** 2 + os ** 2)
        r2 = (-os + R - 1) / (np.sqrt(2))
        r_val = 1 - (np.abs(r1) + np.abs(r2)) / 2
        return r_val
    
    def compute_metrics(self, trues, preds):
        TP, FN, FP = self.classify_predictions(trues, preds)
        
        R = self.recall(TP, FN)
        P = self.precision(TP, FP)
        F1 = self.f1(TP, FN, FP)
        P_det = self.prob_detection(TP, trues)
        P_FA = self.prob_false_alarm(TP, FP)
        R_Val = self.r_value(R, P)
        
        metrics = {'Recall':R,
                   'Precision':P,
                   'F1':F1,
                   'R-Value':R_Val,
                   'P_detection':P_det,
                   'P_false_alarm':P_FA}
        return metrics
