from collections import defaultdict
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import bounded_euclidean_similarity


class NoiseContrastiveEstimationLoss(nn.Module):
    def __init__(
        self,
        pred_steps: int = 1,
        pred_offset: int = 0,
        n_negatives: int = 1,
        sim_metric_params: dict[str, Any] = {"name": "cosine"},
    ) -> None:
        """
        Initializes the NCE loss.

        Args:
            pred_steps (int, optional): Number of prediction steps. Defaults to 1.
            pred_offset (int, optional): Offset for prediction steps. Defaults to 0.
            n_negatives (int, optional): Number of negative samples. Defaults to 1.
            sim_metric_params (Dict[str, Any], optional): Parameters for similarity metric. Defaults to {'name': 'cosine'}.
        """
        super(NoiseContrastiveEstimationLoss, self).__init__()
        self.pred_steps = list(range(1 + pred_offset, 1 + pred_offset + pred_steps))
        self.pred_offset = pred_offset
        self.n_negatives = n_negatives

        sim_name = sim_metric_params.get("name", "cosine")
        if sim_name == "cosine":
            self.similarity_metric: Callable[
                [torch.Tensor, torch.Tensor], torch.Tensor
            ] = lambda z1, z2: F.cosine_similarity(z1, z2, dim=-1)
        elif sim_name == "bounded_euclidean":
            self.similarity_metric = lambda z1, z2: bounded_euclidean_similarity(
                z1, z2, dim=-1, **sim_metric_params.get("params", {})
            )
        else:
            raise ValueError(
                f"Unsupported similarity metric: {sim_metric_params['name']}"
            )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Computes the NCE loss.

        Args:
            z (torch.Tensor): Input embeddings of shape (batch_size, sequence_length, embedding_dim).

        Returns:
            torch.Tensor: Computed loss.
        """
        preds = self.compute_preds(z)
        loss = self.compute_loss(preds)
        return loss

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
        preds = defaultdict(list)

        for t in self.pred_steps:
            # Positive predictions
            z_current = z[:, :-t]  # Shape: (batch_size, seq_length - t, embedding_dim)
            z_future = z[:, t:]  # Shape: (batch_size, seq_length - t, embedding_dim)
            positive_pred = self.similarity(
                z_current, z_future
            )  # Shape: (batch_size, seq_length - t)
            preds[t].append(positive_pred)

            # Negative predictions
            for _ in range(self.n_negatives):
                # Shuffle the future embeddings within the batch to create negatives
                indices = torch.randperm(positive_pred.shape[1])
                z_negative = z[
                    :, indices
                ]  # Shape: (batch_size, seq_length - t, embedding_dim)
                negative_pred = self.similarity(z_current, z_negative)
                preds[t].append(negative_pred)

        return preds

    def loss(self, preds: dict[int, list[torch.Tensor]]) -> torch.Tensor:
        """
        Computes the NCE loss from predictions.

        Args:
            preds (Dict[int, List[torch.Tensor]]): Predictions for each step.

        Returns:
            torch.Tensor: Computed loss.
        """
        total_loss = 0.0
        for t, t_preds in preds.items():
            # Stack predictions: first is positive, rest are negatives
            out = torch.stack(
                t_preds, dim=-1
            )  # Shape: (batch_size, seq_length - t, n_negatives + 1)
            out = F.log_softmax(out, dim=-1)
            # The first entry is the positive sample
            positive_log_prob = out[..., 0]
            total_loss += -positive_log_prob.mean()
        return total_loss


class InfoNCELoss(nn.Module):
    def __init__(
        self,
        pred_steps: int = 1,
        pred_offset: int = 0,
        n_negatives: int = 1,
        temperature: float = 0.07,
        sim_metric_params: dict[str, Any] = {"name": "cosine"},
    ) -> None:
        """
        Initializes the InfoNCE loss.

        Args:
            pred_steps (int, optional): Number of prediction steps. Defaults to 1.
            pred_offset (int, optional): Offset for prediction steps. Defaults to 0.
            n_negatives (int, optional): Number of negative samples per positive. Defaults to 1.
            temperature (float, optional): Temperature scaling factor. Defaults to 0.07.
            sim_metric_params (Dict[str, Any], optional): Parameters for similarity metric. Defaults to {'name': 'cosine'}.
        """
        super(InfoNCELoss, self).__init__()
        self.pred_steps = list(range(1 + pred_offset, 1 + pred_offset + pred_steps))
        self.pred_offset = pred_offset
        self.n_negatives = n_negatives
        self.temperature = temperature

        sim_name = sim_metric_params.get("name", "cosine")
        if sim_name == "cosine":
            self.similarity_metric: Callable[
                [torch.Tensor, torch.Tensor], torch.Tensor
            ] = lambda z1, z2: F.cosine_similarity(z1, z2, dim=-1)
        elif sim_name == "bounded_euclidean":
            self.similarity_metric = lambda z1, z2: bounded_euclidean_similarity(
                z1, z2, dim=-1, **sim_metric_params.get("params", {})
            )
        else:
            raise ValueError(
                f"Unsupported similarity metric: {sim_metric_params['name']}"
            )

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Computes the InfoNCE loss.

        Args:
            z (torch.Tensor): Input embeddings of shape (batch_size, sequence_length, embedding_dim).

        Returns:
            torch.Tensor: Computed loss.
        """
        preds = self.compute_preds(z)
        loss = self.compute_loss(preds)
        return loss

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

    def compute_preds(self, z: torch.Tensor) -> dict[int, torch.Tensor]:
        """
        Computes predictions for each prediction step.

        Args:
            z (torch.Tensor): Input embeddings of shape (batch_size, sequence_length, embedding_dim).

        Returns:
            Dict[int, torch.Tensor]: Dictionary mapping prediction step to logits tensor of shape (batch_size * (seq_length - t), 1 + n_negatives).
        """
        preds = defaultdict(list)
        batch_size, seq_length, _ = z.size()

        for t in self.pred_steps:
            # Define the valid range for current and future embeddings
            z_current = z[:, :-t, :]  # Shape: (batch_size, seq_length - t, embedding_dim)
            z_future = z[:, t:, :]    # Shape: (batch_size, seq_length - t, embedding_dim)

            # Compute positive similarities
            positive_sim = self.similarity(z_current, z_future)  # Shape: (batch_size, seq_length - t)

            # Expand dimensions for concatenation
            positive_sim = positive_sim.unsqueeze(-1)  # Shape: (batch_size, seq_length - t, 1)

            # Negative sampling: shuffle the future embeddings within the batch
            z_negatives = []
            for _ in range(self.n_negatives):
                indices = torch.randperm(batch_size * (seq_length - t)).to(z.device)
                z_neg = z_future.contiguous().view(-1, z.size(-1))[indices].view(batch_size, seq_length - t, -1)
                neg_sim = self.similarity(z_current, z_neg)  # Shape: (batch_size, seq_length - t)
                z_negatives.append(neg_sim.unsqueeze(-1))  # Shape: (batch_size, seq_length - t, 1)

            # Concatenate positive and negative similarities
            logits = torch.cat([positive_sim] + z_negatives, dim=-1)  # Shape: (batch_size, seq_length - t, 1 + n_negatives)

            # Scale by temperature
            logits = logits / self.temperature

            # Reshape for loss computation
            logits = logits.view(-1, 1 + self.n_negatives)  # Shape: (batch_size * (seq_length - t), 1 + n_negatives)

            preds[t] = logits

        return preds

    def compute_loss(self, preds: dict[int, torch.Tensor]) -> torch.Tensor:
        """
        Computes the InfoNCE loss from predictions.

        Args:
            preds (Dict[int, torch.Tensor]): Predictions for each step.

        Returns:
            torch.Tensor: Computed loss.
        """
        total_loss = 0.0
        for t, logits in preds.items():
            # Create target labels (0 for positives)
            targets = torch.zeros(logits.size(0), dtype=torch.long).to(logits.device)

            # Compute cross-entropy loss
            loss = self.loss_fn(logits, targets)
            total_loss += loss

        # Average over prediction steps
        total_loss = total_loss / len(self.pred_steps)
        return total_loss
