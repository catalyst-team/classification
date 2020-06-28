import torch
import torch.nn as nn


class EmbeddingsNormLoss(nn.Module):
    """Embeddings loss."""

    def forward(self, embeddings, *args) -> torch.Tensor:
        """Forward propagation method for the :class:`EmbeddingsNormLoss` loss.

        Args:
            embeddings (torch.Tensor): bash of embeddings
            *args: other args

        Returns:
            torch.Tensor: loss
        """
        return torch.mean(torch.norm(embeddings, dim=1))
