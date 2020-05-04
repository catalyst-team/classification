import torch
import torch.nn as nn


class EmbeddingsNormLoss(nn.Module):
    def forward(self, embeddings, *args):
        return torch.mean(torch.norm(embeddings, dim=1))
