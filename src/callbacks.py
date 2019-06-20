import torch

from catalyst.dl.callbacks import CriterionCallback
from catalyst.dl.core import RunnerState


class EmbeddingsLossCallback(CriterionCallback):
    def __init__(
        self,
        input_key: str = "targets",
        output_key: str = "logits",
        prefix: str = "embeddings_loss",
        criterion_key: str = None,
        loss_key: str = None,
        embeddings_key: str = "embeddings",
        emb_l2_reg: int = -1,
    ):
        super().__init__(
            input_key=input_key,
            output_key=output_key,
            prefix=prefix,
            criterion_key=criterion_key,
            loss_key=loss_key
        )
        self.embeddings_key = embeddings_key
        self.emb_l2_reg = emb_l2_reg

    def _compute_loss(self, state: RunnerState, criterion):
        embeddings = state.output[self.embeddings_key]
        logits = state.output[self.output_key]
        targets = state.input[self.input_key]

        loss = criterion(logits, targets)

        if self.emb_l2_reg > 0:
            emb_loss = torch.mean(torch.norm(embeddings, dim=1))
            loss += emb_loss * self.emb_l2_reg
        return loss
