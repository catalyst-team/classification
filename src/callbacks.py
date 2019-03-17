import torch
from catalyst.dl.callbacks import Callback
from catalyst.dl.state import RunnerState


class EmbeddingsLossCallback(Callback):
    def __init__(
        self,
        emb_l2_reg: int = -1,
        input_key: str = "targets",
        embeddings_key: str = "embeddings",
        logits_key: str = "logits",
    ):
        self.emb_l2_reg = emb_l2_reg
        self.embeddings_key = embeddings_key
        self.logits_key = logits_key
        self.input_key = input_key

    def on_batch_end(self, state: RunnerState):
        embeddings = state.output[self.embeddings_key].float()
        logits = state.output[self.logits_key].float()

        targets = state.input[self.input_key].long()

        loss = state.criterion(logits, targets)

        if self.emb_l2_reg > 0:
            emb_loss = torch.mean(torch.norm(embeddings, dim=1))
            loss += emb_loss * self.emb_l2_reg

        state.loss = loss
