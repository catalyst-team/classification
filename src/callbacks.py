import torch
from catalyst.dl.callbacks import Callback


class EmbeddingsLossCallback(Callback):
    def __init__(self, emb_l2_reg=-1):
        self.emb_l2_reg = emb_l2_reg

    def on_batch_end(self, state):
        embeddings = state.output["embeddings"]
        logits = state.output["logits"]

        loss = state.criterion(logits.float(), state.input["targets"].long())

        if self.emb_l2_reg > 0:
            emb_loss = torch.mean(torch.norm(embeddings.float(), dim=1))
            loss += emb_loss * self.emb_l2_reg

        state.loss = loss
