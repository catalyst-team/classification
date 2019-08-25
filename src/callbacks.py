import torch

from catalyst.dl import RunnerState, CriterionCallback


class EmbeddingsCriterionCallback(CriterionCallback):
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


class AECriterionCallback(CriterionCallback):
    def __init__(
        self,
        input_key: str = "targets",
        output_key: str = "logits",
        prefix: str = "embeddings_loss",
        criterion_key: str = None,
        loss_key: str = None,
        loc_key: str = "embeddings_loc",
        log_scale_key: str = "embeddings_log_scale",
        kld_regularization: float = 1,
    ):
        super().__init__(
            input_key=input_key,
            output_key=output_key,
            prefix=prefix,
            criterion_key=criterion_key,
            loss_key=loss_key
        )
        self.loc_key = loc_key
        self.log_scale_key = log_scale_key
        self.kld_regularization = kld_regularization

    def _compute_loss(self, state: RunnerState, criterion):
        logits = state.output[self.output_key]
        targets = state.input[self.input_key]
        embeddings_loc = state.output[self.loc_key]
        embeddings_log_scale = state.output[self.log_scale_key]

        loss = criterion(logits, targets)

        if embeddings_loc is not None and embeddings_log_scale is not None:
            mu = embeddings_loc
            logvar = embeddings_log_scale
            kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss += kld * self.kld_regularization
        return loss
