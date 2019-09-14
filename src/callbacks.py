import torch

from catalyst.dl import RunnerState, CriterionCallback


class EmbeddingsCriterionCallback(CriterionCallback):
    def __init__(
        self,
        input_key: str = "targets",
        output_key: str = "logits",
        prefix: str = "embeddings_loss",
        criterion_key: str = None,
        embeddings_key: str = "embeddings",
        emb_l2_reg: int = -1,
    ):
        super().__init__(
            input_key=input_key,
            output_key=output_key,
            prefix=prefix,
            criterion_key=criterion_key,
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
        loc_key: str = "embeddings_loc",
        log_scale_key: str = "embeddings_log_scale",
        kld_regularization: float = 1,
        logprob_key: str = "embeddings_logprob",
        logprob_regularization: float = 1,
    ):
        super().__init__(
            input_key=input_key,
            output_key=output_key,
            prefix=prefix,
            criterion_key=criterion_key,
        )
        self.loc_key = loc_key
        self.log_scale_key = log_scale_key
        self.kld_regularization = kld_regularization
        self.logprob_key = logprob_key
        self.logprob_regularization = logprob_regularization

    def _compute_loss(self, state: RunnerState, criterion):
        logits = state.output[self.output_key]
        targets = state.input[self.input_key]
        loss = criterion(logits, targets)

        loc = state.output[self.loc_key]
        log_scale = state.output[self.log_scale_key]
        if loc is not None and log_scale is not None:
            kld = -0.5 * torch.mean(
                1 + log_scale - loc.pow(2) - log_scale.exp()
            )
            loss += kld * self.kld_regularization

        logprob = state.output[self.logprob_key]
        if logprob is not None:
            loss += torch.mean(logprob) * self.logprob_regularization

        return loss
