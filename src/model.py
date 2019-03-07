import torch
import torch.nn as nn
from catalyst.contrib.models import ResnetEncoder, SequentialNet


class MiniNet(nn.Module):
    def __init__(
        self,
        enc,
        n_cls,
        hiddens,
        emb_size,
        activation_fn=torch.nn.ReLU,
        norm_fn=None,
        bias=True,
        dropout=None
    ):
        super().__init__()
        self.encoder = enc
        self.emb_net = SequentialNet(
            hiddens=hiddens + [emb_size],
            activation_fn=activation_fn,
            norm_fn=norm_fn,
            bias=bias,
            dropout=dropout
        )
        self.head = nn.Linear(emb_size, n_cls, bias=True)

    def forward(self, *, image):
        features = self.encoder(image)
        embeddings = self.emb_net(features)
        logits = self.head(embeddings)
        return embeddings, logits


def baseline(encoder_params, head_params):
    img_enc = ResnetEncoder(**encoder_params)
    net = MiniNet(enc=img_enc, **head_params)
    return net
