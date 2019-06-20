from copy import deepcopy
import torch.nn as nn
from catalyst.contrib.models import SequentialNet
from catalyst.contrib.models.encoder import ResnetEncoder


class MultiHeadNet(nn.Module):
    def __init__(self, encoder_params, embedding_net_params, heads_params):
        super().__init__()

        encoder_params_ = deepcopy(encoder_params)
        embedding_net_params_ = deepcopy(embedding_net_params)
        heads_params_ = deepcopy(heads_params)

        self.encoder_net = encoder = ResnetEncoder(**encoder_params_)
        self.enc_size = encoder.out_features

        if self.enc_size is not None:
            embedding_net_params_["hiddens"].insert(0, self.enc_size)

        self.embedding_net = SequentialNet(**embedding_net_params_)
        self.emb_size = embedding_net_params_["hiddens"][-1]

        head_kwargs_ = {}
        for key, value in heads_params_.items():
            head_kwargs_[key] = nn.Linear(self.emb_size, value, bias=True)
        self.heads = nn.ModuleDict(head_kwargs_)

    def forward(self, x):
        features = self.encoder_net(x)
        embeddings = self.embedding_net(features)
        result = {"features": features, "embeddings": embeddings}

        for key, value in self.heads.items():
            result[key] = value(embeddings)

        return result
