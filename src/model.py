from typing import Dict
from copy import deepcopy

import torch
import torch.nn as nn
from catalyst.contrib.modules import Flatten
from catalyst.contrib.models import SequentialNet
from catalyst.contrib.models.encoder import ResnetEncoder

from .autoencoder import AEEncoder, AEDecoder, VAEEncoder

LOG_SIG_MAX = 2
LOG_SIG_MIN = -10


class MultiHeadNet(nn.Module):
    def __init__(
        self, encoder_net: nn.Module, embedding_net: nn.Module,
        head_nets: nn.ModuleList
    ):
        super().__init__()
        self.encoder_net = encoder_net
        self.embedding_net = embedding_net
        self.head_nets = head_nets

    def forward_embeddings(self, x: torch.Tensor):
        features = self.encoder_net(x)
        embeddings = self.embedding_net(features)
        return embeddings

    def forward(self, x: torch.Tensor):
        features = self.encoder_net(x)
        embeddings = self.embedding_net(features)
        result = {"features": features, "embeddings": embeddings}

        for key, head_net in self.head_nets.items():
            result[key] = head_net(embeddings)

        return result

    @classmethod
    def get_from_params(
        cls,
        image_size: int = None,
        encoder_params: Dict = None,
        embedding_net_params: Dict = None,
        heads_params: Dict = None,
    ) -> "MultiHeadNet":

        encoder_params_ = deepcopy(encoder_params)
        embedding_net_params_ = deepcopy(embedding_net_params)
        heads_params_ = deepcopy(heads_params)

        encoder_net = ResnetEncoder(**encoder_params_)

        encoder_input_shape: tuple = (3, image_size, image_size)
        encoder_input = torch.Tensor(torch.randn((1, ) + encoder_input_shape))
        encoder_output = encoder_net(encoder_input)
        enc_size = encoder_output.nelement()
        embedding_net_params_["hiddens"].insert(0, enc_size)

        embedding_net = SequentialNet(**embedding_net_params_)
        emb_size = embedding_net_params_["hiddens"][-1]

        head_kwargs_ = {}
        for key, value in heads_params_.items():
            head_kwargs_[key] = nn.Linear(emb_size, value, bias=True)
        head_nets = nn.ModuleDict(head_kwargs_)

        net = cls(
            encoder_net=encoder_net,
            embedding_net=embedding_net,
            head_nets=head_nets
        )

        return net


class MultiHeadNetAE(nn.Module):
    def __init__(self, encoder_net: nn.Module, head_nets: nn.ModuleList):
        super().__init__()
        self.encoder_net = encoder_net
        self.head_nets = head_nets

    def forward(self, x: torch.Tensor, deterministic=None):
        embeddings = self.encoder_net(x)
        result = {"embeddings": embeddings}

        for key, head_net in self.head_nets.items():
            result[key] = head_net(embeddings)

        return result

    @classmethod
    def get_from_params(
        cls,
        image_size: int = None,
        encoder_params: Dict = None,
        decoder_params: Dict = None,
        heads_params: Dict = None,
    ) -> "MultiHeadNet":

        encoder_params_ = deepcopy(encoder_params)
        heads_params_ = deepcopy(heads_params)

        encoder_net = AEEncoder(**encoder_params_)

        input_shape = (3, image_size, image_size)
        input_t = torch.Tensor(torch.randn((1,) + input_shape))
        output_t = encoder_net(input_t)
        emb_size = output_t.nelement()

        head_kwargs_ = {
            "decoder": AEDecoder(filters=encoder_net.filters, **decoder_params)
        }
        for key, value in heads_params_.items():
            head_kwargs_[key] = nn.Sequential(
                Flatten(),
                nn.Linear(emb_size, value, bias=True)
            )
        head_nets = nn.ModuleDict(head_kwargs_)

        net = cls(encoder_net=encoder_net, head_nets=head_nets)

        return net


class MultiHeadNetVAE(nn.Module):
    def __init__(self, encoder_net: nn.Module, head_nets: nn.ModuleList):
        super().__init__()
        self.encoder_net = encoder_net
        self.head_nets = head_nets

    def forward(self, x: torch.Tensor, deterministic=None):
        embeddings = self.encoder_net(x)
        result = {"embeddings": embeddings}

        for key, head_net in self.head_nets.items():
            result[key] = head_net(embeddings)

        return result

    @classmethod
    def get_from_params(
        cls,
        image_size: int = None,
        encoder_params: Dict = None,
        decoder_params: Dict = None,
        heads_params: Dict = None,
    ) -> "MultiHeadNet":

        encoder_params_ = deepcopy(encoder_params)
        heads_params_ = deepcopy(heads_params)

        encoder_net = VAEEncoder(**encoder_params_)

        input_shape = (3, image_size, image_size)
        input_t = torch.Tensor(torch.randn((1,) + input_shape))
        output_t = encoder_net(input_t)
        emb_size = output_t.nelement()

        head_kwargs_ = {
            "decoder": AEDecoder(filters=encoder_net.filters, **decoder_params)
        }
        for key, value in heads_params_.items():
            head_kwargs_[key] = nn.Sequential(
                Flatten(),
                nn.Linear(emb_size, value, bias=True)
            )
        head_nets = nn.ModuleDict(head_kwargs_)

        net = cls(encoder_net=encoder_net, head_nets=head_nets)

        return net
