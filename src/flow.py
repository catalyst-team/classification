import torch
import torch.nn as nn

from catalyst.contrib.models import SequentialNet
from catalyst.utils import create_optimal_inner_init, outer_init, log1p_exp
from catalyst.contrib.registry import MODULES


class SquashingLayer(nn.Module):
    def __init__(self, squashing_fn=nn.Tanh):
        """
        Layer that squashes samples from some distribution to be bounded.
        """
        super().__init__()

        self.squashing_fn = MODULES.get_if_str(squashing_fn)()

    def forward(self, x, x_logprob=None):
        # compute log det jacobian of squashing transformation
        if isinstance(self.squashing_fn, nn.Tanh):
            log2 = torch.log(torch.tensor(2.0).to(x.device))
            log_det_jacobian = 2 * (log2 + x - log1p_exp(2 * x))
            log_det_jacobian = torch.sum(log_det_jacobian, dim=-1)
        elif isinstance(self.squashing_fn, nn.Sigmoid):
            log_det_jacobian = -x - 2 * log1p_exp(-x)
            log_det_jacobian = torch.sum(log_det_jacobian, dim=-1)
        elif self.squashing_fn is None:
            return x, x_logprob
        else:
            raise NotImplementedError
        x = self.squashing_fn.forward(x)
        if x_logprob is not None:
            x_logprob = x_logprob - log_det_jacobian
        return x, x_logprob


class CouplingLayer(nn.Module):
    def __init__(
        self,
        emb_size,
        layer_fn,
        activation_fn=nn.ReLU,
        bias=True,
        parity="odd"
    ):
        """
        [@TODO: WIP, do not trust docs!]
        Conditional affine coupling layer used in Real NVP Bijector.
        Original paper: https://arxiv.org/abs/1605.08803
        Adaptation to RL: https://arxiv.org/abs/1804.02808
        Important notes
        ---------------
        1. State embeddings are supposed to have size (action_size * 2).
        2. Scale and translation networks used in the Real NVP Bijector
        both have one hidden layer of (action_size) (activation_fn) units.
        3. Parity ("odd" or "even") determines which part of the input
        is being copied and which is being transformed.
        """
        super().__init__()
        assert parity in ["odd", "even"]

        layer_fn = MODULES.get_if_str(layer_fn)
        activation_fn = MODULES.get_if_str(activation_fn)

        self.parity = parity
        if self.parity == "odd":
            self.copy_size = emb_size // 2
        else:
            self.copy_size = emb_size - emb_size // 2

        self.scale_prenet = SequentialNet(
            hiddens=[self.copy_size, emb_size],
            layer_fn=layer_fn,
            activation_fn=activation_fn,
            norm_fn=None,
            bias=bias
        )
        self.scale_net = SequentialNet(
            hiddens=[emb_size, emb_size - self.copy_size],
            layer_fn=layer_fn,
            activation_fn=None,
            norm_fn=None,
            bias=True
        )

        self.translation_prenet = SequentialNet(
            hiddens=[self.copy_size, emb_size],
            layer_fn=layer_fn,
            activation_fn=activation_fn,
            norm_fn=None,
            bias=bias
        )
        self.translation_net = SequentialNet(
            hiddens=[emb_size, emb_size - self.copy_size],
            layer_fn=layer_fn,
            activation_fn=None,
            norm_fn=None,
            bias=True
        )

        inner_init = create_optimal_inner_init(nonlinearity=activation_fn)
        self.scale_prenet.apply(inner_init)
        self.scale_net.apply(outer_init)
        self.translation_prenet.apply(inner_init)
        self.translation_net.apply(outer_init)

    def forward(self, x, x_logprob=None):
        if self.parity == "odd":
            x_copy = x[:, :self.copy_size]
            x_transform = x[:, self.copy_size:]
        else:
            x_copy = x[:, -self.copy_size:]
            x_transform = x[:, :-self.copy_size]
        x = x_copy

        t = self.translation_prenet(x)
        t = self.translation_net(t)

        s = self.scale_prenet(x)
        s = self.scale_net(s)

        out_transform = t + x_transform * torch.exp(s)

        if self.parity == "odd":
            x = torch.cat((x_copy, out_transform), dim=1)
        else:
            x = torch.cat((out_transform, x_copy), dim=1)

        if x_logprob is not None:
            log_det_jacobian = s.sum(dim=1)
            x_logprob = x_logprob - log_det_jacobian

        return x, x_logprob


class RealNVP(nn.Module):
    def __init__(
        self,
        emb_size,
        activation_fn=nn.ReLU,
        bias=False,
        squashing_fn=nn.Tanh,
    ):
        super().__init__()
        layer_fn = nn.Linear
        activation_fn = MODULES.get_if_str(activation_fn)
        self.emb_size = emb_size

        self.coupling1 = CouplingLayer(
            emb_size=emb_size,
            layer_fn=layer_fn,
            activation_fn=activation_fn,
            bias=bias,
            parity="odd"
        )
        self.coupling2 = CouplingLayer(
            emb_size=emb_size,
            layer_fn=layer_fn,
            activation_fn=activation_fn,
            bias=bias,
            parity="even"
        )
        self.squashing_layer = SquashingLayer(squashing_fn)

    def forward(self, x, x_logprob=None):
        bs, nc, nf, _ = x.shape
        x = x.view(bs, -1)
        x, x_logprob = self.coupling1.forward(x, x_logprob)
        x, x_logprob = self.coupling2.forward(x, x_logprob)
        x, x_logprob = self.squashing_layer.forward(x, x_logprob)
        x = x.view(bs, nc, nf, nf)
        return x, x_logprob
