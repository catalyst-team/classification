# flake8: noqa
from catalyst.dl import registry

from .experiment import Experiment
from .runner import ModelRunner as Runner
from .callbacks import EmbeddingsLossCallback
from .model import MultiHeadNet

registry.Model(MultiHeadNet)
registry.Callback(EmbeddingsLossCallback)
