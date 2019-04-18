# flake8: noqa
from catalyst.dl import registry

from .experiment import Experiment
from .runner import ModelRunner as Runner
from .callbacks import EmbeddingsLossCallback, AUCCallback, \
    ConfusionMatrixCallback, ConfusionMatrixCallbackV2
from .model import MultiHeadNet

registry.Model(MultiHeadNet)
registry.Callback(EmbeddingsLossCallback)
registry.Callback(AUCCallback)
registry.Callback(ConfusionMatrixCallback)
registry.Callback(ConfusionMatrixCallbackV2)
