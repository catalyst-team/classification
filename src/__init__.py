# flake8: noqa
from catalyst.dl import registry

from .experiment import Experiment
from .runner import ModelRunner as Runner
from .callbacks import EmbeddingsCriterionCallback
from .model import MultiHeadNet

registry.Model(MultiHeadNet)
registry.Callback(EmbeddingsCriterionCallback)
