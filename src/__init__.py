# flake8: noqa
from catalyst.dl import registry

from .experiment import Experiment
from .runner import ModelRunner as Runner

from .callbacks import EmbeddingsCriterionCallback, AECriterionCallback
from .model import MultiHeadNet, MultiHeadNetAE

registry.Model(MultiHeadNet)
registry.Model(MultiHeadNetAE)
registry.Callback(EmbeddingsCriterionCallback)
registry.Callback(AECriterionCallback)
