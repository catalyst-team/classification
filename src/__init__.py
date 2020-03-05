# flake8: noqa
# isort:skip_file

from catalyst.dl import registry

from .callbacks import EmbeddingsCriterionCallback
from .experiment import Experiment
from .runner import ModelRunner as Runner

from .model import MultiHeadNet

registry.Callback(EmbeddingsCriterionCallback)
registry.Model(MultiHeadNet)
