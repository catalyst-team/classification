# flake8: noqa
from catalyst.contrib.registry import Registry

from .experiment import Experiment
from .runner import ModelRunner as Runner
from .callbacks import EmbeddingsLossCallback
from .model import baseline

Registry.model(baseline)
Registry.callback(EmbeddingsLossCallback)
