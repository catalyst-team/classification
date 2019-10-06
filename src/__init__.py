# flake8: noqa
from catalyst.dl import registry

from .experiment import Experiment
from .runner import ModelRunner as Runner

from .model import MultiHeadNet

registry.Model(MultiHeadNet)
