# flake8: noqa
# isort:skip_file

from catalyst.dl import registry, SupervisedRunner as Runner
from .experiment import Experiment
from .model import MultiHeadNet

registry.Model(MultiHeadNet)
