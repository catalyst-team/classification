# flake8: noqa
from catalyst.dl import registry

from .experiment import Experiment
try:
    import os

    if os.environ.get("USE_WANDB", "0") == "1":
        from catalyst.dl import SupervisedWandbRunner as Runner
    else:
        from catalyst.dl import SupervisedRunner as Runner
except ImportError:
    from catalyst.dl import SupervisedRunner as Runner

from .callbacks import EmbeddingsCriterionCallback, AECriterionCallback
from .model import MultiHeadNet, MultiHeadNetAE

registry.Model(MultiHeadNet)
registry.Model(MultiHeadNetAE)
registry.Callback(EmbeddingsCriterionCallback)
registry.Callback(AECriterionCallback)
