# flake8: noqa
# isort:skip_file

from catalyst.dl import registry

from .experiment import Experiment
try:
    import os

    if os.environ.get("USE_ALCHEMY", "0") == "1":
        from catalyst.contrib.dl import SupervisedAlchemyRunner as Runner
    elif os.environ.get("USE_NEPTUNE", "0") == "1":
        from catalyst.contrib.dl import SupervisedNeptuneRunner as Runner
    elif os.environ.get("USE_WANDB", "0") == "1":
        from catalyst.contrib.dl import SupervisedWandbRunner as Runner
    else:
        from catalyst.dl import SupervisedRunner as Runner
except ImportError:
    from catalyst.dl import SupervisedRunner as Runner

from .model import MultiHeadNet

registry.Model(MultiHeadNet)
