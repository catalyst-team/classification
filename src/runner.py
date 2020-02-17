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


class ModelRunner(Runner):
    def __init__(self, model=None, device=None):
        super().__init__(
            model=model, device=device, input_key="image", output_key=None
        )
