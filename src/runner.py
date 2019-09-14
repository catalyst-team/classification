try:
    import os

    if os.environ.get("USE_WANDB", "0") == "1":
        from catalyst.dl import SupervisedWandbRunner as Runner
    else:
        from catalyst.dl import SupervisedRunner as Runner
except ImportError:
    from catalyst.dl import SupervisedRunner as Runner


class ModelRunner(Runner):
    def __init__(self):
        super().__init__(input_key="image", output_key=None)
