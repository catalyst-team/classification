from typing import Mapping, Any
from catalyst.dl.experiments import Runner


class ModelRunner(Runner):
    def predict_batch(self, batch: Mapping[str, Any]):
        output = self.model(batch["image"])
        return output
