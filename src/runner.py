from typing import Mapping, Any
from catalyst.dl.experiments import Runner


class ModelRunner(Runner):

    def predict_batch(self, batch: Mapping[str, Any]):
        embeddings, logits = self.model(image=batch["image"])
        output = {"embeddings": embeddings, "logits": logits}
        return output
