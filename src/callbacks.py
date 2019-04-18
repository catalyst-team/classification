import numpy as np
import torch
from torchnet.meter import AUCMeter, ConfusionMeter
from sklearn.metrics import confusion_matrix

from catalyst.dl.callbacks import Callback
from catalyst.dl.state import RunnerState
from .utils import to_numpy, plot_confusion_matrix, render_figure_to_tensor, \
    _get_tensorboard_logger

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.ioff() #http://matplotlib.org/faq/usage_faq.html (interactive mode)
import seaborn as sns


class EmbeddingsLossCallback(Callback):
    def __init__(
        self,
        emb_l2_reg: int = -1,
        input_key: str = "targets",
        embeddings_key: str = "embeddings",
        logits_key: str = "logits",
    ):
        self.emb_l2_reg = emb_l2_reg
        self.embeddings_key = embeddings_key
        self.logits_key = logits_key
        self.input_key = input_key

    def on_batch_end(self, state: RunnerState):
        embeddings = state.output[self.embeddings_key]
        logits = state.output[self.logits_key]
        targets = state.input[self.input_key]

        loss = state.criterion(logits, targets)

        if self.emb_l2_reg > 0:
            emb_loss = torch.mean(torch.norm(embeddings, dim=1))
            loss += emb_loss * self.emb_l2_reg

        state.loss = loss


class AUCCallback(Callback):
    def __init__(
        self,
        input_key: str = "targets",
        logits_key: str = "logits",
    ):
        self.auc_meter = AUCMeter()
        self.logits_key = logits_key
        self.input_key = input_key

    def on_loader_start(self, state):
        self.auc_meter.reset()

    def on_batch_end(self, state: RunnerState):
        logits = state.output[self.logits_key].float()
        targets = state.input[self.input_key].float()
        probs = torch.sigmoid(logits).detach()

        for i in range(probs.shape[1]):
            self.auc_meter.add(probs[:, i], targets[:, i])

    def on_loader_end(self, state: RunnerState):
        area, _, _ = self.auc_meter.value()
        state.metrics.epoch_values[state.loader_name]["auc"] = float(area)
        # state.metrics.epoch_values[state.loader_name]["tpr"] = float(tpr)
        # state.metrics.epoch_values[state.loader_name]["fpr"] = float(fpr)
        self.auc_meter.reset()


class ConfusionMatrixCallback(Callback):
    def __init__(
        self,
        input_key: str = "targets",
        output_key: str = "logits",
        prefix: str = "confusion_matrix",
        num_classes: int = None
    ):
        self.prefix = prefix
        self.output_key = output_key
        self.input_key = input_key
        assert num_classes is not None
        self.num_classes = num_classes
        self.confusion_matrix = ConfusionMeter(self.num_classes)

    def on_loader_start(self, state):
        self.confusion_matrix = ConfusionMeter(self.num_classes)

    def on_batch_end(self, state: RunnerState):
        self.confusion_matrix.add(
            state.output[self.output_key].detach(),
            state.input[self.input_key].detach()
        )

    def on_loader_end(self, state):
        class_names = [str(i) for i in range(self.num_classes)]
        cm = self.confusion_matrix.value()

        fig = plot_confusion_matrix(
            cm, class_names=class_names, normalize=True, noshow=True)
        fig = render_figure_to_tensor(fig)

        logger = _get_tensorboard_logger(state)
        logger.add_image(f"{self.prefix}/epoch", fig, global_step=state.epoch)


class ConfusionMatrixCallbackV2(Callback):
    def __init__(
        self,
        input_key: str = "targets",
        output_key: str = "logits",
        prefix: str = "confusion_matrix",
    ):
        self.prefix = prefix
        self.output_key = output_key
        self.input_key = input_key
        self.outputs = []
        self.targets = []

    def on_loader_start(self, state):
        self.outputs = []
        self.targets = []

    def on_batch_end(self, state: RunnerState):
        outputs = to_numpy(state.output[self.output_key])
        targets = to_numpy(state.input[self.input_key])

        outputs = np.argmax(outputs, axis=1)

        self.outputs.extend(outputs)
        self.targets.extend(targets)

    def on_loader_end(self, state):
        targets = self.targets
        outputs = self.outputs

        class_names = [str(i) for i in range(max(targets))]
        cm = confusion_matrix(outputs, targets)

        fig = plot_confusion_matrix(
            cm, class_names=class_names, normalize=True, noshow=True)
        fig = render_figure_to_tensor(fig)

        logger = _get_tensorboard_logger(state)
        logger.add_image(f"{self.prefix}/epoch", fig, global_step=state.epoch)
