import itertools
import numpy as np
import torch

from catalyst.dl.callbacks import RunnerState, TensorboardLogger
from tensorboardX import SummaryWriter


def to_numpy(x) -> np.ndarray:
    """
    Convert whatever to numpy array
    :param x: List, tuple, PyTorch tensor or numpy array
    :return: Numpy array
    """
    if isinstance(x, np.ndarray):
        return x
    elif isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    elif isinstance(x, list) or isinstance(x, tuple):
        return np.array(x)
    else:
        raise ValueError("Unsupported type")
    return x


def tensor_from_rgb_image(image: np.ndarray) -> torch.Tensor:
    image = np.moveaxis(image, -1, 0)
    image = np.ascontiguousarray(image)
    image = torch.from_numpy(image)
    return image


def plot_confusion_matrix(
    cm,
    class_names,
    normalize=False,
    title="Confusion matrix",
    fname=None,
    noshow=False
):
    """
    Render the confusion matrix and return matplotlib"s figure with it.
    Normalization can be applied by setting `normalize=True`.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cmap = plt.cm.Blues

    if normalize:
        cm = cm.astype(np.float32) / cm.sum(axis=1)[:, np.newaxis]

    f = plt.figure(figsize=(16, 16))
    plt.title(title)
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    # f.tick_params(direction="inout")
    # f.set_xticklabels(varLabels, rotation=45, ha="right")
    # f.set_yticklabels(varLabels, rotation=45, va="top")

    plt.yticks(tick_marks, class_names)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j, i, format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")

    if fname is not None:
        plt.savefig(fname=fname)

    if not noshow:
        plt.show()

    return f


def render_figure_to_tensor(figure):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure.canvas.draw()

    # string = figure.canvas.tostring_argb()

    image = np.array(figure.canvas.renderer._renderer)
    plt.close(figure)
    del figure

    image = tensor_from_rgb_image(image)
    return image


def _get_tensorboard_logger(state: RunnerState) -> SummaryWriter:
    for logger in state.loggers:
        if isinstance(logger, TensorboardLogger):
            return logger.loggers[state.loader_name]
    raise RuntimeError(
        f"Cannot find Tensorboard logger for loader {state.loader_name}")
