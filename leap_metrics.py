import torch
import numpy as np
import torch.nn.functional as F
from numpy.typing import NDArray
from code_loader.contract.enums import MetricDirection
from code_loader.inner_leap_binder.leapbinder_decorators import tensorleap_custom_metric


@tensorleap_custom_metric('accuracy', direction=MetricDirection.Upward)
def accuracy(predictions: NDArray[np.float32], targets: NDArray[np.float32]) -> NDArray[np.float32]:
    """
    Computes categorical accuracy.

    Args:
        predictions (torch.Tensor): Logits of shape (batch, categories).
        targets (torch.Tensor): One-hot encoded ground truth labels of shape (batch, categories).

    Returns:
        torch.Tensor: Scalar accuracy value.
    """
    bs = predictions.shape[0]
    pred_index = get_predicted_label(predictions)
    target_index = targets.argmax(axis=-1)

    acc = (pred_index == target_index).sum()
    return acc.reshape(bs)

@tensorleap_custom_metric('predicted_label', compute_insights=False)
def get_predicted_label(pred: NDArray[np.float32]) -> NDArray[np.float32]:
    bs = pred.shape[0]
    pred =  F.log_softmax(torch.tensor(pred), dim=1).detach().numpy()
    pred_index = pred.argmax(axis=-1)
    return pred_index.reshape(bs)

