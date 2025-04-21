import torch
import torch.nn as nn
import numpy as np
from numpy.typing import NDArray
from typing import List

import kinetics_classes
# Tensorleap imports
from code_loader import leap_binder

from code_loader.inner_leap_binder.leapbinder_decorators import tensorleap_custom_loss
from code_loader.contract.datasetclasses import PreprocessResponse, DataStateType
from code_loader.inner_leap_binder.leapbinder_decorators import (
    tensorleap_preprocess,
    tensorleap_input_encoder,
    tensorleap_gt_encoder,
)
from leap_visualizers import frame_visualzier, frames_grid_visualzier, label_visualizer
from leap_metadata import sample_metadata
from leap_metrics import accuracy, get_predicted_label
from leap_data_utils import get_datasets
from leap_config import CONFIG
from leap_data_utils import download_sample_index_from_dataset

# Preprocess Function
@tensorleap_preprocess()
def preprocess_func() -> List[PreprocessResponse]:
    datasets_type = {"train":DataStateType.training,
                     "val": DataStateType.validation,
                     "test": DataStateType.test}
    datasets_keys = {"train": 'train_size_limit',
                     "val": 'val_size_limit',
                     "test": 'test_size_limit'}
    datasets = get_datasets()

    responses = []
    for split in ["train", "val", "test"]:
        responses.append(PreprocessResponse(length=min(CONFIG[datasets_keys[split]], len(datasets[split])),
                                            data={'dataset': datasets[split]},
                                            state=datasets_type[split]))
    return responses

@tensorleap_input_encoder('video', channel_dim=-1)
def input_encoder(idx: int, preprocess: PreprocessResponse) -> np.ndarray:
    download_sample_index_from_dataset(idx, preprocess.data['dataset']) # Before accessing the dataset index, make sure the sample is downloaded
    frames = preprocess.data['dataset'][idx]['video']
    frames = torch.permute(frames,(1,2,3,0))
    if hasattr(frames, 'detach'):
        frames = frames.detach().numpy().astype('float32')
    return frames

@tensorleap_gt_encoder('label')
def gt_encoder(idx: int, preprocess: PreprocessResponse) -> np.ndarray:
    download_sample_index_from_dataset(idx, preprocess.data['dataset']) # Before accessing the dataset index, make sure the sample is downloaded
    gt_label = preprocess.data['dataset'][idx]['label']
    one_hot_label = np.zeros(400, dtype=np.float32)
    one_hot_label[gt_label] = 1
    return one_hot_label

@tensorleap_custom_loss('cross_entropy_loss')
def cross_entropy_loss(predictions: NDArray[np.float32], targets: NDArray[np.float32]) -> NDArray[np.float32]:
    """
    Computes categorical cross-entropy loss.

    Args:
        predictions (torch.Tensor): Logits of shape (batch, categories).
        targets (torch.Tensor): One-hot encoded ground truth labels of shape (batch, categories).

    Returns:
        torch.Tensor: Scalar loss value.
    """
    bs = predictions.shape[0]
    loss = nn.CrossEntropyLoss()(torch.tensor(predictions), torch.tensor(targets))
    return loss.detach().numpy().reshape(bs)

leap_binder.add_prediction(name='classes', labels=kinetics_classes.kinetics_labels)
if __name__ == "__main__":
    leap_binder.check()