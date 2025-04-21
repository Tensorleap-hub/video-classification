from code_loader.inner_leap_binder.leapbinder_decorators import *
from numpy.typing import NDArray
import torch
from torch.nn import functional as F
import kinetics_classes

def add_col_padding(image: np.array, np_dtype: str = 'uint8') -> np.array:
    padded_columns = np.zeros(shape=(image.shape[0], 5, image.shape[2])).astype(np_dtype)
    image_with_padding = np.concatenate((image, padded_columns), axis=1)
    return image_with_padding


def add_row_padding(image: np.array, np_dtype: str = 'uint8') -> np.array:
    padded_rows = np.zeros(shape=(5, image.shape[1], image.shape[2])).astype(np_dtype)
    image_with_padding = np.concatenate((image, padded_rows), axis=0)
    return image_with_padding

def rescale_min_max(image: np.array, np_dtype: str = 'uint8') -> np.array:
    image_min = np.min(image)
    image_max = np.max(image)
    image_rescaled = (image - image_min) / (image_max - image_min)
    image_rescaled = (image_rescaled * 255.0).astype(np_dtype)
    return image_rescaled

def frames_grid_heatmap(frames: NDArray[np.float32]) -> LeapImage:
    """
    Arranges a batch of heatmap frames into a 4-row, 4-column grid with separators.

    Args:
        images (np.ndarray): Input array of shape (1, 16, 256, 256, 3).

    Returns:
        np.ndarray: Concatenated array of heatmaps with separators.
    """
    print(f"$$$ heatmap shape: {frames.shape}")
    # frames = frames.transpose(0,1,3,4,2)
    frames = np.squeeze(frames, 0)
    assert frames.shape == (16, 256, 256, 1), "Input shape must be (16, 256, 256, 1)"

    # frames = frames[0]  # Remove batch dimension
    stacked_rows = []

    for i in range(4):  # Four rows
        stacked_columns = []
        for j in range(i * 4, (i * 4) + 4):  # Four columns per row
            heatmap = frames[j, :, :, :]  # Extract frame (H, W, 3)

            if j < (i * 4) + 4 - 1:
                heatmap = add_col_padding(heatmap)  # Add vertical separator

            stacked_columns.append(heatmap)

        hstack = np.hstack(stacked_columns)  # Stack columns horizontally
        if i < 3:
            hstack = add_row_padding(hstack)  # Add horizontal separator

        stacked_rows.append(hstack)

    res = np.vstack(stacked_rows)  # Stack rows vertically
    return res



@tensorleap_custom_visualizer('frames_grid', visualizer_type=LeapDataType.Image, heatmap_function=frames_grid_heatmap)
def frames_grid_visualzier(frames: NDArray[np.float32]) -> LeapImage:
    """
    Arranges a batch of RGB video frames into a 4-row, 4-column grid with separators.

    Args:
        images (np.ndarray): Input array of shape (1, 16, 256, 256, 3).

    Returns:
        np.ndarray: Concatenated array of images with separators.
    """
    frames = np.squeeze(frames, 0)
    frames = frames.transpose(3, 0, 1, 2)
    print(f"$$$ frames_grid shape: {frames.shape}")
    assert frames.shape == (3, 16, 256, 256), "Input shape must be (3, 16, 256, 256)"

    # frames = frames[0]  # Remove batch dimension
    stacked_rows = []

    for i in range(4):  # Four rows
        stacked_columns = []
        for j in range(i * 4, (i * 4) + 4):  # Four columns per row
            image = frames[:, j, :, :]  # Extract frame
            image = np.moveaxis(image, 0, -1)  # Convert (C, H, W) to (H, W, C)
            image = rescale_min_max(image)  # Rescale values if necessary

            if j < (i * 4) + 4 - 1:
                image = add_col_padding(image)  # Add vertical separator

            stacked_columns.append(image)

        hstack = np.hstack(stacked_columns)  # Stack columns horizontally
        if i < 3:
            hstack = add_row_padding(hstack)  # Add horizontal separator

        stacked_rows.append(hstack)

    res = np.vstack(stacked_rows)  # Stack rows vertically
    return LeapImage(res)


def frame_heatmap(frames: NDArray[np.float32]) -> LeapImage:
    print(f"$$$ heatmap shape: {frames.shape}")
    return frames[0, 0, ...]


@tensorleap_custom_visualizer('frame', visualizer_type=LeapDataType.Image, heatmap_function=frame_heatmap)
def frame_visualzier(frames: np.ndarray) -> LeapImage:
    frames = frames.squeeze(0)
    frames = frames.transpose(3,0,1,2)
    frame = frames[:, 0, ...].transpose(1, 2, 0)
    frame = ((frame - frame.min()) / (frame.max() - frame.min()) * 255).astype(np.uint8)
    return LeapImage(frame)


@tensorleap_custom_visualizer('label', visualizer_type=LeapDataType.HorizontalBar)
def label_visualizer(pred: NDArray[np.float32], gt: NDArray[np.float32]) -> LeapHorizontalBar:
    pred =  F.log_softmax(torch.tensor(pred), dim=1)
    pred = torch.softmax(pred, dim=-1).detach().numpy()
    return LeapHorizontalBar(body=pred[0, ...], gt=gt[0, ...], labels=kinetics_classes.kinetics_labels)