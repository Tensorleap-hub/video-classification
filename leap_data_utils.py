from pytorchvideo.data import Kinetics
from pytorchvideo.data.clip_sampling import ConstantClipsPerVideoSampler
from torch.utils.data import RandomSampler
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)

from pytorchvideo.transforms import (
    ApplyTransformToKey,
    RemoveKey,
    UniformTemporalSubsample,
    ShortSideScale
)
from leap_config import CONFIG
import requests
import os
import logging

logger = logging.getLogger(__name__)

def _make_transforms():
    transform = [
        _video_transform(),
        RemoveKey("audio"),
    ]

    return Compose(transform)


def _video_transform():
    """
    This function contains example transforms using both PyTorchVideo and TorchVision
    in the same Callable. For 'train' mode, we use augmentations (prepended with
    'Random'), for 'val' mode we use the respective determinstic function.
    """
    transform = ApplyTransformToKey(
        key="video",
        transform=Compose(
            [
                UniformTemporalSubsample(CONFIG["model_transform_params"]["num_frames"]),
                Lambda(lambda x: x / 255.0),
                NormalizeVideo(CONFIG["mean"], CONFIG["std"]),
                ShortSideScale(size=CONFIG["model_transform_params"]["side_size"]),
                CenterCropVideo(
                    crop_size=(CONFIG["model_transform_params"]["crop_size"], CONFIG["model_transform_params"]["crop_size"])
                )
            ]
        ),
    )

    return transform

def get_full_path(path) -> str:
    if os.path.isabs(path):
        return path
    elif "IS_CLOUD" in os.environ and os.path.exists("/fsx"):  # SaaS env
        return os.path.join("/fsx", path)
    elif "IS_CLOUD" in os.environ and os.path.exists("/nfs"):  # SaaS env
        return os.path.join("/nfs", path)
    elif "GENERIC_HOST_PATH" in os.environ:  # OnPrem
        return os.path.join(os.environ["GENERIC_HOST_PATH"], path)
    else:
        return os.path.join(os.path.expanduser("~/tensorleap/data"), path)

def download_sample_index_from_dataset(idx, dataset):
    aws_path = dataset._labeled_videos[idx][0].lstrip(get_full_path(CONFIG["dataset_path_from_root"]))
    local_path = download_from_s3(aws_path)
    if local_path is not None and not os.path.exists(local_path):
        raise RuntimeError("Failed to download video from AWS bucket")

def download_from_s3(s3_relative_file_path):
    aws_data_root_path = CONFIG["aws_data_root_path"]
    # Construct the local file path
    local_file_path = get_full_path(s3_relative_file_path)
    if os.path.exists(local_file_path):
        return local_file_path

    os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

    public_url = f"https://{CONFIG['bucket_name']}.s3.amazonaws.com/{aws_data_root_path}/{s3_relative_file_path}"

    try:
        response = requests.get(public_url)
        response.raise_for_status()
        with open(local_file_path, "wb") as f:
            f.write(response.content)
            logger.info(f"Downloading video from {public_url}")
    except Exception as e:
        logger.error(f"Failed to download file from public S3 URL: {e}")
        return None

    return local_file_path

def get_datasets():
    datasets = {}
    for split in "train", "val", "test":
        download_from_s3(CONFIG["dataset_splits_paths"][split]) # Download the split txt files if they don't exist
        datasets[split] = (
            Kinetics(
                data_path=get_full_path(CONFIG["dataset_splits_paths"][split]),
                video_path_prefix=get_full_path(split),
                clip_sampler=ConstantClipsPerVideoSampler(CONFIG["clip_duration"], 1),
                transform=_make_transforms(),
                video_sampler=RandomSampler)
        )
    return datasets

