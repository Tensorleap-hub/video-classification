import cv2
import numpy as np
from typing import Dict, Optional, Union, Any
from numpy.typing import NDArray
from skimage.filters import gaussian, laplace  # type: ignore
from skimage.metrics import structural_similarity as ssim
from skimage.color import rgb2gray
import inspect

from code_loader.inner_leap_binder.leapbinder_decorators import tensorleap_metadata
from code_loader.contract.datasetclasses import PreprocessResponse
import kinetics_classes
from leap_data_utils import download_sample_index_from_dataset

# --------------------------- Utility Functions ---------------------------

def validate_image(image: NDArray[np.float32], expected_channels: Optional[int] = None) -> None:
    caller = inspect.stack()[1].function
    if not isinstance(image, np.ndarray):
        raise NotImplementedError(f"{caller}: Expected numpy array, got {type(image)}.")
    if image.dtype.name != 'float32':
        raise Exception(f"{caller}: Expected dtype float32, got {image.dtype.name}.")
    if image.ndim != 3:
        raise ValueError(f"{caller}: Expected 3D input, got {image.ndim}D.")
    if expected_channels and expected_channels != image.shape[-1]:
        raise ValueError(f"{caller}: Expected {expected_channels} channels, got {image.shape[-1]}.")

def normalize_video(frames) -> NDArray[np.float32]:
    frames = frames.numpy()
    frames = (frames - frames.min()) / (frames.max() - frames.min() + 1e-8)
    return np.transpose(frames, [1, 2, 3, 0])

def average_over_frames(video: NDArray[np.float32], frame_function) -> Union[float, NDArray]:
    return np.mean([frame_function(frame) for frame in video], axis=0)

# --------------------------- Frame-Level Features ---------------------------

def get_mean_abs_log_metadata(image: NDArray[np.float32], sigma: int = 1) -> float:
    validate_image(image)
    smoothed = gaussian(image, sigma=sigma, mode='reflect')
    log = laplace(smoothed, ksize=3)
    return float(np.mean(np.abs(log)))

def detect_sharpness(image: NDArray[np.float32]) -> float:
    validate_image(image)
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    return float(np.round(np.mean(np.sqrt(grad_x**2 + grad_y**2)), 2))

def total_variation(image: NDArray[np.float32]) -> float:
    validate_image(image)
    n_image = image[..., 0] if image.shape[-1] == 1 else image
    grad = np.array(np.gradient(n_image))
    return float(np.sum(np.sqrt(np.sum(grad**2, axis=0))))

# --------------------------- Video-Level Features ---------------------------

def compute_ssim_sequence(video: NDArray[np.float32]) -> list:
    return [
        ssim(rgb2gray(video[i]), rgb2gray(video[i + 1]), data_range=1.0)
        for i in range(video.shape[0] - 1)
    ]

# --------------------------- Motion and Temporal Dynamics ---------------------------

def extract_temporal_features(video: NDArray[np.float32]) -> Dict[str, float]:
    if video.max() <= 1.0:
        video *= 255.0
    video = video.astype(np.uint8)

    edge_densities, frame_diffs, flow_mags = [], [], []
    prev_gray = None

    for frame in video:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        edge_densities.append(np.count_nonzero(edges) / edges.size)

        if prev_gray is not None:
            diff = np.mean(np.abs(gray.astype(float) - prev_gray.astype(float)))
            frame_diffs.append(diff)

            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
                                                0.5, 1, 15, 3, 5, 1.2, 0)
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            flow_mags.append(np.mean(mag))

        prev_gray = gray

    return {
        'edge_density_mean': np.mean(edge_densities),
        'edge_density_std': np.std(edge_densities),
        'frame_diff_mean': np.mean(frame_diffs),
        'frame_diff_std': np.std(frame_diffs),
        'flow_mag_mean': np.mean(flow_mags),
        'flow_mag_min': np.min(flow_mags),
        'flow_mag_max': np.max(flow_mags),
        'flow_mag_std': np.std(flow_mags),
        'periodicity_score': compute_periodicity(flow_mags),
        **extract_temporal_dynamics(flow_mags),
    }

def compute_periodicity(signal: list) -> float:
    fft_vals = np.fft.fft(signal)
    mag = np.abs(fft_vals)
    return np.max(mag[1:len(mag)//2]) / np.sum(mag[1:])

def extract_temporal_dynamics(signal: list) -> Dict[str, Any]:
    velocity = np.diff(signal)
    acceleration = np.diff(velocity)
    return {
        'velocity_mean': np.mean(velocity),
        'velocity_std': np.std(velocity),
        'acceleration_mean': np.mean(acceleration),
        'acceleration_std': np.std(acceleration),
    }

# --------------------------- High-Level Video Metadata ---------------------------

def center_of_motion(video: NDArray[np.float32]) -> Dict[str, float]:
    if video.max() <= 1:
        video *= 255.
    centers_y, centers_x = [], []
    prev_gray = None

    for frame in video:
        gray = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 1, 15, 3, 5, 1.2, 0)
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            y, x = np.nonzero(mag > 1.0)
            if x.size:
                centers_y.append(np.mean(y))
                centers_x.append(np.mean(x))
        prev_gray = gray

    return {
        'motion_y_mean': np.mean(centers_y) if centers_y else 0,
        'motion_y_std': np.std(centers_y) if centers_y else 0,
        'motion_x_mean': np.mean(centers_x) if centers_x else 0,
        'motion_x_std': np.std(centers_x) if centers_x else 0,
    }

def foreground_ratio(video: NDArray[np.float32], threshold: float = 20) -> Dict[str, Any]:
    if video.max() <= 1:
        video *= 255.
    prev_gray = None
    motion_pixels = []

    for frame in video:
        gray = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        if prev_gray is not None:
            diff = np.abs(gray.astype(float) - prev_gray.astype(float))
            motion_pixels.append(np.mean(diff > threshold))
        prev_gray = gray

    return {
        'foreground_ratio_mean': np.mean(motion_pixels),
        'foreground_ratio_std': np.std(motion_pixels),
    }

def motion_direction_entropy(video: NDArray[np.float32]) -> Dict[str, float]:
    if video.max() <= 1:
        video *= 255.
    prev_gray = None
    angles_all = []

    for frame in video:
        gray = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 1, 15, 3, 5, 1.2, 0)
            _, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            angles_all.extend(ang.flatten())
        prev_gray = gray

    hist, _ = np.histogram(angles_all, bins=8, range=(0, 2 * np.pi), density=True)
    hist += 1e-6
    entropy = -np.sum(hist * np.log2(hist))
    return {'motion_entropy': entropy}

# --------------------------- Sample Metadata Entry ---------------------------

@tensorleap_metadata('metadata')
def sample_metadata(idx: int, preprocess: PreprocessResponse) -> Dict[str, Union[int, str, float]]:
    # Before accessing the dataset index, make sure the sample is downloaded
    download_sample_index_from_dataset(idx, preprocess.data['dataset'])
    sample = preprocess.data['dataset'][idx]
    video = normalize_video(sample["video"])

    ssim_vals = compute_ssim_sequence(video)
    rgb_mean = np.mean(video, axis=(0, 1, 2))

    metadata = {
        'gt_label': sample['label'],
        'video_name': sample['video_name'],
        'video_index': sample['video_index'],
        'clip_index': sample['clip_index'],
        'gt_action_label_name': kinetics_classes.get_label_by_id(sample['label']),

        'ssim_mean': np.mean(ssim_vals),
        'ssim_std': np.std(ssim_vals),
        'ssim_min': np.min(ssim_vals),
        'ssim_max': np.max(ssim_vals),

        'total_variation': average_over_frames(video, total_variation),
        'sharpness': average_over_frames(video, detect_sharpness),
        'mean_abs_log_metadata': average_over_frames(video, get_mean_abs_log_metadata),

        'mean_r': rgb_mean[0],
        'mean_g': rgb_mean[1],
        'mean_b': rgb_mean[2],
    }

    metadata.update(extract_temporal_features(video))
    metadata.update(center_of_motion(video))
    metadata.update(foreground_ratio(video))
    metadata.update(motion_direction_entropy(video))

    return metadata
