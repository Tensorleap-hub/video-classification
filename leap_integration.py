import os
import numpy as np
import tensorflow as tf
from code_loader.contract.datasetclasses import PredictionTypeHandler
from code_loader.inner_leap_binder.leapbinder_decorators import tensorleap_load_model, tensorleap_integration_test
from code_loader.plot_functions.visualize import visualize

import kinetics_classes
from leap_binder import preprocess_func, input_encoder, gt_encoder, cross_entropy_loss
from leap_visualizers import frame_visualzier, frames_grid_visualzier, label_visualizer
from leap_metrics import accuracy, get_predicted_label
from leap_metadata import sample_metadata

# Define prediction types with class labels
prediction_type = PredictionTypeHandler('classes', kinetics_classes.kinetics_labels)

@tensorleap_load_model([prediction_type])
def load_model():
    """Load pre-trained X3D model from Keras H5 file"""
    dir_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(dir_path, 'models/x3d.h5')
    model = tf.keras.models.load_model(model_path)
    return model

@tensorleap_integration_test()
def check_integration(idx, subset, plot=True):
    """
    Defines the complete inference pipeline (replaces leap_mapping.yaml).
    Shows Tensorleap how all components are connected.
    """
    # 1. Load input video and ground truth (batch dimension already added by decorators)
    video = input_encoder(idx, subset)
    gt = gt_encoder(idx, subset)

    # 2. Run model inference
    model = load_model()
    predictions = model([video])

    # 3. Call visualizers (defines visualizer connections)
    frame_vis = frame_visualzier(video)
    grid_vis = frames_grid_visualzier(video)
    label_vis = label_visualizer(predictions, gt)

    # Visualize for debugging/sanity check
    if plot:
        visualize(frame_vis, "Single Frame")
        visualize(grid_vis, "Frame Grid (4x4)")

    # 4. Call metrics (defines metric connections)
    acc = accuracy(predictions, gt)
    pred_label = get_predicted_label(predictions)

    # 5. Call loss (defines loss connection)
    loss = cross_entropy_loss(predictions, gt)

    # 6. Call metadata (optional)
    meta = sample_metadata(idx, subset)


if __name__ == "__main__":
    # Run integration test with first sample from training set
    subsets = preprocess_func()
    check_integration(0, subsets[0])
