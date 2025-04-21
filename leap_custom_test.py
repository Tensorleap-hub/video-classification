import numpy as np
import tensorflow as tf
from code_loader.helpers import visualize
from leap_binder import gt_encoder, input_encoder, preprocess_func, cross_entropy_loss
from leap_metadata import sample_metadata
from leap_metrics import get_predicted_label, accuracy
from leap_visualizers import frame_visualzier, frames_grid_visualzier


def check_integration():
    # Load h5 model
    model = tf.keras.models.load_model('models/x3d.h5')

    # Load datasets
    responses = preprocess_func()
    val_data = responses[1]

    for i in range(val_data.length):
        print(f"Processing sample {i + 1}/{val_data.length}")
        frames = input_encoder(i, val_data)
        frames = frames[np.newaxis, ...]
        label = gt_encoder(i, val_data)
        label = label[np.newaxis, ...]
        metadata = sample_metadata(i, val_data)

        # Predict
        prediction = model(frames).numpy()


        # focal_loss_res = focal_loss(prediction, label)
        cross_entropy_loss_res = cross_entropy_loss(prediction, label)
        predicted_label = get_predicted_label(prediction)
        acc = accuracy(prediction, label)

        frame_vis = frame_visualzier(frames)
        frames_grid_vis = frames_grid_visualzier(frames)

        # Visualize
        visualize(frame_vis)
        visualize(frames_grid_vis)

        # plt.show()
    print("Integration check passed")
check_integration()