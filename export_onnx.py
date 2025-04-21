from pytorchvideo.models.hub.x3d import x3d_m
import torch
import onnx
import onnxruntime as ort
import numpy as np
import os


model = x3d_m(pretrained=True)
model.eval()
input_tensor = torch.randn(1,3,16,256,256)

onnx_path = "models/x3d.onnx"

if not os.path.exists(os.path.dirname(onnx_path)):
    os.mkdir(os.path.dirname(onnx_path))

torch.onnx.export(
    model,
    input_tensor,
    onnx_path,
    input_names=['input'],
    output_names=['output'],
)

onnx_model = onnx.load(onnx_path)
onnx.checker.check_model(onnx_model)

# Run inference using ONNX Runtime
ort_session = ort.InferenceSession(onnx_path)
onnx_input = input_tensor.numpy()
onnx_outputs = ort_session.run(None, {'input': onnx_input})
onnx_output = onnx_outputs[0]

# Run inference using PyTorch
with torch.no_grad():
    torch_output = model(input_tensor).numpy()

# Compare ONNX and PyTorch outputs
np.testing.assert_allclose(torch_output, onnx_output, rtol=1e-03, atol=1e-05)
print("SUCCESS: The outputs from PyTorch and ONNX match!")