#pip install onnx
#pip install onnx2keras

import torch
import torchvision
import keras
import onnx
from onnx2keras import onnx_to_keras

model = torch.load("Model.pt")

# Convert the model to a format that can be used by Keras
model = torchvision.models.resnet18(pretrained=False)
model.eval()

# Create a dummy input to run the model on
input_data = torch.randn(1, 3, 224, 224)

# Export the model to ONNX format
torch.onnx.export(model, input_data, "model.onnx", verbose=True)