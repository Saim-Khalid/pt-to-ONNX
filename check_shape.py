import onnxruntime as rt
import numpy as np

sess_1 = rt.InferenceSession("model.onnx")
print("The model expects input shape:", sess_1.get_inputs()[0].shape)