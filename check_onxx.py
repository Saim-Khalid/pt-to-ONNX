import onnx

onnx_model = onnx.load("/Users/saimkhalid/Desktop/Streamlit/model.onnx")
try:
    onnx.checker.check_model(onnx_model)
except onnx.checker.ValidationError as e:
    print('The model is invalid: %s' % e)
else:
    print('The model is valid!')