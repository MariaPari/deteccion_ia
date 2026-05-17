import onnx

modelo = onnx.load_model(
    "age_gender.onnx",
    load_external_data=True
)

onnx.save_model(
    modelo,
    "age_gender_single.onnx",
    save_as_external_data=False
)

print("Modelo convertido")