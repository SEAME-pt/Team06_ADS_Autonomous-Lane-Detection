import torch
from linenet import create_linenet

# Parâmetros
model_path = "model.pth"
onnx_path = "model.onnx"
input_size = (224, 224 )
variant = "lite"        

# Crie o modelo e carregue os pesos
model = create_linenet(variant=variant, num_classes=1, input_channels=3)
checkpoint = torch.load(model_path, map_location="cpu")
if "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)
model.eval()

# Crie um dummy input
dummy_input = torch.randn(1, 3, input_size[0], input_size[1])

# Exporte para ONNX
torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    input_names=["input"],
    output_names=["output"],
    opset_version=11,
    do_constant_folding=True,
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
)
print(f"Modelo exportado para {onnx_path}")

