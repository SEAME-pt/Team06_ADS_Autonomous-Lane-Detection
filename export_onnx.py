import torch
from optimized_linenet import create_optimized_linenet

def export_model_to_onnx(
    weights_path="best_model.pth",
    onnx_output_path="model.onnx",
    input_size=(1, 3, 224, 224),
    variant="balanced"
):
    model = create_optimized_linenet(variant=variant)
    checkpoint = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(checkpoint)
    model.eval()

    dummy_input = torch.randn(input_size)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    print(f" Modelo exportado para ONNX: {onnx_output_path}")

if __name__ == "__main__":
    export_model_to_onnx(
        weights_path="outputs_balanced/best_model.pth",
        onnx_output_path="outputs_balanced/model.onnx",
        input_size=(1, 3, 224, 224),
        variant="balanced"
    )

