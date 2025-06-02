import torch
import torch.onnx
from model import TwinLite as net
import numpy as np

def convert_to_onnx():
 
    model = net.TwinLiteNet()
    model = model.cuda()
    model.load_state_dict(torch.load('pretrained/model.pth'))
    model.eval()  
    model = model.cuda()
    
    dummy_input = torch.randn(1, 3, 360, 640).cuda()
     
    torch.onnx.export(
        model,                          
        dummy_input,                    # entrada dummy
        "model.onnx",      # nome do arquivo de saída
        export_params=True,             # armazenar pesos treinados
        opset_version=11,               # versão ONNX
        do_constant_folding=True,       # otimização
        input_names=['input'],          # nomes das entradas
        output_names=['drivable_area', 'lane_lines'],  # nomes das saídas
        dynamic_axes={
            'input': {0: 'batch_size'},
            'drivable_area': {0: 'batch_size'},
            'lane_lines': {0: 'batch_size'}
        }
    )
    #onnx_model = onnx.load("model.onnx")
    #onnx.checker.check_model(onnx_model)
    print("Modelo convertido para ONNX com sucesso!")

if __name__ == "__main__":
    convert_to_onnx()

