""" import cv2
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt

TRT_LOGGER = trt.Logger()

def load_engine(engine_path):
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    return engine

def preprocess(img, input_size):
    img_resized = cv2.resize(img, input_size)
    img_rgb = img_resized[:, :, ::-1]
    img_transposed = img_rgb.transpose(2, 0, 1)  # HWC ➝ CHW
    img_norm = img_transposed.astype(np.float32) / 255.0
    return np.ascontiguousarray(img_norm)

def infer(context, inputs, outputs, bindings, stream):
    cuda.memcpy_htod_async(inputs[0]['device'], inputs[0]['host'], stream)
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(outputs[0]['host'], outputs[0]['device'], stream)
    stream.synchronize()
    return outputs[0]['host']

# Caminho do modelo
engine_path = "best.engine"
engine = load_engine(engine_path)
context = engine.create_execution_context()

# Input info
input_binding_idx = engine.get_tensor_index("images")
output_binding_idx = engine.get_binding_index("output0")
input_shape = engine.get_binding_shape(input_binding_idx)
input_size = (input_shape[-1], input_shape[-2])  # (W, H)

# Alocar buffers
inputs = [{'host': cuda.pagelocked_empty(trt.volume(input_shape), dtype=np.float32),
           'device': cuda.mem_alloc(trt.volume(input_shape) * 4)}]
outputs = [{'host': cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(output_binding_idx)), dtype=np.float32),
            'device': cuda.mem_alloc(trt.volume(engine.get_binding_shape(output_binding_idx)) * 4)}]
bindings = [int(inputs[0]['device']), int(outputs[0]['device'])]
stream = cuda.Stream()

# === Testar com uma imagem ===
img_path = "exemplo.jpg"
img = cv2.imread(img_path)
img_input = preprocess(img, input_size)
inputs[0]['host'] = img_input.ravel()

output = infer(context, inputs, outputs, bindings, stream)

# Exibir resultado bruto (para debug)
print("Output shape:", output.shape)
print("Output data (primeiros 10 valores):", output[:10]) """


import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2

# Carregar o engine
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
with open("best.engine", "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())

# DEBUG: listar todos os bindings
for i in range(engine.num_bindings):
    name = engine.get_binding_name(i)
    print(f"Binding {i}: {name}")

# Definir o binding correto manualmente
input_binding_idx = 0  # ou outro número, dependendo do que aparecer

# Criar context
context = engine.create_execution_context()

# Preparar entrada
img = cv2.imread("image.jpg")  # <- teu caminho de teste
img = cv2.resize(img, (640, 640))
img = img.transpose((2, 0, 1))  # channels first
img = np.expand_dims(img, axis=0).astype(np.float32) / 255.0  # Normalizar, se necessário

# Alocar buffers
d_input = cuda.mem_alloc(img.nbytes)
d_output = cuda.mem_alloc(1000 * 4)  # tamanho aproximado, ajustar se souber

bindings = [int(d_input), int(d_output)]

# Copiar dados de input
cuda.memcpy_htod(d_input, img)

# Inference
context.execute_v2(bindings)

# Pegar output
output = np.empty([1000], dtype=np.float32)  # ajustar tamanho se necessário
cuda.memcpy_dtoh(output, d_output)

print("Output:", output)
