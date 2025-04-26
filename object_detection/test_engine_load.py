import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

engine_file_path = "best.engine"  # Ajusta o caminho se precisares

# Carregar o engine
with open(engine_file_path, "rb") as f:
    engine_data = f.read()

runtime = trt.Runtime(TRT_LOGGER)
engine = runtime.deserialize_cuda_engine(engine_data)

assert engine is not None, "Erro: não foi possível carregar o engine!"

# Criar o context
context = engine.create_execution_context()

assert context is not None, "Erro: não foi possível criar o execution context!"

# Listar bindings (adaptado para TensorRT 10+)
for idx in range(engine.num_io_tensors):
    tensor_name = engine.get_tensor_name(idx)
    dtype = engine.get_tensor_dtype(tensor_name)
    shape = engine.get_tensor_shape(tensor_name)
    is_input = engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT

    print(f"{'Input' if is_input else 'Output'} tensor {idx}:")
    print(f"  Name: {tensor_name}")
    print(f"  Shape: {shape}")
    print(f"  DType: {dtype}")
