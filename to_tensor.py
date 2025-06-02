import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

def build_engine():
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    
    with open("model.onnx", 'rb') as model:
        if not parser.parse(model.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
    
    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 28  # 256MB
    config.set_flag(trt.BuilderFlag.FP16) 
    
    engine = builder.build_engine(network, config)
    with open("model.engine", "wb") as f:
        f.write(engine.serialize())
    
    print("TensorRT Engine criado com sucesso!")
    return engine

