import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from threading import Thread
import time

class TensorRTInference:
    def __init__(self, engine_path):
 
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f:
            self.engine = trt.Runtime(self.logger).deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
        
 
        self.inputs, self.outputs, self.bindings = self.allocate_buffers()
        
    def allocate_buffers(self):
        inputs = []
        outputs = []
        bindings = []
        
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding))
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            
 
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            bindings.append(int(device_mem))
            
            if self.engine.binding_is_input(binding):
                inputs.append({'host': host_mem, 'device': device_mem})
            else:
                outputs.append({'host': host_mem, 'device': device_mem})
                
        return inputs, outputs, bindings
    
    def infer(self, input_data):
        # Copiar dados para GPU
        np.copyto(self.inputs[0]['host'], input_data.ravel())
        cuda.memcpy_htod(self.inputs[0]['device'], self.inputs[0]['host'])
        
        # Executar inferência
        self.context.execute_v2(bindings=self.bindings)
        
        # Copiar resultados da GPU
        outputs = []
        for output in self.outputs:
            cuda.memcpy_dtoh(output['host'], output['device'])
            outputs.append(output['host'])
            
        return outputs

class CSICamera:
    def __init__(self, width=640, height=360, fps=30):
        self.pipeline = (
            f"nvarguscamerasrc ! "
            f"video/x-raw(memory:NVMM), width={width}, height={height}, "
            f"format=NV12, framerate={fps}/1 ! "
            f"nvvidconv flip-method=0 ! "
            f"video/x-raw, width={width}, height={height}, format=BGRx ! "
            f"videoconvert ! "
            f"video/x-raw, format=BGR ! appsink"
        )
        
        self.cap = cv2.VideoCapture(self.pipeline, cv2.CAP_GSTREAMER)
        self.frame = None
        self.running = False
        
    def start(self):
        self.running = True
        self.thread = Thread(target=self.update)
        self.thread.start()
        
    def update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.frame = frame
                
    def read(self):
        return self.frame
    
    def stop(self):
        self.running = False
        self.thread.join()
        self.cap.release()

def preprocess_frame(frame):
    img = cv2.resize(frame, (640, 360))
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR para RGB e HWC para CHW
    img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
    return img

def postprocess_outputs(outputs, original_frame):
    drivable_area = outputs[0].reshape(1, 2, 360, 640)
    lane_lines = outputs[1].reshape(1, 2, 360, 640)
    
    # Aplicar argmax para obter predições
    da_predict = np.argmax(drivable_area, axis=1)[0]
    ll_predict = np.argmax(lane_lines, axis=1)[0]
    
    # Redimensionar para frame original
    h, w = original_frame.shape[:2]
    da_mask = cv2.resize((da_predict * 255).astype(np.uint8), (w, h))
    ll_mask = cv2.resize((ll_predict * 255).astype(np.uint8), (w, h))
    
    # Aplicar cores como no test_image.py
    result = original_frame.copy()
    result[da_mask > 100] = [255, 0, 0]  # Azul para área dirigível
    result[ll_mask > 100] = [0, 255, 0]  # Verde para linhas
    
    return result

def main():
    trt_inference = TensorRTInference('model.engine')
    
 
    camera = CSICamera(width=640, height=360, fps=30)
    camera.start()
    
    print("Pressiona 'q' para sair")
    
    fps_counter = 0
    start_time = time.time()
    
    try:
        while True:
            frame = camera.read()
            if frame is None:
                continue
            
            # Preprocessamento
            input_data = preprocess_frame(frame)
            
            # Inferência TensorRT
            outputs = trt_inference.infer(input_data)
            
            # Pós-processamento
            result = postprocess_outputs(outputs, frame)
            
 
            fps_counter += 1
            if fps_counter % 30 == 0:
                elapsed = time.time() - start_time
                fps = 30 / elapsed
                print(f"FPS: {fps:.1f}")
                start_time = time.time()
            
 
            cv2.imshow('Lane Detection', result)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        camera.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

