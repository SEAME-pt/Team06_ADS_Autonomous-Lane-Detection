#!/usr/bin/env python3
import time
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from Jetcar import JetCar
from PIL import Image

IMG_SIZE = (64, 64)

def transform(pil_img):
    pil_img = pil_img.resize(IMG_SIZE)
    img_array = np.array(pil_img).astype(np.float32) / 255.0  # Normaliza
    img_array = np.transpose(img_array, (2, 0, 1))  # HWC → CHW
    return np.expand_dims(img_array, axis=0)  # Adiciona dimensão do batch

class TensorRTRunner:
    def __init__(self, engine_path):
        self.trt_logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.trt_logger)
        
        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
        
        self.input_shape = (1, 3, 64, 64)
        self.output_shape = (1, 1)
        
        self.d_input = cuda.mem_alloc(np.prod(self.input_shape) * np.float32().nbytes)
        self.d_output = cuda.mem_alloc(np.prod(self.output_shape) * np.float32().nbytes)
        
        self.stream = cuda.Stream()

    def infer(self, input_data):
        input_data = input_data.astype(np.float32).ravel()
        output_data = np.zeros(self.output_shape, dtype=np.float32)
        
        cuda.memcpy_htod_async(self.d_input, input_data, self.stream)
        self.context.execute_async_v2(bindings=[int(self.d_input), int(self.d_output)], stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(output_data, self.d_output, self.stream)
        self.stream.synchronize()
        
        return output_data[0][0]

class Controller:
    def __init__(self):
        self.car = JetCar()
        self.car.start()
        time.sleep(0.5)

        self.steering = 0.0
        self.speed = 0.0
        self.max_speed = 0.7

        self.running = True
        self.trt_model = TensorRTRunner("model_light.engine")

        try:
            self.init_camera()
        except Exception as e:
            print(f"ERRO: {e}")
            exit(1)

    def init_camera(self):
        self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            raise Exception("Falha ao abrir câmera")
        print("Câmera inicializada")

    def run(self):
        try:
            while self.running:
                ret, frame = self.camera.read()
                if not ret:
                    print("Erro ao capturar frame")
                    continue

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(rgb)
                input_tensor = transform(pil)

                self.steering = self.trt_model.infer(input_tensor)
                self.steering = max(-1.0, min(1.0, self.steering))
                self.car.set_steering(self.steering)

                cv2.imshow('Main', frame)

                if cv2.waitKey(1) == 27:  # ESC para sair
                    break

        except KeyboardInterrupt:
            print("\nInterrompido")
        finally:
            self.car.set_speed(0)
            self.car.set_steering(0)
            self.car.stop()
            self.camera.release()
            cv2.destroyAllWindows()
            print("ByBy!")

if __name__ == "__main__":
    controller = Controller()
    controller.run()
