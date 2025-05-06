import cv2
import numpy as np
import time
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  
from collections import deque



def gstreamer_pipeline(
    capture_width=320,
    capture_height=240,
    display_width=640,
    display_height=480,
    framerate=30,
    flip_method=0,
):
    return (
        f"nvarguscamerasrc ! "
        f"video/x-raw(memory:NVMM), width=(int){capture_width}, height=(int){capture_height}, "
        f"format=(string)NV12, framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)I420 ! "
        f"videoconvert ! "
        f"video/x-raw, format=(string)BGR ! appsink drop=1 max-buffers=1 sync=false"
    )


class AutonomousCar:
    def __init__(self):
        self.engine_path = "unet_model_256.engine"
        self.img_size = (256, 256)
        self.threshold = 0.5
        self.ghost_timeout = 2.0
        self.min_distance = 40
        self.show_mask = False  # Podes mudar para False
 

        self.point_history = deque(maxlen=5)
        self.angle_history = deque(maxlen=5)

        self.ghost_point = None
        self.ghost_time = 0

        self._init_trt()
        self._init_video()

    def _init_trt(self):
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(self.engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()

        self.input_shape = (1, 3, *self.img_size)
        self.input_binding_idx = self.engine.get_binding_index("input")
        self.output_binding_idx = self.engine.get_binding_index("output")

        self.input_mem = cuda.mem_alloc(int(np.prod(self.input_shape) * np.float32().nbytes))
        self.output_mem = cuda.mem_alloc(int(np.prod((1, 1, *self.img_size)) * np.float32().nbytes))

        self.bindings = [int(self.input_mem), int(self.output_mem)]

        print("TensorRT engine carregado com sucesso")

    def _init_video(self):
        self.cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
        if not self.cap.isOpened():
            raise Exception("Não foi possível abrir o vídeo")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps <= 0:
                self.fps = 30
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.center_x = self.width // 2
        self.base_y = self.height - 10

        print(f"Vídeo inicializado: {self.width}x{self.height}")

    def _preprocess(self, frame):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.img_size)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0).astype(np.float32)
        return np.ascontiguousarray(img)

    def infer_trt(self, frame):
        input_data = self._preprocess(frame)

        cuda.memcpy_htod(self.input_mem, input_data)

        self.context.execute_v2(self.bindings)

        output_data = np.empty((1, 1, *self.img_size), dtype=np.float32)
        cuda.memcpy_dtoh(output_data, self.output_mem)

        mask = (output_data.squeeze() > self.threshold).astype(np.uint8) * 255
        return cv2.resize(mask, (self.width, self.height))

    def encontrar_ponto(self, mask):
        ys, xs = np.where(mask > 0)
        if len(xs) == 0 or len(xs) < 50:
            return None
        sorted_pts = sorted(zip(ys, xs), key=lambda p: abs(p[0] - self.base_y) + abs(p[1] - self.center_x))
        for y, x in sorted_pts:
            if self.base_y - y > self.min_distance:
                return int(x), int(y)
        return None

    def calcular_angulo_suavizado(self, point):
        if point:
            self.point_history.append(point)
        if len(self.point_history) > 0:
            avg_x = sum(p[0] for p in self.point_history) / len(self.point_history)
            avg_y = sum(p[1] for p in self.point_history) / len(self.point_history)
            dx = avg_x - self.center_x
            dy = self.base_y - avg_y
            angle = np.arctan2(dx, dy)
            self.angle_history.append(angle)
            smooth_angle = sum(self.angle_history) / len(self.angle_history)
            return smooth_angle, (avg_x, avg_y)
        return 0, (self.center_x, self.base_y - 60)

    def atualizar_ghost_point(self, point):
        if point:
            self.ghost_point = point
            self.ghost_time = time.time()
            return self.ghost_point, True
        else:
            if self.ghost_point and (time.time() - self.ghost_time < self.ghost_timeout):
                return self.ghost_point, False
            else:
                return (self.center_x, self.base_y - 60), False

    def processar_frame(self, frame):
        start = time.time()
        mask = self.infer_trt(frame)
        point = self.encontrar_ponto(mask)
        target_point, is_real = self.atualizar_ghost_point(point)
        angle, smooth_point = self.calcular_angulo_suavizado(target_point)
        proc_time = time.time() - start

        return {
            'mask': mask,
            'point': point,
            'target_point': target_point,
            'is_real_point': is_real,
            'angle': angle,
            'smooth_point': smooth_point,
            'process_time': proc_time
        }

    def renderizar_visualizacao(self, frame, result):
        overlay = frame.copy()

        if self.show_mask:
            mask_overlay = np.zeros_like(overlay)
            mask_overlay[result['mask'] > 0] = (0, 0, 255)
            overlay = cv2.addWeighted(overlay, 0.7, mask_overlay, 0.9, 0)

        if result['is_real_point']:
            cv2.circle(overlay, result['target_point'], 8, (0, 255, 0), -1)
        else:
            color = (0, 200, 200) if time.time() - self.ghost_time < self.ghost_timeout else (100, 100, 255)
            cv2.circle(overlay, result['target_point'], 8, color, -1)

        cv2.circle(overlay, (int(result['smooth_point'][0]), int(result['smooth_point'][1])), 5, (255, 0, 255), -1)

        angle = result['angle']
        arrow_x = int(self.center_x + 60 * np.sin(angle))
        arrow_y = int(self.base_y - 60 * np.cos(angle))
        cv2.arrowedLine(overlay, (self.center_x, self.base_y), (arrow_x, arrow_y), (0, 255, 255), 4)

        cv2.putText(overlay, f"Angle: {np.degrees(angle):.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(overlay, f"FPS: {1.0 / max(result['process_time'], 0.001):.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(overlay, "Predict: " + ("Real" if result['is_real_point'] else "Ghost"), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return overlay

    def enviar_comando(self, angle):
        steering_angle = np.degrees(angle)
        max_angle = 60.0
        steering_angle = max(min(steering_angle, max_angle), -max_angle)
        return steering_angle

    def executar(self):
        frame_count = 0
        skip_frames = 20  
        while self.cap.isOpened():
            frame_count += 1
            if frame_count % (skip_frames + 1) != 0:
                continue  # Skip este frame
            ret, frame = self.cap.read()
            if not ret:
                print("Can't receive frame. Exiting ...")
                break

            result = self.processar_frame(frame)
            steering_angle = self.enviar_comando(result['angle']) / 255.0
            output_frame = self.renderizar_visualizacao(frame, result)

            cv2.imshow("Main", output_frame)
            if cv2.waitKey(1) == ord("q"):
                break

        self.cap.release()
        cv2.destroyAllWindows()

def main():
    car = AutonomousCar()
    car.executar()

if __name__ == "__main__":
    main()