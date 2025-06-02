import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from threading import Thread, Lock, Event
import time
from collections import deque
import queue

class FPSCounter:
    def __init__(self, window_size=30):
        self.window_size = window_size
        self.frame_times = deque(maxlen=window_size)
        self.last_time = time.time()
        
    def update(self):
        current_time = time.time()
        self.frame_times.append(current_time - self.last_time)
        self.last_time = current_time
        
    def get_fps(self):
        if len(self.frame_times) < 2:
            return 0.0
        avg_frame_time = sum(self.frame_times) / len(self.frame_times)
        return 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0

class BufferedCSICamera:
    def __init__(self, width=640, height=360, fps=30, buffer_size=10):
        self.width = width
        self.height = height
        self.fps = fps
        self.buffer_size = buffer_size
        
        # Pipeline otimizado para CSI camera (Jetson)
        self.pipeline = (
            f"nvarguscamerasrc ! "
            f"video/x-raw(memory:NVMM), width={width}, height={height}, "
            f"format=NV12, framerate={fps}/1 ! "
            f"nvvidconv flip-method=0 ! "
            f"video/x-raw, width={width}, height={height}, format=BGRx ! "
            f"videoconvert ! "
            f"video/x-raw, format=BGR ! "
            f"queue max-size-buffers=1 leaky=downstream ! "  # Buffer interno
            f"appsink drop=true max-buffers=1"  # Drop frames antigos
        )
        
        # Buffer circular thread-safe
        self.frame_buffer = queue.Queue(maxsize=buffer_size)
        self.current_frame = None
        self.frame_lock = Lock()
        
        # Controle de threads
        self.capture_thread = None
        self.running = False
        self.stop_event = Event()
        
        # Estatísticas
        self.frames_captured = 0
        self.frames_dropped = 0
        self.buffer_overflows = 0
        
        # Inicializar câmera
        self.cap = None
        self._init_camera()
        
    def _init_camera(self):
        """Inicializa a câmera com retry"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.cap = cv2.VideoCapture(self.pipeline, cv2.CAP_GSTREAMER)
                
                if not self.cap.isOpened():
                    raise Exception("Falha ao abrir câmera")
                
                # Configurações otimizadas
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Buffer mínimo no OpenCV
                self.cap.set(cv2.CAP_PROP_FPS, self.fps)
                
                print(f"Câmera CSI inicializada: {self.width}x{self.height}@{self.fps}fps")
                return
                
            except Exception as e:
                print(f"Tentativa {attempt + 1} falhou: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
                else:
                    raise Exception("Falha ao inicializar câmera após múltiplas tentativas")
    
    def _capture_loop(self):
        """Loop de captura em thread separada"""
        print("Thread de captura iniciada")
        
        while not self.stop_event.is_set():
            ret, frame = self.cap.read()
            
            if not ret or frame is None:
                self.frames_dropped += 1
                continue
            
            self.frames_captured += 1
            
            # Atualizar frame atual (sempre o mais recente)
            with self.frame_lock:
                self.current_frame = frame.copy()
            
            # Tentar adicionar ao buffer (não-bloqueante)
            try:
                # Se buffer cheio, remove o mais antigo
                if self.frame_buffer.full():
                    try:
                        self.frame_buffer.get_nowait()  # Remove frame antigo
                        self.buffer_overflows += 1
                    except queue.Empty:
                        pass
                
                self.frame_buffer.put_nowait({
                    'frame': frame,
                    'timestamp': time.time(),
                    'frame_id': self.frames_captured
                })
                
            except queue.Full:
                self.buffer_overflows += 1
        
        print("Thread de captura finalizada")
    
    def start(self):
        """Inicia captura threaded"""
        if self.running:
            return
        
        self.running = True
        self.stop_event.clear()
        self.capture_thread = Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        
        # Aguarda primeiro frame
        timeout = 5.0
        start_wait = time.time()
        while self.current_frame is None and (time.time() - start_wait) < timeout:
            time.sleep(0.01)
        
        if self.current_frame is None:
            raise Exception("Timeout aguardando primeiro frame")
        
        print("Captura buffered iniciada")
    
    def read(self):
        """Lê frame mais recente (não-bloqueante)"""
        with self.frame_lock:
            return self.current_frame.copy() if self.current_frame is not None else None
    
    def read_buffered(self, timeout=0.1):
        """Lê frame do buffer (com timeout)"""
        try:
            frame_data = self.frame_buffer.get(timeout=timeout)
            return frame_data['frame'], frame_data['timestamp'], frame_data['frame_id']
        except queue.Empty:
            return None, None, None
    
    def get_buffer_size(self):
        """Retorna tamanho atual do buffer"""
        return self.frame_buffer.qsize()
    
    def get_stats(self):
        """Retorna estatísticas da câmera"""
        return {
            'frames_captured': self.frames_captured,
            'frames_dropped': self.frames_dropped,
            'buffer_overflows': self.buffer_overflows,
            'buffer_size': self.get_buffer_size(),
            'buffer_max': self.buffer_size
        }
    
    def clear_buffer(self):
        """Limpa buffer (útil para reduzir latência)"""
        while not self.frame_buffer.empty():
            try:
                self.frame_buffer.get_nowait()
            except queue.Empty:
                break
    
    def stop(self):
        """Para captura e limpa recursos"""
        if not self.running:
            return
        
        print("Parando câmera buffered...")
        self.running = False
        self.stop_event.set()
        
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)
        
        if self.cap:
            self.cap.release()
        
        # Limpar buffer
        while not self.frame_buffer.empty():
            try:
                self.frame_buffer.get_nowait()
            except queue.Empty:
                break
        
        print("Câmera buffered parada")

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
        np.copyto(self.inputs[0]['host'], input_data.ravel())
        cuda.memcpy_htod(self.inputs[0]['device'], self.inputs[0]['host'])
        
        self.context.execute_v2(bindings=self.bindings)
        
        outputs = []
        for output in self.outputs:
            cuda.memcpy_dtoh(output['host'], output['device'])
            outputs.append(output['host'])
            
        return outputs

def preprocess_frame(frame):
    img = cv2.resize(frame, (640, 360))
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR para RGB e HWC para CHW
    img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
    return img

def postprocess_outputs(outputs, original_frame):
    drivable_area = outputs[0].reshape(1, 2, 360, 640)
    lane_lines = outputs[1].reshape(1, 2, 360, 640)
    
    da_predict = np.argmax(drivable_area, axis=1)[0]
    ll_predict = np.argmax(lane_lines, axis=1)[0]
    
    h, w = original_frame.shape[:2]
    da_mask = cv2.resize((da_predict * 255).astype(np.uint8), (w, h))
    ll_mask = cv2.resize((ll_predict * 255).astype(np.uint8), (w, h))
    
    result = original_frame.copy()
    result[da_mask > 100] = [255, 0, 0]  # Azul para área dirigível
    result[ll_mask > 100] = [0, 255, 0]  # Verde para linhas
    
    return result

def main():
    # Modo de leitura: 'latest' (mais recente) ou 'buffered' (do buffer)
    READ_MODE = 'latest'  # ou 'buffered'
    
    trt_inference = TensorRTInference('model2.engine')
    
    # Câmera buffered com buffer de 5 frames
    camera = BufferedCSICamera(width=640, height=360, fps=30, buffer_size=5)
    
    fps_counter = FPSCounter(window_size=30)
    
    print(f"Modo de leitura: {READ_MODE}")
    print("Pressiona Ctrl+C para sair")
    
    frames_processed = 0
    start_time = time.time()
    
    try:
        camera.start()
        time.sleep(1)  # Aguarda estabilizar
        
        while True:
            if READ_MODE == 'latest':
                # Lê frame mais recente (menor latência)
                frame = camera.read()
                frame_timestamp = time.time()
                frame_id = 0
            else:
                # Lê do buffer (pode ter latência maior)
                frame, frame_timestamp, frame_id = camera.read_buffered(timeout=0.1)
            
            if frame is None:
                continue
            
            # Calcular latência (se usando buffer)
            if READ_MODE == 'buffered' and frame_timestamp:
                latency = (time.time() - frame_timestamp) * 1000  # ms
            else:
                latency = 0
            
            process_start = time.time()
            
            input_data = preprocess_frame(frame)
            outputs = trt_inference.infer(input_data)
            result = postprocess_outputs(outputs, frame)
            
            process_time = (time.time() - process_start) * 1000  # ms
            
            frames_processed += 1
            fps_counter.update()
            
            # Relatório a cada 30 frames
            if frames_processed % 30 == 0:
                elapsed = time.time() - start_time
                avg_fps = frames_processed / elapsed
                current_fps = fps_counter.get_fps()
                stats = camera.get_stats()
                
                print(f"Frames: {frames_processed:4d} | "
                      f"FPS: {current_fps:5.1f} | "
                      f"Proc: {process_time:4.1f}ms | "
                      f"Buf: {stats['buffer_size']}/{stats['buffer_max']} | "
                      f"Cap: {stats['frames_captured']} | "
                      f"Drop: {stats['frames_dropped']} | "
                      f"Overflow: {stats['buffer_overflows']}" +
                      (f" | Lat: {latency:.1f}ms" if READ_MODE == 'buffered' else ""))
                      
    except KeyboardInterrupt:
        print("\nParando...")
        
    finally:
        total_time = time.time() - start_time
        final_fps = frames_processed / total_time
        final_stats = camera.get_stats()
        
        print(f"\n=== ESTATÍSTICAS FINAIS ===")
        print(f"Tempo total: {total_time:.1f}s")
        print(f"Frames processados: {frames_processed}")
        print(f"FPS médio: {final_fps:.1f}")
        print(f"Frames capturados: {final_stats['frames_captured']}")
        print(f"Frames perdidos: {final_stats['frames_dropped']}")
        print(f"Buffer overflows: {final_stats['buffer_overflows']}")
        
        camera.stop()

if __name__ == "__main__":
    main()