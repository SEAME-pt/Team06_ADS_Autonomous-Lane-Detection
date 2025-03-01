#!/usr/bin/env python3
import time
import threading
from Jetcar import JetCar
import cv2
import os
import datetime

def gstreamer_pipeline(
        capture_width=400,
        capture_height=400,
        display_width=640,
        display_height=480,
        framerate=30,
        flip_method=0,
    ):
        return (
            "nvarguscamerasrc ! "
            "video/x-raw(memory:NVMM), "
            f"width=(int){capture_width}, height=(int){capture_height}, "
            f"format=(string)NV12, framerate=(fraction){framerate}/1 ! "
            "nvvidconv flip-method=%d ! "
            "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=(string)BGR ! appsink"
            % (flip_method, display_width, display_height)
        )

class Controller:
    def __init__(self):
        self.car = JetCar()
        self.car.start()
        time.sleep(0.5)
        
        self.steering = 0.0
        self.speed = 0.0
        self.max_speed = 0.7
        
        self.running = True
        
        self.is_recording = False
        self.video_writer = None
        self.recording_start_time = None
        
        self.collecting_dataset = False
        self.dataset_dir = None
        self.dataset_images_dir = None
        self.dataset_file = None
        self.frame_count = 0
        
        try:
            self.init_camera()
        except Exception as e:
            print(f"ERRO: {e}")
            exit(1)
    
    def init_camera(self):
        try:
            self.camera = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
            if not self.camera.isOpened():
                raise Exception("Falha ao abrir câmera")
                
            print("Câmera inicializada com sucesso")
            

            self.fps = self.camera.get(cv2.CAP_PROP_FPS)
            if self.fps <= 0:
                self.fps = 30  


        except Exception as e:
            print(f"Erro ao inicializar câmera: {e}")
            self.camera = None


        
    def create_dataset_session(self):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.dataset_dir = f"dataset/session_{timestamp}"
        self.dataset_images_dir = f"{self.dataset_dir}/images"
        os.makedirs(self.dataset_images_dir, exist_ok=True)
        
        self.dataset_file = open(f"{self.dataset_dir}/steering_data.csv", "w")
        self.dataset_file.write("image_path,steering\n")
        
        self.frame_count = 0
        print(f"Nova sessão de dataset criada: {self.dataset_dir}")
        return self.dataset_dir
        
    def toggle_dataset_collection(self):
        if not self.collecting_dataset:
            os.makedirs("dataset", exist_ok=True)
            self.create_dataset_session()
            self.collecting_dataset = True
            print("Iniciando coleta de dados para treino")
        else:
            if self.dataset_file:
                self.dataset_file.close()
            self.collecting_dataset = False
            print(f"Coleta de dados finalizada. Total de frames: {self.frame_count}")
            
    def save_frame_to_dataset(self, frame):
        if not self.collecting_dataset or not self.dataset_file:
            return
            
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        image_filename = f"frame_{timestamp}.jpg"
        image_path = f"{self.dataset_images_dir}/{image_filename}"
        
        cv2.imwrite(image_path, frame)
        
        self.dataset_file.write(f"images/{image_filename},{self.steering:.6f}\n")
        self.dataset_file.flush()
        
        self.frame_count += 1
        
        if self.frame_count % 10 == 0:
            print(f"Frames capturados: {self.frame_count}", end="\r")
    
    def capture_frame_manually(self):
 
        if not self.collecting_dataset or not self.dataset_file:
            print("Dataset não está ativo. Pressiona T para iniciar a coleta.")
            return
            
        ret, frame = self.camera.read()
        if not ret:
            print("Erro: Não foi possível capturar o frame.")
            return
            
 
        current_frame = frame.copy()
            
        # nome único baseado em timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        image_filename = f"frame_{timestamp}.jpg"
        image_path = f"{self.dataset_images_dir}/{image_filename}"
        
 
        cv2.imwrite(image_path, current_frame)
        
        # Registra no CSV
        self.dataset_file.write(f"images/{image_filename},{self.steering:.6f}\n")
        self.dataset_file.flush()
 
        self.frame_count += 1
        print(f"Frame {self.frame_count} capturado - Steering: {self.steering:.2f}")
        
 

    def handle_keyboard(self, key):
        key_char = chr(key & 0xFF).lower()
        
        if key_char == 'a':
            self.steering = max(-1.0, self.steering - 0.1)
            print(f"Direção: {self.steering:.2f} (esquerda)")
        elif key_char == 'd':
            self.steering = min(1.0, self.steering + 0.1)
            print(f"Direção: {self.steering:.2f} (direita)")
        elif key_char == 'c':
            self.steering = 0.0
            print("Direção centrada")
        
        elif key_char == 'w':  # Frente
            self.speed = min(1.0, self.speed + 0.02)
            print(f"Velocidade: {self.speed * self.max_speed:.2f} (frente)")
        elif key_char == 's': 
            self.speed = max(-1.0, self.speed - 0.02)
            print(f"Velocidade: {self.speed * self.max_speed:.2f} (tras)")
        elif key_char == ' ':
            self.speed = 0.0
            print("Velocidade: 0.00 (parado)")
        elif key_char == 't':
            self.toggle_dataset_collection()

        
        actual_speed = self.speed * self.max_speed
        self.car.set_speed(actual_speed)
        self.car.set_steering(self.steering)
    

    
    def run(self):
        try:
            while self.running:
                ret, frame = self.camera.read()
                if not ret:
                    print("Error: Failed to capture frame.")
                    time.sleep(0.1)
                    continue
                
                self.process_frame(frame)
                
                key = cv2.waitKey(1)
                if key != -1:
                    if key == 27:
                        print("\nSaindo...")
                        break
                    elif key == 13:  # Enter key
                        if self.collecting_dataset:
                            self.capture_frame_manually()
                    else:
                        self.handle_keyboard(key)
                
        except KeyboardInterrupt:
            print("\nPrograma interrompido pelo usuário")
        finally:
            self.car.set_speed(0)
            self.car.set_steering(0)
            self.car.stop()
            
            if self.is_recording:
                self.stop_recording()
                
            if self.collecting_dataset and self.dataset_file:
                self.dataset_file.close()
                print(f"Dataset salvo com {self.frame_count} frames")
                
            if self.camera:
                self.camera.release()
                
            cv2.destroyAllWindows()
            print("Sistema finalizado com sucesso")
    
    def process_frame(self, frame):
        current_speed = self.speed * self.max_speed
        current_steering = self.steering
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_color = (0, 255, 0)
        font_thickness = 2
        
        speed_text = f"Velocidade: {current_speed:.2f}"
        cv2.putText(frame, speed_text, (10, 30), font, font_scale, font_color, font_thickness)
        
        steering_text = f"Direcao: {current_steering:.2f}"
        cv2.putText(frame, steering_text, (10, 60), font, font_scale, font_color, font_thickness)

            
        if self.collecting_dataset:
            dataset_text = f"DATASET: {self.frame_count} frames"
            if int(time.time() * 2) % 2 == 0:
                cv2.circle(frame, (30, 130), 10, (255, 0, 0), -1)
            cv2.putText(frame, dataset_text, (45, 135), font, font_scale, (255, 0, 0), font_thickness)
            cv2.putText(frame, "ENTER para capturar frame", (10, 165), font, font_scale, (255, 255, 0), font_thickness)
        
        

        frame_height, frame_width = frame.shape[0], frame.shape[1]
        steering_bar_width = 200
        steering_bar_height = 20
        steering_bar_x = frame_width - steering_bar_width - 10
        steering_bar_y = 30
        

        cv2.rectangle(frame, 
                     (steering_bar_x, steering_bar_y), 
                     (steering_bar_x + steering_bar_width, steering_bar_y + steering_bar_height),
                     (100, 100, 100), -1)  # Cinza
        

        center_x = steering_bar_x + steering_bar_width // 2
        indicator_pos_x = center_x + int(current_steering * (steering_bar_width // 2))
        cv2.rectangle(frame, 
                     (indicator_pos_x - 5, steering_bar_y - 5), 
                     (indicator_pos_x + 5, steering_bar_y + steering_bar_height + 5),
                     (0, 0, 255), -1)  # Vermelho
        

        cv2.line(frame, 
                (center_x, steering_bar_y - 5), 
                (center_x, steering_bar_y + steering_bar_height + 5),
                (255, 255, 255), 1)
                

        speed_bar_width = 20
        speed_bar_height = 150
        speed_bar_x = frame_width - speed_bar_width - 10
        speed_bar_y = 80
        
  
        cv2.rectangle(frame, 
                     (speed_bar_x, speed_bar_y), 
                     (speed_bar_x + speed_bar_width, speed_bar_y + speed_bar_height),
                     (100, 100, 100), -1)  # Cinza
        

        speed_center_y = speed_bar_y + speed_bar_height // 2
        indicator_height = int(current_speed * (speed_bar_height // 2))
        

        if current_speed >= 0:
            speed_color = (0, 255, 0)  
            speed_y_start = speed_center_y
            speed_y_end = speed_center_y - indicator_height
        else:
            speed_color = (0, 0, 255) 
            speed_y_start = speed_center_y
            speed_y_end = speed_center_y - indicator_height
            
        if speed_y_start != speed_y_end: 
            cv2.rectangle(frame, 
                        (speed_bar_x, speed_y_start), 
                        (speed_bar_x + speed_bar_width, speed_y_end),
                        speed_color, -1)
        

        cv2.line(frame, 
                (speed_bar_x - 5, speed_center_y), 
                (speed_bar_x + speed_bar_width + 5, speed_center_y),
                (255, 255, 255), 1)

        controls_text = "Controles: W (frente) | S (tras) | A (esquerda) | D (direita) | C (centralizar) | ESPACO (parar)  | T (dataset) | ESC (sair)"

        text_size = cv2.getTextSize(controls_text, font, font_scale * 0.7, 1)[0]
        cv2.rectangle(frame, 
                     (10, frame_height - 30), 
                     (10 + text_size[0], frame_height - 10),
                     (0, 0, 0), -1)

        cv2.putText(frame, controls_text, (10, frame_height - 15), font, font_scale * 0.7, (255, 255, 255), 1)
        

        current_time = time.time()
        if hasattr(self, 'last_frame_time'):
            fps = 1 / (current_time - self.last_frame_time)
            cv2.putText(frame, f"FPS: {fps:.1f}", (frame_width - 100, 20), font, font_scale, (255, 255, 0), 1)
        self.last_frame_time = current_time
        


        cv2.imshow('Main', frame)


if __name__ == "__main__":
    controller = Controller()
    controller.run()