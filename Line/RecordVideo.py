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
        

        self.steering = 0.0  # -1.0 (esquerda) a 1.0 (direita)
        self.speed = 0.0     # -1.0 (tras) a 1.0 (frente)
        self.max_speed = 0.7  # 70% da velocidade máxima
        

        self.running = True
        

        self.is_recording = False
        self.video_writer = None
        self.recording_start_time = None
        
        try:
            self.init_camera()
        except Exception as e:
            print(f"ERRO: {e}")
            exit(1)
    
    def init_camera(self):
        """Inicializa a câmera"""
        try:
            self.camera = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
            if not self.camera.isOpened():
                raise Exception("Falha ao abrir câmera")
                
            print("Câmera inicializada com sucesso")
            
   
            self.fps = self.camera.get(cv2.CAP_PROP_FPS)
            if self.fps <= 0:
                self.fps = 30  
                
     
            os.makedirs("videos", exist_ok=True)
            self.session = self.create_session()
                
        except Exception as e:
            print(f"Erro ao inicializar câmera: {e}")
            self.camera = None

    def create_session(self):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = f"videos/session_{timestamp}"
        os.makedirs(self.session_dir, exist_ok=True)
        print(f"Nova sessão criada: {self.session_dir}")
        return self.session_dir
    
    def handle_keyboard(self, key):
        """Processa teclas do OpenCV e atualiza controles"""
        # Converte para tecla ASCII
        key_char = chr(key & 0xFF).lower()
        
        # Teclas para controle de direção (esquerda/direita)
        if key_char == 'a':  # Esquerda
            self.steering = max(-1.0, self.steering - 0.1)
            print(f"Direção: {self.steering:.2f} (esquerda)")
        elif key_char == 'd':  # Direita
            self.steering = min(1.0, self.steering + 0.1)
            print(f"Direção: {self.steering:.2f} (direita)")
        elif key_char == 'c':  # Centralizar direção
            self.steering = 0.0
            print("Direção centralizada")
 
        elif key_char == 'w':  # Frente
            self.speed = min(1.0, self.speed + 0.02)
            print(f"Velocidade: {self.speed * self.max_speed:.2f} (frente)")
        elif key_char == 's': 
            self.speed = max(-1.0, self.speed - 0.02)
            print(f"Velocidade: {self.speed * self.max_speed:.2f} (tras)")
        elif key_char == ' ': 
            self.speed = 0.0
            print("Velocidade: 0.00 (parado)")
        

        elif key_char == 'r':
            self.toggle_recording()
        

        actual_speed = self.speed * self.max_speed
        self.car.set_speed(actual_speed)
        self.car.set_steering(self.steering)
    
    def toggle_recording(self):
        """Inicia ou para a gravação de vídeo"""
        if not self.is_recording:

            self.start_recording()
        else:
            self.stop_recording()
    
    def start_recording(self):
        """Inicia a gravação de vídeo"""
        if self.camera is None:
            print("Erro: Câmera não está disponível para gravação")
            return
            
        #  nome de arquivo com timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        video_filename = f"{self.session_dir}/video_{timestamp}.mp4"
        
        #  resolução da câmera
        width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.fps if hasattr(self, 'fps') and self.fps > 0 else 30
     
        video_filename = f"{self.session_dir}/video_{timestamp}.avi"
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        
        self.video_writer = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))
        
        if not self.video_writer.isOpened():
            print(f"Tentando codec alternativo...")
            # Tentar com XVID  compatível em algumas plataformas
            video_filename = f"{self.session_dir}/video_{timestamp}_xvid.avi"
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.video_writer = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))
            
            if not self.video_writer.isOpened():
                #  tentativa com formato raw
                video_filename = f"{self.session_dir}/video_{timestamp}.avi"
                fourcc = cv2.VideoWriter_fourcc(*'IYUV')
                self.video_writer = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))
                
                if not self.video_writer.isOpened():
                    print(f"Erro ao criar arquivo de vídeo: {video_filename}")
                    return
        
        self.is_recording = True
        self.recording_start_time = time.time()
        print(f"\nIniciando gravação: {video_filename}")
    
    def stop_recording(self):
 
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
            
 
            duration = time.time() - self.recording_start_time
            print(f"\nGravação finalizada. Duração: {duration:.1f} segundos")
            
        self.is_recording = False
    
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
                    if key == 27:  # ESC para sair
                        print("\nSaindo...")
                        break
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
                
            if self.camera:
                self.camera.release()
                
            cv2.destroyAllWindows()
            print("ByBy!")
    
    def process_frame(self, frame):

        current_speed = self.speed * self.max_speed
        current_steering = self.steering
        

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_color = (0, 255, 0)  # Verde
        font_thickness = 2
        


        if self.is_recording:
            rec_time = time.time() - self.recording_start_time
            if int(rec_time * 2) % 2 == 0:  
                cv2.circle(frame, (30, 30), 10, (0, 0, 255), -1)
            cv2.putText(frame, f"REC {rec_time:.1f}s", (45, 35), font, font_scale, (0, 0, 255), font_thickness)
        
        if not self.is_recording:
            speed_text = f"Velocidade: {current_speed:.2f}"
            cv2.putText(frame, speed_text, (10, 30), font, font_scale, font_color, font_thickness)
            

            steering_text = f"Direcao: {current_steering:.2f}"
            cv2.putText(frame, steering_text, (10, 60), font, font_scale, font_color, font_thickness)
            
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
                        (0, 0, 255), -1)  
            

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
                        (100, 100, 100), -1)  
            
    
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
            

            controls_text = "Controles: W (frente) | S (tras) | A (esquerda) | D (direita) | C (centralizar) | ESPACO (parar) | R (gravar) | ESC (sair)"

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
            

        if self.is_recording and self.video_writer:
            self.video_writer.write(frame)

        cv2.imshow('Main', frame)


if __name__ == "__main__":
    controller = Controller()
    controller.run()