#!/usr/bin/env python3
import time
import threading
from inputs import get_gamepad
from Jetcar import JetCar
import cv2


def csi_pipeline(width=640, height=480, fps=30,flip_method=0,  sensor_id=0):
    return ('nvarguscamerasrc sensor-id=%d ! '
            'video/x-raw(memory:NVMM), '
            'width=(int)%d, height=(int)%d, '
            'format=(string)NV12, framerate=(fraction)%d/1 ! '
            'nvvidconv flip-method=%d ! '
            'video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! '
            'videoconvert ! '
            'video/x-raw, format=(string)BGR ! appsink' % (sensor_id,
                                                            width, height, fps, flip_method,
                                                            width, height))
class Controller:
    def __init__(self):
        # Inicializar o carro
        self.car = JetCar()
        self.car.start()
        time.sleep(0.5)
        
        # Valores de controle
        self.steering = 0.0  # -1.0 (esquerda) a 1.0 (direita)
        self.speed = 0.0     # -1.0 (ré) a 1.0 (frente)
        self.max_speed = 0.7  # 70% da velocidade máxima
     
 
        self.running = False
        self.gamepad_thread = None
        
 
        try:
            get_gamepad()
            print("Gamepad detectado!")
        except Exception as e:
            print(f"ERRO: Não foi possível detectar o gamepad: {e}")

            exit(1)
    
    def read_gamepad(self):
 
        self.running = True
        
        while self.running:
            try:
  
                events = get_gamepad()
                
                for event in events:
   
                    self._process_event(event)
                    
            except Exception as e:
                print(f"Erro na leitura do gamepad: {e}")
                time.sleep(0.1)  # Evita sobrecarregar a CPU em caso de erros
    def normalize_axis(self, value):
        """Normaliza o valor de um eixo para o intervalo -1.0 a 1.0"""
        # Utiliza o intervalo específico: 0 mínimo, 127 meio, 255 máximo
        center = 127
        range_max = 127
        normalized = (value - center) / range_max
        return max(-1.0, min(1.0, normalized))
    
    def _process_event(self, event):
       
        #print(event.code)
        # Analógico esquerdo (horizontal) - direção
        if event.code == 'ABS_X':
        
            raw_value =self.normalize_axis(event.state)   

            
            # Aplica zona morta e suavização
            if abs(raw_value) < 0.1:
                self.steering = 0.0
            else:
                self.steering = raw_value * abs(raw_value)
            
            # Aplica o comando ao carro
            self.car.set_steering(self.steering)
        
        # Analógico esquerdo (vertical) ou gatilhos - aceleração/freio
        elif event.code == 'ABS_RZ':
            # Normaliza e inverte (para cima = positivo)
            raw_value  = -self.normalize_axis(event.state)
            
            # Aplica zona morta e limita a velocidade máxima
            if abs(raw_value) < 0.05:
                self.speed = 0.0
            else:
                self.speed = raw_value * abs(raw_value) * self.max_speed
            

            self.car.set_speed(self.speed)
        

        elif event.code == 'BTN_EAST' and event.state == 1:  # 1 = pressionado
            self.speed = 0.0
            self.steering = 0.0
            self.car.set_speed(0.0)
            self.car.set_steering(0.0)
        elif event.code=="BTN_NORTH" and event.state == 1:  # 1 = pressionado
            self.speed+=0.025
            self.car.set_speed(self.speed)
        elif event.code=="BTN_SOUTH" and event.state == 1:  # 1 = pressionado
            self.speed-=0.025
            self.car.set_speed(self.speed)

    
    def run(self):
        """Inicia o controlador"""
        print("""
 
- Analógico esquerdo (horizontal): Direciona o carro (esquerda/direita)
- Analógico esquerdo (vertical): Acelera/freia o carro
- Botão B ou X: Para de emergência
- Ctrl+C: Sai do programa
        """)
        

        self.gamepad_thread = threading.Thread(target=self.read_gamepad)
        self.gamepad_thread.daemon = True  # Thread termina quando o programa principal termina
        self.gamepad_thread.start()
        
        try:
            while True:
                print(f"Direção: {self.steering:.2f}  Velocidade: {self.speed:.2f}", end="\r")
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\nPrograma interrompido pelo usuário")
        finally:
            # Limpeza
            self.running = False
            if self.gamepad_thread and self.gamepad_thread.is_alive():
                self.gamepad_thread.join(timeout=1.0)
                
            self.car.set_speed(0)
            self.car.set_steering(0)
            self.car.stop()
            


if __name__ == "__main__":

    # pip install inputs
    

    controller = Controller()
    controller.run()