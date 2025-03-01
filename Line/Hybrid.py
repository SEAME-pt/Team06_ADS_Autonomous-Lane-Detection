import socket
import cv2
import numpy as np
import threading
import time
import logging
from dataclasses import dataclass
from typing import Tuple, Optional, List, Union
from collections import deque
from Jetcar import JetCar

def gstreamer_pipeline(
        capture_width=480,
        capture_height=500,
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


# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('LaneFollower')

@dataclass
class CarConfig:
    base_speed: float = 1.0
    max_steering_angle: float = 90.0
    steering_smoothing: float = 0.75

# ====================  1: CONTROLADOR FUZZY ====================
class FuzzyController:
    """
    Implementação de um controlador Fuzzy simplificado para seguimento de linha.
    Utiliza lógica Fuzzy para determinar o ângulo de direção baseado no erro.
    """
    def __init__(self):
        self.last_output = 0.0
        logger.info("Controlador Fuzzy inicializado")
        
    def compute(self, error: float) -> float:
        """
        Calcula a saída do controlador Fuzzy baseado no erro atual
        Args:
            error: Desvio normalizado do centro da faixa (-1.0 a 1.0)
        Returns:
            Fator de correção para o ângulo de direção
        """
        # Fuzzificação - Categorizando o erro em conjuntos difusos
        error_abs = abs(error)
        
        # Regras de pertinência (membership)
        small_error = max(0, 1 - error_abs * 5)      # Erro pequeno (0 a 0.2)
        medium_error = max(0, min(error_abs * 5 - 1, 3 - error_abs * 5))  # Erro médio (0.2 a 0.6)
        large_error = max(0, error_abs * 5 - 3)      # Erro grande (0.6 a 1.0)
        
        # Base de regras e inferência
        # - Se erro é pequeno: correção pequena
        # - Se erro é médio: correção média
        # - Se erro é grande: correção grande
        small_correction = 0.2
        medium_correction = 0.6
        large_correction = 1.0
        
        # Agregação e defuzzificação (centro de gravidade)
        numerator = (small_error * small_correction + 
                     medium_error * medium_correction + 
                     large_error * large_correction)
        denominator = small_error + medium_error + large_error
        
        if denominator == 0:
            output = 0
        else:
            output = numerator / denominator
            
        # Aplica sinal do erro e suavização
        output = output * (1.0 if error >= 0 else -1.0)
        output = 0.7 * output + 0.3 * self.last_output  # Suavização
        
        self.last_output = output
        return output

# ====================  2: CONTROLADOR PREDITIVO ====================
class PredictiveController:
    """
    Controlador Preditivo que estima a trajetória futura da faixa
    e aplica correções antecipadas baseadas em previsão.
    """
    def __init__(self, history_size=10, prediction_horizon=5):
        self.error_history = deque(maxlen=history_size)
        self.history_size = history_size
        self.prediction_horizon = prediction_horizon
        self.last_output = 0.0
        logger.info(f"Controlador Preditivo inicializado (histórico={history_size}, horizonte={prediction_horizon})")
        
    def compute(self, error: float) -> float:
        """
        Calcula a correção de direção com base na previsão de trajetória
        Args:
            error: Desvio normalizado do centro da faixa (-1.0 a 1.0)
        Returns:
            Fator de correção para o ângulo de direção
        """
        # Armazena o erro atual no histórico
        self.error_history.append(error)
        
        # Se não temos histórico suficiente, retorna resposta proporcional
        if len(self.error_history) < 3:
            return error * 0.5
            
        # Calcula a derivada da tendência (taxa de mudança do erro)
        errors = list(self.error_history)
        error_trend = np.polyfit(range(len(errors)), errors, 1)[0]
        
        # Prevê o erro futuro baseado na tendência atual
        predicted_error = error + error_trend * self.prediction_horizon
        
        # Calcula a saída combinando o erro atual e a previsão
        current_weight = 0.6
        prediction_weight = 0.4
        
        output = current_weight * error + prediction_weight * predicted_error
        
        # Aplica suavização para evitar mudanças bruscas
        smoothed_output = 0.7 * output + 0.3 * self.last_output
        self.last_output = smoothed_output
        
        return smoothed_output
        
# ====================  3: CONTROLADOR BASEADO EM REDES NEURAIS ====================
class NeuralController:
    """
    Controlador baseado em uma pequena rede neural para aprendizado de comportamento.
    Implementação simplificada que simula o comportamento de uma rede treinada.
    """
    def __init__(self):
        # Em uma implementação real, carregaríamos pesos pré-treinados
        # Aqui simulamos o comportamento de uma rede treinada com uma função de resposta
        self.errors = deque(maxlen=5)
        logger.info("Controlador Neural inicializado")
        
    def compute(self, error: float) -> float:
        """
        Calcula a correção de direção usando um modelo neural simplificado
        Args:
            error: Desvio normalizado do centro da faixa (-1.0 a 1.0)
        Returns:
            Fator de correção para o ângulo de direção
        """
        # Armazena histórico de erros para análise temporal
        self.errors.append(error)
        
        # Função de ativação sigmoide para não-linearidade
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
        
        # Em um cenário real, teríamos camadas e pesos da rede
        # Aqui simulamos um comportamento aprendido com base na experiência
        
        error_abs = abs(error)
        error_sign = np.sign(error)
        
        # Características temporais (diferença entre erros consecutivos)
        error_diff = 0
        if len(self.errors) >= 2:
            error_diff = self.errors[-1] - self.errors[-2]
        
        # Camada 1: Feature extraction (simulada)
        f1 = sigmoid(3 * error)
        f2 = sigmoid(2 * error_diff)
        f3 = error_abs * error_abs  # Resposta quadrática
        
        # Camada 2: Combinação (simulada)
        output_linear = 0.5 * f1 + 0.3 * f2 + 0.2 * f3
        
        # Resposta não-linear
        output = error_sign * (1 - np.exp(-3 * error_abs))
        
        # Ajuste adicional baseado na tendência da faixa
        if len(self.errors) >= 3:
            trend = np.mean([self.errors[-1] - self.errors[-2], 
                            self.errors[-2] - self.errors[-3]])
            output += 0.2 * trend
        
        return output

# ====================  4: CONTROLADOR HÍBRIDO ====================
class HybridController:
    """
    Controlador híbrido que combina diferentes estratégias de controle
    dependendo da situação de direção, com foco em manter-se no centro.
    """
    def __init__(self):
        # Componentes de controladores individuais
        self.predictive = PredictiveController(history_size=8, prediction_horizon=3)
        self.fuzzy = FuzzyController()
        
        # Parâmetros PID para situações específicas
        self.pid_params = {
            'straight': {'kp': 0.3, 'ki': 0.005, 'kd': 1.0},  # Ajuste mais sensível para centralização
            'curve': {'kp': 0.4, 'ki': 0.001, 'kd': 2.5}      # Resposta mais rápida em curvas
        }
        
        # Estado do controlador
        self.integral = 0.0
        self.last_error = 0.0
        self.last_time = time.time()
        self.last_output = 0.0
        self.error_history = deque(maxlen=20)
        
        # Parâmetros de centralização
        self.center_deadzone = 0.05  # Zona morta (não ajusta se estiver quase no centro)
        
        logger.info("Controlador Híbrido inicializado com foco em centralização")
        
    def compute(self, error: float) -> float:
        """
        Seleciona e combina diferentes estratégias de controle baseado na situação atual
        Args:
            error: Desvio normalizado do centro da faixa (-1.0 a +1.0)
        Returns:
            Fator de correção para o ângulo de direção
        """
        # Atualiza histórico de erros
        self.error_history.append(error)
        
        # Aplica uma pequena zona morta para evitar oscilações no centro
        if abs(error) < self.center_deadzone:
            error = 0.0
        
        # Determina se estamos em curva ou reta baseado no histórico de erros
        is_curve = False
        if len(self.error_history) >= 5:
            errors = list(self.error_history)[-5:]
            variance = np.var(errors)
            is_curve = variance > 0.05 or abs(error) > 0.3
            
        # Seleciona parâmetros PID baseado na situação
        pid_config = self.pid_params['curve'] if is_curve else self.pid_params['straight']
        
        # Cálculo do PID básico
        dt = time.time() - self.last_time
        if dt <= 0: dt = 0.01
        
        # Componente proporcional - resposta imediata ao erro
        p_term = pid_config['kp'] * error
        
        # Componente integral - correção de erro persistente
        if abs(self.integral) < 30:
            self.integral += error * dt
        i_term = pid_config['ki'] * self.integral
        
        # Componente derivativa - amortecimento/antecipação
        d_term = pid_config['kd'] * (error - self.last_error) / dt
        
        # Saída PID
        pid_output = p_term + i_term + d_term
        
        # Combina PID com outros controladores
        if is_curve:
            # Em curvas, damos mais peso ao controlador preditivo
            predictive_output = self.predictive.compute(error)
            output = 0.5 * pid_output + 0.5 * predictive_output
        else:
            # Em retas, mais foco na centralização suave
            fuzzy_output = self.fuzzy.compute(error)
            output = 0.6 * pid_output + 0.4 * fuzzy_output
        
        # Suavização da resposta para evitar movimentos bruscos
        smoothed_output = 0.8 * output + 0.2 * self.last_output
        
        # Atualiza estado para próxima iteração
        self.last_error = error
        self.last_time = time.time()
        self.last_output = smoothed_output
        
        return smoothed_output

# ==================== DETECTOR DE PISTA  ====================
class AdvancedLaneDetector:
    def __init__(self):
        # Parâmetros de detecção
        self.roi_height_ratio = 0.4  # Proporção da altura para ROI
        self.last_lane_position = None
        self.confidence = 0.0
        self.history = deque(maxlen=5)
        # Adicionando atributos para rastreamento de centro da pista
        self.lane_width_history = deque(maxlen=10)
        self.ideal_center_offset = 0.0  # Offset para o centro ideal (0 = meio exato)
        self.lane_center_history = deque(maxlen=5)
        
    def detect_lane(self, frame):
        if frame is None:
            return None, None
            
        height, width = frame.shape[:2]

        # 1. Detecção por cor (HSV) - ajustado para laranja
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Detecção específica para as linhas laranjas da pista
        lower_orange = np.array([5, 120, 150])
        upper_orange = np.array([25, 255, 255])

        mask = cv2.inRange(hsv, lower_orange, upper_orange)

        # Operações morfológicas para melhorar a máscara
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=2)

        # 2. Região de interesse (ROI) - parte inferior da imagem
        roi_height = int(height * self.roi_height_ratio)
        roi_y_start = height - roi_height
        roi = mask[roi_y_start:height, :]

        # 3. Dividir a imagem para detectar as linhas esquerda e direita separadamente
        mid_x = width // 2
        left_half = np.copy(roi[:, :mid_x])
        right_half = np.copy(roi[:, mid_x:])

        # Encontrar contornos nas duas metades
        left_contours, _ = cv2.findContours(left_half, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        right_contours, _ = cv2.findContours(right_half, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Visualização da máscara processada
        vis_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        # 4. Processamento quando ambas as linhas são detectadas
        if left_contours and right_contours:
            # Encontrar os maiores contornos em cada lado
            left_contour = max(left_contours, key=cv2.contourArea)
            right_contour = max(right_contours, key=cv2.contourArea)
            
            # Calcular os centros dos contornos
            left_M = cv2.moments(left_contour)
            right_M = cv2.moments(right_contour)
            
            if left_M["m00"] > 0 and right_M["m00"] > 0:
                # Calcular os centros de cada linha
                left_cx = int(left_M["m10"] / left_M["m00"])
                right_cx = int(right_M["m10"] / right_M["m00"]) + mid_x
                
                # Calcular largura da pista e armazenar histórico
                lane_width = right_cx - left_cx
                self.lane_width_history.append(lane_width)
                avg_lane_width = np.mean(self.lane_width_history)
                
                # Calcular o ponto central entre as duas linhas (centro atual da pista)
                lane_center_x = (left_cx + right_cx) // 2
                self.lane_center_history.append(lane_center_x)
                
                # Calcular centro alvo (pode incluir offset se desejado)
                target_center_x = width / 2 + (width * self.ideal_center_offset)
                
                # Calcular o desvio do centro normalizado (-1 a +1)
                # Positivo: precisa virar à direita, Negativo: precisa virar à esquerda
                center_deviation = (lane_center_x - target_center_x) / (width / 2)
                
                # Limitar o desvio máximo
                center_deviation = max(min(center_deviation, 1.0), -1.0)
                
                # Desenhar visualizações avançadas
                vis_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                
                # Linha esquerda em azul
                cv2.drawContours(vis_mask[roi_y_start:height, :mid_x], [left_contour], -1, (255, 0, 0), 2)
                
                # Linha direita em vermelho
                adjusted_right_contour = right_contour.copy()
                adjusted_right_contour[:,:,0] += mid_x
                cv2.drawContours(vis_mask[roi_y_start:height, mid_x:], [right_contour], -1, (0, 0, 255), 2)
                
                # Centro atual da pista em verde
                cv2.line(vis_mask, (lane_center_x, roi_y_start), (lane_center_x, height), (0, 255, 0), 2)
                
                # Centro alvo em amarelo
                cv2.line(vis_mask, (int(target_center_x), roi_y_start), (int(target_center_x), height), (0, 255, 255), 2)
                
                # Mostrar a largura da pista
                cv2.putText(vis_mask, f"Largura: {int(avg_lane_width)}px", (10, height - 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Atualizar confiança e histórico
                self.confidence = 1.0
                self.last_lane_position = center_deviation
                self.history.append(center_deviation)
                
                return center_deviation, vis_mask

        # 5. Fallback para caso de falha na detecção das duas linhas

        # Tentar encontrar qualquer contorno significativo na ROI completa
        contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Filtrar contornos por área
            valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 100]
            
            if valid_contours:
                # Usar abordagem original para encontrar o contorno mais relevante
                def contour_score(cnt):
                    area = cv2.contourArea(cnt)
                    M = cv2.moments(cnt)
                    if M["m00"] == 0:
                        return 0
                    cx = int(M["m10"] / M["m00"])
                    center_distance = abs(cx - width/2)
                    return area - (center_distance * 0.5)
                
                contours = sorted(valid_contours, key=contour_score, reverse=True)
                largest_contour = contours[0]
                M = cv2.moments(largest_contour)
                
                if M["m00"] > 0:
                    # Calcular centro do contorno
                    cx = int(M["m10"] / M["m00"])
                    center_deviation = (cx - (width / 2)) / (width / 2)
                    
                    # Desenhar visualização
                    cv2.drawContours(vis_mask[roi_y_start:height, :], [largest_contour], -1, (0, 255, 255), 2)
                    
                    # Reduzir confiança por estar usando apenas um contorno
                    self.confidence = 0.6
                    self.last_lane_position = center_deviation
                    self.history.append(center_deviation)
                    
                    return center_deviation, vis_mask

        # 6. Tentar abordagem alternativa usando Hough Lines
        edges = cv2.Canny(frame[roi_y_start:height, :], 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 30, minLineLength=40, maxLineGap=25)

        if lines is not None:
            # Separar linhas em esquerda e direita
            left_lines = []
            right_lines = []
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Calcular inclinação
                if x2 - x1 != 0:
                    slope = (y2 - y1) / (x2 - x1)
                    # Filtrar linhas horizontais
                    if abs(slope) > 0.1:
                        # Classificar como esquerda ou direita
                        if x1 < width/2 and x2 < width/2:
                            left_lines.append(line)
                        elif x1 > width/2 and x2 > width/2:
                            right_lines.append(line)
            
            # Se detectou linhas em ambos os lados
            if left_lines and right_lines:
                # Calcular centro médio das linhas esquerdas
                left_x = []
                for line in left_lines:
                    x1, y1, x2, y2 = line[0]
                    left_x.append((x1 + x2) / 2)
                left_center_x = np.mean(left_x)
                
                # Calcular centro médio das linhas direitas
                right_x = []
                for line in right_lines:
                    x1, y1, x2, y2 = line[0]
                    right_x.append((x1 + x2) / 2)
                right_center_x = np.mean(right_x)
                
                # Calcular o centro entre as linhas
                center_x = (left_center_x + right_center_x) / 2
                center_deviation = (center_x - (width / 2)) / (width / 2)
                
                # Visualização
                vis_edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
                for line in left_lines:
                    x1, y1, x2, y2 = line[0]
                    cv2.line(vis_edges, (x1, y1), (x2, y2), (255, 0, 0), 2)
                for line in right_lines:
                    x1, y1, x2, y2 = line[0]
                    cv2.line(vis_edges, (x1, y1), (x2, y2), (0, 0, 255), 2)
                
                # Marca o centro calculado
                center_x = int(center_x)
                cv2.line(vis_edges, (center_x, 0), (center_x, edges.shape[0]), (0, 255, 0), 2)
                
                self.confidence = 0.5  # Confiança moderada para detecção por Hough
                self.last_lane_position = center_deviation
                self.history.append(center_deviation)
                
                return center_deviation, vis_edges

        # 7. Se tudo falhar, use o histórico de posições
        if self.history:
            avg_position = np.mean(list(self.history))
            self.confidence *= 0.7  # Reduz confiança progressivamente
            
            # Adiciona texto informativo à máscara
            cv2.putText(vis_mask, "Usando histórico", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            return avg_position, vis_mask
                
        # 8. Última alternativa: sem detecção
        self.confidence = 0
        cv2.putText(vis_mask, "Sem detecção", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return None, vis_mask
# ==================== CLASSE BRIDGE UNITY MODIFICADA ====================
class Bridge:
    def __init__(self,
                 controller_type='hybrid',
                 car_config=CarConfig()):
        
     
    
        # Configurações do carro
        self.car_config = car_config
        
        # Seleção do controlador
        self.controller_type = controller_type
        self.controller = self._create_controller(controller_type)
        
        # Detector de faixa avançado
        self.lane_detector = AdvancedLaneDetector()
        
        # Estado do sistema
        self.current_frame = None
        self.running = True
        self.last_command_time = time.time()
        self.command_rate_limit = 0.01
        
        # Estatísticas
        self.frames_processed = 0
        self.start_time = time.time()
        self.last_fps_update = self.start_time
        self.current_fps = 0
        self.cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
        self.car = JetCar()
        self.car.start()
        time.sleep(2.0)
        self.car.reset()

        
        
      
    
    def _create_controller(self, controller_type):
        controllers = {
            'fuzzy': FuzzyController(),
            'predictive': PredictiveController(),
            'neural': NeuralController(),
            'hybrid': HybridController()
        }
        
        if controller_type not in controllers:
            logger.warning(f"Tipo de controlador '{controller_type}' não reconhecido. Usando híbrido.")
            return controllers['hybrid']
            
        return controllers[controller_type]
    
    def _draw_visualization(self, frame, deviation, steering_angle, speed):
        height, width = frame.shape[:2]
        center_x = int(width/2)
        center_y = height - 50
        
        # Normalizing steering_angle to visualization purposes
        # Convert normalized steering_angle (-1 to 1) to visual angle
        visual_angle = steering_angle * 45  # Use normalized value for display
        
        # Linha de direção
        steer_x = int(center_x + visual_angle * width/180)
        cv2.line(frame, (center_x, center_y), (steer_x, center_y-50), (0, 255, 0), 2)
        cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
        
        # Marcador de desvio
        target_x = width // 4
        cv2.circle(frame, (target_x, height - 20), 5, (255, 0, 0), -1)
        cx = int(target_x + (deviation * width // 2))
        cv2.line(frame, (cx, height), (cx, height - 50), (0, 255, 0), 2)
        
        # Informações de telemetria
        cv2.putText(frame, f"Controlador: {self.controller_type}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)
        cv2.putText(frame, f"Angulo: {steering_angle:.2f}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Velocidade: {speed:.2f}", (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"FPS: {self.current_fps:.1f}", (10, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Trust: {self.lane_detector.confidence:.2f}", (10, 150), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    def process(self):
        frame_count = 0
        frame_skip = 2
        
        while self.running:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    logger.warning("Recebido frame vazio ou corrompido")
                    break
                    
                frame_count += 1
                self.frames_processed += 1
                
                # Atualiza FPS a cada segundo
                current_time = time.time()
                if current_time - self.last_fps_update >= 1.0:
                    self.current_fps = self.frames_processed / (current_time - self.last_fps_update)
                    self.frames_processed = 0
                    self.last_fps_update = current_time
                
                # Detecção de faixa usando detector avançado
                deviation, mask = self.lane_detector.detect_lane(frame)
                
                if deviation is not None:
                    # Calcular ângulo de direção usando o controlador selecionado
                    steering_correction = self.controller.compute(deviation)
                    
                    # Normalize steering between -1 and 1
                    steering_angle = max(min(steering_correction, 1.0), -1.0)
                    
                    # Ajustar velocidade baseado na severidade da curva
                    turn_factor = abs(steering_angle)
                    
                    # Normalize speed between -1 and 1
                    base_speed_normalized = self.car_config.base_speed / 5.0  # Assuming max speed
                    adjusted_speed = base_speed_normalized * (1 - (turn_factor * self.car_config.steering_smoothing))
                    adjusted_speed = max(min(adjusted_speed, 1.0), -1.0)  # Ensure between -1 and 1
                    
                    self.car.drive(adjusted_speed,steering_angle)
                    # Enviar comandos com controle de taxa
                    if frame_count % frame_skip == 0 or current_time - self.last_command_time >= self.command_rate_limit:
                        #self.car.drive(adjusted_speed,steering_angle)
                        self.last_command_time = current_time
                    
                    # Visualização
                    self._draw_visualization(frame, deviation, steering_angle, adjusted_speed)
                else:
                    # Parar quando nenhuma linha é detectada
                    cv2.putText(frame, "Linha perdida - Parando", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    self.car.reset()

            

                self.current_frame = frame
                cv2.imshow('Linha', frame)
                cv2.imshow('Mask', mask)
                
  
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC key
                    self.running = False
                
            except socket.timeout:
                logger.warning("Timeout ao receber imagem")
                continue
            except Exception as e:
                logger.error(f"Erro ao processar imagem: {e}")
                continue
        

    
    def close(self):
        self.running = False
        try:
            self.car.reset()
            self.car.stop()
            cv2.destroyAllWindows()
        except Exception as e:
            logger.error(f"Erro ao encerrar: {e}")


def main():
    """Função principal do programa"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Seguimento de linha com diferentes controladores')
    parser.add_argument('--controller', type=str, default='hybrid',
                      choices=['fuzzy', 'predictive', 'neural', 'hybrid'],
                      help='Tipo de controlador a ser usado (default: hybrid)')
    parser.add_argument('--speed', type=float, default=1.5,
                      help='Velocidade base do carro (default: 1.5)')
    
    args = parser.parse_args()
    
    # Configurações do carro
    car_config = CarConfig(
        base_speed=args.speed,
        max_steering_angle=100.0,
        steering_smoothing=0.3
    )

    # Inicializa a bridge com o controlador escolhido
    bridge = Bridge(
        controller_type=args.controller,
        car_config=car_config
    )
    
    try:
        logger.info(f"Sistema iniciado com controlador {args.controller}. Pressione Ctrl+C para sair.")
        bridge.process()
        # Loop principal
        #while bridge.running:

        #    time.sleep(0.1)
    
    except KeyboardInterrupt:
        logger.info("Interrupção de teclado detectada")
    except Exception as e:
        logger.error(f"Erro não tratado: {e}")
    finally:
        bridge.close()
        logger.info("Programa encerrado.")


if __name__ == "__main__":
    main()
