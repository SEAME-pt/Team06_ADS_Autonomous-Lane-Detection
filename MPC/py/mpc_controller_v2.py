import cv2
import numpy as np
import math
from typing import Tuple, Optional, List
from scipy.optimize import minimize

# Instância global do VehicleModel
vehicle_model = None
# Última reta válida (m, b)
last_valid_line_params = None

class VehicleModel:
    def __init__(self):
        self.width = 640
        self.height = 480
        self.v = np.float32(3.0)  # Speed: 3 m/s
        self.l = np.float32(0.15)  # Wheelbase: 0.15 m
        self.dt = np.float32(0.05)  # Time step: 0.05 s
        self.lines_dist = np.float32(0.25)  # Distance between lines: 0.25 m
        self.meters_per_pixel = np.float32(0.0)
        self.y = np.float32(0.0)
        self.psi = np.float32(0.0)

    def update(self, delta: np.ndarray) -> None:
        delta = np.float32(delta)
        self.y += self.dt * self.v * np.sin(self.psi, dtype=np.float32)
        self.psi += self.dt * (self.v / self.l) * np.tan(delta, dtype=np.float32)

    def convert_shift(self, pixel_shift: float) -> np.ndarray:
        return np.float32(pixel_shift * self.meters_per_pixel)

    def convert_to_pixels(self, y_meters: float) -> float:
        """Converte y (em metros) para x (em pixels) na imagem."""
        return (y_meters / self.meters_per_pixel) + (self.width / 2.0)

    def compute_meters_per_pixel(self, pixel_lane_width: float) -> None:
        if pixel_lane_width > 0:
            self.meters_per_pixel = np.float32(self.lines_dist / pixel_lane_width)

class MPCController:
    def __init__(self, vehicle: VehicleModel):
        self.n = 20  # Horizonte de predição
        self.r = np.float32(0.05)  # Penalidade de controle
        self.delta_max = np.float32(math.pi / 6.0)  # Máximo ângulo de esterçamento
        self.dt = np.float32(0.05)
        self.vehicle = vehicle

    def compute_cost(self, delta_seq: np.ndarray, y_ref_seq: np.ndarray) -> Tuple[float, List[Tuple[float, float]]]:
        delta_seq = np.array(delta_seq, dtype=np.float32)
        y_ref_seq = np.array(y_ref_seq, dtype=np.float32)
        cost = np.float32(0.0)
        y = np.float32(self.vehicle.y)
        psi = np.float32(self.vehicle.psi)
        predicted_path = []

        for k in range(self.n):
            cost += (y - y_ref_seq[k]) ** 2
            cost += self.r * delta_seq[k] ** 2
            # Converter y (metros) para x (pixels)
            x_pixel = self.vehicle.convert_to_pixels(y)
            y_pixel = self.vehicle.height - 1 - k * (self.vehicle.v * self.dt * 10)  # Mesmo ajuste usado em y_ref_seq
            predicted_path.append((x_pixel, y_pixel))
            # Atualizar estados
            y += self.dt * self.vehicle.v * np.sin(psi)
            psi += self.dt * (self.vehicle.v / self.vehicle.l) * np.tan(delta_seq[k])

        return cost, predicted_path

    def optimize(self, y_ref_seq: np.ndarray) -> Tuple[float, List[Tuple[float, float]]]:
        delta_init = np.zeros(self.n, dtype=np.float32)
        bounds = [(-self.delta_max, self.delta_max)] * self.n
        result = minimize(
            fun=lambda delta_seq: self.compute_cost(delta_seq, y_ref_seq)[0],
            x0=delta_init,
            method='SLSQP',
            bounds=bounds,
            options={'disp': False}
        )
        # Calcular o caminho previsto com a sequência otimizada
        _, predicted_path = self.compute_cost(result.x, y_ref_seq)
        return result.x[0], predicted_path

def track_imaginary_center(mask: np.ndarray) -> Tuple[bool, float, float, Optional[np.ndarray], Optional[np.ndarray], bool, Optional[Tuple[float, float]]]:
    """Extrai line_l e line_r, calcula a reta imaginária com base em pontos médios a cada 2 pixels."""
    global last_valid_line_params
    
    # Garantir máscara binária
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    # Detectar contornos
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 20]  # Reduzido para 20
    
    if len(valid_contours) == 0:
        print("No lines detected in mask!")
        return False, 0.0, 0.0, None, None, False, None
    
    # Separar linhas esquerda e direita
    if len(valid_contours) >= 2:
        # Ordenar contornos por x médio
        contours_with_x = [(cnt, np.mean(cnt[:, 0, 0])) for cnt in valid_contours]
        contours_with_x.sort(key=lambda x: x[1])  # Ordem crescente de x
        left_contour = contours_with_x[0][0]
        right_contour = contours_with_x[-1][0]
        
        # Extrair pontos (x, y)
        line_l = left_contour[:, 0, :]  # [[x, y], ...]
        line_r = right_contour[:, 0, :]
        
        # Determinar faixa de y
        y_min = max(np.min(line_l[:, 1]), np.max(line_r[:, 1]))
        y_max = min(np.max(line_l[:, 1]), np.max(line_r[:, 1]))
        
        # Amostrar pontos médios a cada 2 pixels
        x_centers, y_coords = [], []
        for y in range(int(y_min), int(y_max) + 1, 2):
            x_lefts = line_l[line_l[:, 1] == y, 0]
            x_rights = line_r[line_r[:, 1] == y, 0]
            if len(x_lefts) > 0 and len(x_rights) > 0:
                x_center_y = (np.mean(x_lefts) + np.mean(x_rights)) / 2.0
                x_centers.append(x_center_y)
                y_coords.append(y)
        
        if len(x_centers) < 2:  # Pelo menos 2 pontos para polyfit
            print("Insufficient points for polyfit, using fallback.")
            x_left = np.mean(line_l[:, 0])
            x_right = np.mean(line_r[:, 0])
            x_center = (x_left + x_right) / 2.0
            pixel_lane_width = float(x_right - x_left)
            # Estimar uma inclinação mínima com base na diferença vertical
            y_range = y_max - y_min
            if y_range > 0:
                m_est = (x_right - x_left) / y_range if y_range > 0 else 0.0
                b_est = x_center - m_est * (mask.shape[0] - 1)
                last_valid_line_params = (m_est, b_est)
            else:
                last_valid_line_params = (0.0, x_center)
            return True, x_center, pixel_lane_width, line_l, line_r, True, last_valid_line_params
        
        # Ajustar reta aos pontos médios
        try:
            m, b = np.polyfit(y_coords, x_centers, 1)  # x = m*y + b
        except np.linalg.LinAlgError:
            print("Polyfit failed, using fallback.")
            x_left = np.mean(line_l[:, 0])
            x_right = np.mean(line_r[:, 0])
            x_center = (x_left + x_right) / 2.0
            pixel_lane_width = float(x_right - x_left)
            y_range = y_max - y_min
            if y_range > 0:
                m_est = (x_right - x_left) / y_range if y_range > 0 else 0.0
                b_est = x_center - m_est * (mask.shape[0] - 1)
                last_valid_line_params = (m_est, b_est)
            else:
                last_valid_line_params = (0.0, x_center)
            return True, x_center, pixel_lane_width, line_l, line_r, True, last_valid_line_params

        # Calcular x_center na linha inferior (y = height - 1)
        x_center = m * (mask.shape[0] - 1) + b
        # Calcular pixel_lane_width na linha inferior
        x_left = np.mean(line_l[line_l[:, 1] == y_max, 0]) if len(line_l[line_l[:, 1] == y_max]) > 0 else np.mean(line_l[:, 0])
        x_right = np.mean(line_r[line_r[:, 1] == y_max, 0]) if len(line_r[line_r[:, 1] == y_max]) > 0 else np.mean(line_r[:, 0])
        pixel_lane_width = float(x_right - x_left)
        
        last_valid_line_params = (m, b)
        return True, x_center, pixel_lane_width, line_l, line_r, True, (m, b)
    
    elif len(valid_contours) == 1 and last_valid_line_params is not None:
        # Uma linha detectada
        print("Only one line detected, using last valid line parameters.")
        contour = valid_contours[0]
        line_single = contour[:, 0, :]
        x_mean = np.mean(line_single[:, 0])
        if x_mean < mask.shape[1] / 2:
            line_l, line_r = line_single, None
        else:
            line_l, line_r = None, line_single
        m, b = last_valid_line_params
        x_center = m * (mask.shape[0] - 1) + b
        return True, x_center, 200.0, line_l, line_r, True, last_valid_line_params
    
    print("No valid lines or previous line parameters available!")
    return False, 0.0, 0.0, None, None, False, None

def process_mask(mask: np.ndarray) -> Tuple[float, float, float, float, bool, float, Optional[Tuple[float, float]], List[Tuple[float, float]]]:
    """Processa a máscara e retorna ajustes de direção, estados, flag de detecção, centro, parâmetros da reta e trajetória prevista."""
    global vehicle_model
    if vehicle_model is None:
        vehicle_model = VehicleModel()
    
    mpc = MPCController(vehicle_model)

    # Redimensionar máscara para 640x480
    mask = cv2.resize(mask, (640, 480), interpolation=cv2.INTER_NEAREST)
    
    if mask.shape[1] != vehicle_model.width or mask.shape[0] != vehicle_model.height:
        print(f"Invalid mask dimensions! Expected: {vehicle_model.width}x{vehicle_model.height}")
        return 0.0, 0.0, 0.0, 0.0, False, 0.0, None, []

    # Detectar centro imaginário e linhas
    lines_detected, x_center, pixel_lane_width, line_l, line_r, line_valid, line_params = track_imaginary_center(mask)
    
    if not lines_detected:
        return 0.0, vehicle_model.y, vehicle_model.psi, 0.0, False, 0.0, None, []

    vehicle_model.compute_meters_per_pixel(pixel_lane_width)
    # Criar sequência de referência para o horizonte
    if line_params is not None:
        m, b = line_params
        y_ref_seq = np.zeros(mpc.n, dtype=np.float32)
        for k in range(mpc.n):
            y_future = vehicle_model.height - 1 - k * (vehicle_model.v * mpc.dt * 10)
            x_center_future = m * y_future + b
            shift = x_center_future - vehicle_model.width / 2.0
            y_ref_seq[k] = vehicle_model.convert_shift(shift)
    else:
        # Forçar variação em y_ref_seq mesmo no fallback
        x_center_image = vehicle_model.width / 2.0
        shift_base = x_center - x_center_image
        y_ref_seq = np.linspace(vehicle_model.convert_shift(shift_base - 0.1), vehicle_model.convert_shift(shift_base + 0.1), mpc.n, dtype=np.float32)
        print(f"Using fallback y_ref_seq: {y_ref_seq}")

    delta, predicted_path = mpc.optimize(y_ref_seq)
    vehicle_model.update(delta)

    # Confirmação de retorno
    result = (delta, vehicle_model.y, vehicle_model.psi, y_ref_seq[0], lines_detected, x_center, line_params, predicted_path)
    return result