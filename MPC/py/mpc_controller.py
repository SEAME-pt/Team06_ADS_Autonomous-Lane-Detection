import cv2
import numpy as np
import math
from typing import Tuple, Optional

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

    def compute_meters_per_pixel(self, pixel_lane_width: float) -> None:
        if pixel_lane_width > 0:
            self.meters_per_pixel = np.float32(self.lines_dist / pixel_lane_width)

class MPCController:
    def __init__(self, vehicle: VehicleModel):
        self.n = 10
        self.r = np.float32(0.05)
        self.delta_max = np.float32(math.pi / 6.0)
        self.dt = np.float32(0.05)
        self.vehicle = vehicle

    def compute_cost(self, delta: np.ndarray, y_ref: float) -> float:
        delta = np.array(delta, dtype=np.float32)
        y_ref = np.float32(y_ref)
        cost = np.float32(0.0)
        y = np.float32(self.vehicle.y)
        psi = np.float32(self.vehicle.psi)

        for k in range(self.n):
            cost += (y - y_ref) ** 2 + self.r * delta[k] ** 2
            y += self.dt * self.vehicle.v * np.sin(psi)
            psi += self.dt * (self.vehicle.v / self.vehicle.l) * np.tan(delta[k])
        return cost

    def optimize(self, y_ref: float) -> float:
        delta = np.float32(0.0)
        learning_rate = np.float32(0.02)
        max_iterations = 50

        for _ in range(max_iterations):
            delta_seq = np.full(self.n, delta, dtype=np.float32)
            cost = self.compute_cost(delta_seq, y_ref)

            delta_plus = delta_seq.copy()
            delta_plus[0] = np.minimum(np.maximum(delta + np.float32(0.01), -self.delta_max), self.delta_max)
            cost_plus = self.compute_cost(delta_plus, y_ref)

            delta_minus = delta_seq.copy()
            delta_minus[0] = np.minimum(np.maximum(delta - np.float32(0.01), -self.delta_max), self.delta_max)
            cost_minus = self.compute_cost(delta_minus, y_ref)

            gradient = (cost_plus - cost_minus) / np.float32(0.02)
            delta -= learning_rate * gradient
            delta = np.minimum(np.maximum(delta, -self.delta_max), self.delta_max)

        return delta

def track_imaginary_center(mask: np.ndarray) -> Tuple[bool, float, float, Optional[np.ndarray], Optional[np.ndarray], Optional[Tuple[float, float]]]:
    """Extrai line_l e line_r, calcula a reta imaginária com base em pontos médios a cada 10 pixels."""
    global last_valid_line_params
    
    # Garantir máscara binária
    _, mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
    
    # Detectar contornos
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 0]
    
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
        y_min = max(int(np.min(line_l[:, 1])), int(np.min(line_r[:, 1])))
        y_max = min(int(np.max(line_l[:, 1])), int(np.max(line_r[:, 1])))
        
        # Amostrar pontos médios a cada 10 pixels
        x_centers, y_coords = [], []
        for y in range(y_min, y_max + 1, 10):
            x_lefts = line_l[line_l[:, 1] == y, 0]
            x_rights = line_r[line_r[:, 1] == y, 0]
            if len(x_lefts) > 0 and len(x_rights) > 0:
                x_center_y = (np.mean(x_lefts) + np.mean(x_rights)) / 2.0
                x_centers.append(x_center_y)
                y_coords.append(y)
        
        if not x_centers:
            # Fallback se não houver pontos válidos
            x_left = np.mean(line_l[:, 0])
            x_right = np.mean(line_r[:, 0])
            x_center = (x_left + x_right) / 2.0
            pixel_lane_width = float(x_right - x_left)
            last_valid_line_params = (0.0, x_center)  # Reta vertical
            return True, x_center, pixel_lane_width, line_l, line_r, True, last_valid_line_params
        
        # Ajustar reta aos pontos médios
        m, b = np.polyfit(y_coords, x_centers, 1)  # x = m*y + b
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

def process_mask(mask: np.ndarray) -> Tuple[float, float, float, float, bool, float, Optional[Tuple[float, float]]]:
    """Processa a máscara e retorna ajustes de direção, estados, flag de detecção, centro e parâmetros da reta."""
    global vehicle_model
    if vehicle_model is None:
        vehicle_model = VehicleModel()
    
    mpc = MPCController(vehicle_model)

    # Redimensionar máscara para 640x480
    mask = cv2.resize(mask, (640, 480), interpolation=cv2.INTER_NEAREST)
    
    if mask.shape[1] != vehicle_model.width or mask.shape[0] != vehicle_model.height:
        print(f"Invalid mask dimensions! Expected: {vehicle_model.width}x{vehicle_model.height}")
        return 0.0, 0.0, 0.0, 0.0, False, 0.0, None

    # Detectar centro imaginário e linhas
    lines_detected, x_center, pixel_lane_width, line_l, line_r, line_valid, line_params = track_imaginary_center(mask)
    
    if not lines_detected:
        return 0.0, vehicle_model.y, vehicle_model.psi, 0.0, False, 0.0, None

    vehicle_model.compute_meters_per_pixel(pixel_lane_width)
    x_center_image = vehicle_model.width / 2.0
    shift = x_center - x_center_image
    y_ref = vehicle_model.convert_shift(shift)
    delta = mpc.optimize(y_ref)
    vehicle_model.update(delta)

    return delta, vehicle_model.y, vehicle_model.psi, y_ref, True, x_center, line_params