import cv2
import numpy as np
import math

class VehicleModel:
    def __init__(self):
        self.width = 640
        self.height = 480
        self.v = np.float32(3.0)  # Usar float32
        self.l = np.float32(0.15)
        self.dt = np.float32(0.05)
        self.lines_dist = np.float32(0.25)
        self.meters_per_pixel = np.float32(0.0)
        self.y = np.float32(0.0)
        self.psi = np.float32(0.0)

    def update(self, delta):
        delta = np.float32(delta)
        self.y += self.dt * self.v * np.sin(self.psi, dtype=np.float32)
        self.psi += self.dt * (self.v / self.l) * np.tan(delta, dtype=np.float32)

    def convert_shift(self, pixel_shift):
        return np.float32(pixel_shift * self.meters_per_pixel)

    def compute_meters_per_pixel(self, pixel_lane_width):
        if pixel_lane_width > 0:
            self.meters_per_pixel = np.float32(self.lines_dist / pixel_lane_width)

class MPCController:
    def __init__(self, vehicle):
        self.n = 10
        self.r = np.float32(0.05)
        self.delta_max = np.float32(math.pi / 6.0)
        self.dt = np.float32(0.05)
        self.vehicle = vehicle

    def compute_cost(self, delta, y_ref):
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

    def optimize(self, y_ref):
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

def compute_lateral_shift(mask, vehicle):
    """Compute lateral shift from mask image."""
    if mask.shape[1] != vehicle.width or mask.shape[0] != vehicle.height:
        print(f"Invalid mask dimensions! Expected: {vehicle.width}x{vehicle.height}")
        return 0.0

    y_line = vehicle.height - 1  # Line at bottom
    line = mask[y_line, :]

    # Garantir que a máscara seja binária (0 ou 255)
    _, line = cv2.threshold(line, 127, 255, cv2.THRESH_BINARY)
    white_pixels = np.where(line == 255)[0]
    
    if len(white_pixels) == 0:
        print("No lines detected in mask!")
        return 0.0

    x_left = np.min(white_pixels)
    x_right = np.max(white_pixels)
    pixel_lane_width = float(x_right - x_left)  # Garantir float
    vehicle.compute_meters_per_pixel(pixel_lane_width)

    x_center = (x_left + x_right) / 2.0
    x_center_image = vehicle.width / 2.0
    shift = x_center - x_center_image


    
    return shift

def main():
    vehicle = VehicleModel()
    mpc = MPCController(vehicle)

    # Simulate 20 steps (1 second at 20 Hz)
    for k in range(20):
        mask = cv2.imread("../mask/mask_test01.png", cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print("Error loading mask!")
            return -1

        pixel_shift = compute_lateral_shift(mask, vehicle)
        y_ref = vehicle.convert_shift(pixel_shift)

        delta = mpc.optimize(y_ref)
        vehicle.update(delta)

        print(f"Step {k}: y = {vehicle.y:.9f} m, psi = {vehicle.psi:.9f} rad, "
              f"delta = {delta * 180.0 / math.pi:.9f} deg, y_ref = {y_ref:.9f} m")

if __name__ == "__main__":
    main()