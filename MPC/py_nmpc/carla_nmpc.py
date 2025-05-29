import carla
import cv2
import numpy as np
import time
import math
import random
from nmpc import setup_nmpc, compute_control, MAX_STEER, MAX_ACC, MAX_SPEED

# Configurações
WIDTH = 640
HEIGHT = 320
DT = 0.1

# Variáveis globais
current_image = None
current_mask = None
frame_count = 0
y_ref_history = []

def process_mask(mask):
    """Processa a máscara para extrair y_ref e psi_ref com debug."""
    global y_ref_history
    roi_x = int(WIDTH * 0.1)
    roi_width = int(WIDTH * 0.8)
    roi_y = int(HEIGHT * 0.5)
    roi_height = int(HEIGHT * 0.6)
    roi = mask[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]

    edges = cv2.Canny(roi, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30, minLineLength=30, maxLineGap=20)
    
    if lines is not None and len(lines) > 0:
        x_sum, y_sum, angles, count = 0, 0, [], 0
        for line in lines:
            x1, y1, x2, y2 = line[0]
            x_sum += (x1 + x2) / 2
            y_sum += (y1 + y2) / 2
            angle = math.atan2(y2 - y1, x2 - x1)
            angles.append(angle)
            count += 1
        if count > 0:
            x_center = x_sum / count + roi_x
            y_center = y_sum / count + roi_y
            y_ref = (x_center - WIDTH / 2) * 0.005
            y_ref_history.append(y_ref)
            if len(y_ref_history) > 5:
                y_ref_history.pop(0)
            y_ref = np.mean(y_ref_history)
            psi_ref = np.mean(angles) if angles else 0.0
            psi_ref = np.clip(psi_ref, -np.pi / 2, np.pi / 2)
            delta_initial = np.arctan2(y_ref, 10.0)
            return delta_initial, y_ref, psi_ref, True
    return 0.0, 0.0, 0.0, False

def world_to_image(x, y, z, transform, fov=90, width=WIDTH, height=HEIGHT):
    """Converte coordenadas do mundo para imagem com debug."""
    f = width / (2 * math.tan(math.radians(fov / 2)))
    cx, cy = width / 2, height / 2
    rel_x = x - transform.location.x
    rel_y = y - transform.location.y
    rel_z = z - transform.location.z
    if rel_y > 0.1:
        u = cx + f * rel_x / rel_y
        v = cy - f * rel_z / rel_y
        return int(u), int(v)
    return None

def control_vehicle(world, vehicle, solver, n_states, n_controls, lbx, ubx, N):
    """Controla o veículo usando o NMPC."""
    global current_image, current_mask, frame_count
    running = True
    paused = False
    speed_factor = 0.1
    target_speed = MAX_SPEED * speed_factor
    prev_x0 = None

    vehicle_control = carla.VehicleControl()

    while running:
        world.tick()

        if not paused:
            transform = vehicle.get_transform()
            x = transform.location.x
            y = transform.location.y
            psi = np.deg2rad(transform.rotation.yaw)
            velocity = vehicle.get_velocity()
            v = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)

            delta_ref, y_ref, psi_ref, lines_detected = process_mask(current_mask) if current_mask is not None else (0.0, 0.0, 0.0, False)

            state_init = np.array([x, y, psi, v])
            state_ref = np.array([x, y + y_ref, psi + psi_ref, target_speed])

            delta, a, prev_x0, X_pred = compute_control(solver, n_states, n_controls, state_init, state_ref, lbx, ubx, N, prev_x0)

            vehicle_control.steer = delta / MAX_STEER
            vehicle_control.throttle = max(0.0, a / MAX_ACC) if a > 0 else 0.0
            vehicle_control.brake = max(0.0, -a / MAX_ACC) if a < 0 else 0.0
            vehicle.apply_control(vehicle_control)

            if current_image is not None:
                img_display = current_image.copy()
                speed = 3.6 * v
                cv2.putText(img_display, f"Speed: {speed:.1f} km/h", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(img_display, f"Steering: {delta * 180.0 / np.pi:.2f} deg", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(img_display, f"Lines Detected: {lines_detected}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(img_display, f"MPC Delta: {delta * 180.0 / np.pi:.2f} deg", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(img_display, f"Speed factor: {speed_factor:.2f}", (WIDTH - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                if paused:
                    cv2.putText(img_display, "PAUSED (press 'R' to resume)", (10, HEIGHT - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                if X_pred is not None and N < X_pred.shape[1]:
                    x_pred = float(X_pred[0, N])
                    y_pred = float(X_pred[1, N])
                    coords = world_to_image(x_pred, y_pred, transform.location.z, transform)
                    if coords:
                        u, v = coords
                        if 0 <= u < WIDTH and 0 <= v < HEIGHT:
                            cv2.circle(img_display, (u, v), 5, (255, 0, 0), -1)
                            cv2.putText(img_display, "Prediction", (u + 10, v), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                        else:
                            print(f"Coordenadas fora da tela: u={u}, v={v}")
                    else:
                        print(f"Coordenadas inválidas: x_pred={x_pred}, y_pred={y_pred}")

                cv2.imshow('CARLA View', img_display)

                if current_mask is not None:
                    mask_display = cv2.cvtColor(current_mask, cv2.COLOR_GRAY2BGR)
                    roi_x = int(WIDTH * 0.1)
                    roi_width = int(WIDTH * 0.8)
                    roi_y = int(HEIGHT * 0.5)
                    roi_height = int(HEIGHT * 0.6)
                    cv2.rectangle(mask_display, (roi_x, roi_y), (roi_x + roi_width, roi_y + roi_height), (0, 255, 0), 2)
                    cv2.imshow('CARLA Mask', mask_display)

            if lines_detected and frame_count % 10 == 0:
                print(f"Frame {frame_count}: y_ref = {y_ref:.3f} m, psi_ref = {psi_ref:.3f} rad, delta = {delta * 180.0 / np.pi:.3f} deg")

            frame_count += 1

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                running = False
            elif key == ord('p'):
                paused = not paused
                print(f"Navigation {'paused' if paused else 'resumed'}.")
                if paused:
                    vehicle_control.throttle = 0.0
                    vehicle_control.brake = 1.0
                    vehicle.apply_control(vehicle_control)
            elif key == ord('+'):
                speed_factor = min(1.0, speed_factor + 0.05)
                target_speed = MAX_SPEED * speed_factor
                print(f"Speed factor: {speed_factor:.2f}")
            elif key == ord('-'):
                speed_factor = max(0.1, speed_factor - 0.05)
                target_speed = MAX_SPEED * speed_factor
                print(f"Speed factor: {speed_factor:.2f}")

        else:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('r'):
                paused = False
                print("Navigation resumed.")
            elif key == 27:
                running = False

    vehicle_control.throttle = 0.0
    vehicle_control.brake = 1.0
    vehicle.apply_control(vehicle_control)
    print("Vehicle stopped.")

def process_image(image):
    """Processa imagens RGB capturadas pela câmera."""
    global current_image
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    current_image = np.reshape(array, (image.height, image.width, 4))[:, :, :3].copy()

def process_segmentation(image):
    """Processa imagens de segmentação para criar máscara binária (ID 24)."""
    global current_mask
    image.convert(carla.ColorConverter.Raw)
    arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
    mask = (arr[:, :, 2] == 24).astype(np.uint8) * 255
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    current_mask = mask

def main():
    """Configura a simulação CARLA e executa o NMPC."""
    try:
        print("Conectando ao CARLA...")
        client = carla.Client('localhost', 2000)
        client.set_timeout(20.0)
        world = client.load_world("/Game/Carla/Maps/Town10HD_Opt")
        world = client.get_world()

        world.unload_map_layer(carla.MapLayer.ParkedVehicles)
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = DT
        world.apply_settings(settings)

        blueprint = world.get_blueprint_library()
        spawn_points = world.get_map().get_spawn_points()
        vehicle_bp = blueprint.find('vehicle.tesla.model3')
        spawn_point = random.choice(spawn_points)
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)

        camera_transform = carla.Transform(carla.Location(x=2.0, z=1.4), carla.Rotation(pitch=-15))
        cam_rgb = blueprint.find('sensor.camera.rgb')
        cam_rgb.set_attribute('image_size_x', str(WIDTH))
        cam_rgb.set_attribute('image_size_y', str(HEIGHT))
        cam_rgb.set_attribute('fov', '90')
        camera = world.spawn_actor(cam_rgb, camera_transform, attach_to=vehicle)

        cam_seg = blueprint.find('sensor.camera.semantic_segmentation')
        cam_seg.set_attribute('image_size_x', str(WIDTH))
        cam_seg.set_attribute('image_size_y', str(HEIGHT))
        cam_seg.set_attribute('fov', '90')
        seg = world.spawn_actor(cam_seg, camera_transform, attach_to=vehicle)

        camera.listen(process_image)
        seg.listen(process_segmentation)

        print("Configurando NMPC...")
        solver, n_states, n_controls, _, lbx, ubx, N = setup_nmpc()

        print("Iniciando navegação...")
        for i in range(3, 0, -1):
            print(f"Starting in {i}...")
            time.sleep(1)
            world.tick()

        control_vehicle(world, vehicle, solver, n_states, n_controls, lbx, ubx, N)

    except Exception as e:
        print(f"Erro: {e}")
    finally:
        if 'camera' in locals():
            camera.stop()
            camera.destroy()
        if 'seg' in locals():
            seg.stop()
            seg.destroy()
        if 'vehicle' in locals():
            vehicle.destroy()
        if 'world' in locals():
            world.apply_settings(carla.WorldSettings(synchronous_mode=False))
        cv2.destroyAllWindows()
        print(f"Sessão terminada. {frame_count} imagens processadas.")

if __name__ == "__main__":
    main()