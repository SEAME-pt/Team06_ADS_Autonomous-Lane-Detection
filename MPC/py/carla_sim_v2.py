import carla
import cv2
import numpy as np
import time
import random
import sys
import os
import datetime
import math
from mpc_controller import process_mask, vehicle_model

# Configurações de exibição
WIDTH = 1024
HEIGHT = 720

# Variáveis globais
frame_count = 0
current_image = None
current_mask = None
direction_forward = True

def get_road_waypoints(world, vehicle):
    """Gera uma lista de waypoints que formam um segmento completo de estrada."""
    map = world.get_map()
    vehicle_location = vehicle.get_location()
    current_waypoint = map.get_waypoint(vehicle_location)
    waypoints = [current_waypoint]
    
    next_waypoint = current_waypoint
    for i in range(200):
        next_waypoints = next_waypoint.next(5.0)
        if not next_waypoints:
            break
        next_waypoint = next_waypoints[0]
        waypoints.append(next_waypoint)
    
    print(f"Collected {len(waypoints)} waypoints")
    return waypoints

def drive_waypoints(world, vehicle, waypoints, direction_forward=True, speed_factor=0.5):
    """Controla o veículo para seguir os waypoints, usando MPC para direção."""
    global current_image, current_mask, frame_count
    frame_count = 0
    
    if not direction_forward:
        waypoints = waypoints[::-1]
    
    current_waypoint_index = 0
    total_waypoints = len(waypoints)
    vehicle_control = carla.VehicleControl()
    
    running = True
    paused = False
    
    print("\n=== NAVIGATION STARTED ===")
    print(f"Direction: {'FORWARD' if direction_forward else 'REVERSE'}")
    print("ENTER: Capture frame | P: Pause | +/-: Speed | ESC: Exit")
    
    while running and current_waypoint_index < total_waypoints:
        world.tick()
        
        target_waypoint = waypoints[current_waypoint_index]
        
        if not paused:
            vehicle_transform = vehicle.get_transform()
            vehicle_location = vehicle_transform.location
            target_location = target_waypoint.transform.location
            
            direction = target_location - vehicle_location
            distance = math.sqrt(direction.x**2 + direction.y**2)
            
            if distance < 2.0:
                current_waypoint_index += 1
                if current_waypoint_index % 10 == 0:
                    print(f"Waypoint {current_waypoint_index}/{total_waypoints}")
                if current_waypoint_index >= total_waypoints:
                    print("Route completed!")
                    break
                continue
            
            # Calcular direção de fallback
            vehicle_forward = vehicle_transform.get_forward_vector()
            dot = vehicle_forward.x * direction.x + vehicle_forward.y * direction.y
            cross = vehicle_forward.x * direction.y - vehicle_forward.y * direction.x
            angle = math.atan2(cross, dot)
            steering_fallback = max(-1.0, min(1.0, angle * 2.0))
            
            # Inicializar controle
            vehicle_control.throttle = speed_factor
            vehicle_control.steer = steering_fallback
            vehicle_control.brake = 0.0
            
            # Atualizar velocidade do modelo
            vehicle_model.v = np.float32(speed_factor * 3.0)
            
            # Processar máscara com MPC
            if current_image is not None and current_mask is not None:
                delta, y, psi, y_ref, lines_detected, x_center, line_params, predicted_path = process_mask(current_mask)
                
                if lines_detected:
                    delta_max = math.pi / 6.0
                    steering = max(-1.0, min(1.0, delta / delta_max))
                    vehicle_control.steer = steering
                    print(f"Step {frame_count}: y = {y:.9f} m, psi = {psi:.9f} rad, "
                          f"delta = {delta * 180.0 / math.pi:.9f} deg, y_ref = {y_ref:.9f} m")
                    frame_count += 1
                
                vehicle.apply_control(vehicle_control)
            
            # Exibir imagem RGB
            img_display = current_image.copy() if current_image is not None else np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
            
            velocity = vehicle.get_velocity()
            speed = 3.6 * np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
            current_steering = vehicle_control.steer
            
            cv2.putText(img_display, f"Speed: {speed:.1f} km/h", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(img_display, f"Steering: {current_steering:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(img_display, f"Waypoint: {current_waypoint_index}/{total_waypoints}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(img_display, f"Direction: {'FORWARD' if direction_forward else 'REVERSE'}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(img_display, f"MPC Delta: {(delta * 180.0 / math.pi if 'delta' in locals() else 0.0):.2f} deg", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(img_display, f"Lines Detected: {lines_detected if 'lines_detected' in locals() else False}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            if paused:
                cv2.putText(img_display, "NAVIGATION PAUSED (press 'R' to continue)", 
                           (10, HEIGHT - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            cv2.putText(img_display, f"Speed factor: {speed_factor:.2f}", 
                       (WIDTH - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            cv2.putText(img_display, "ENTER: Capture | T: New session | +/-: Speed | ESC: Exit", 
                       (10, HEIGHT - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('CARLA View', img_display)
            
            # Exibir máscara com ROI e trajetória prevista
            if current_mask is not None:
                mask_display = cv2.cvtColor(current_mask, cv2.COLOR_GRAY2BGR)
                roi_x = int(WIDTH * 0.05)
                roi_width = int(WIDTH * 0.90)
                roi_y = int(HEIGHT * 0.5)
                roi_height = int(HEIGHT * 0.6)
                cv2.rectangle(mask_display, (roi_x, roi_y), (roi_x + roi_width, roi_y + roi_height), (0, 255, 0), 2)
                cv2.putText(mask_display, f"Lines Detected: {lines_detected if 'lines_detected' in locals() else False}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                if lines_detected and predicted_path:
                    # Mapear para 1024x720
                    scale_x = WIDTH / 640
                    scale_y = HEIGHT / 480
                    # Desenhar pontos da trajetória prevista
                    for x, y in predicted_path:
                        x_mapped = int(x * scale_x)
                        y_mapped = int(y * scale_y)
                        # Garantir que o ponto esteja dentro da ROI
                        if roi_y <= y_mapped <= roi_y + roi_height and roi_x <= x_mapped <= roi_x + roi_width:
                            cv2.circle(mask_display, (x_mapped, y_mapped), 5, (0, 255, 255), -1)  # Amarelo, preenchido
                
                cv2.imshow('CARLA Mask', mask_display)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:
                running = False
                print("Navigation interrupted by user.")
                return False
            elif key == ord('p') or key == ord('r'):
                paused = not paused if key == ord('p') else False
                print(f"Navigation {'paused' if paused else 'resumed'}.")
                if paused:
                    vehicle_control.throttle = 0.0
                    vehicle_control.brake = 1.0
                    vehicle.apply_control(vehicle_control)
            elif key == ord('+') or key == ord('='):
                speed_factor = min(1.0, speed_factor + 0.05)
                print(f"Speed increased to: {speed_factor:.2f}")
            elif key == ord('-') or key == ord('_'):
                speed_factor = max(0.1, speed_factor - 0.05)
                print(f"Speed decreased to: {speed_factor:.2f}")
    
    vehicle_control.throttle = 0.0
    vehicle_control.brake = 1.0
    vehicle.apply_control(vehicle_control)
    time.sleep(1)
    print("Vehicle stopped.")
    return True

def process_image(image):
    """Processa imagens RGB capturadas pela câmera do veículo."""
    global current_image
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    current_image = np.reshape(array, (image.height, image.width, 4))[:, :, :3].copy()

def process_segmentation(image):
    """Processa imagens de segmentação para criar uma máscara binária (ID 24)."""
    global current_mask
    image.convert(carla.ColorConverter.Raw)
    arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
    class_ids = arr[:, :, 2]
    
    # Máscara binária para ID 24
    mask = (class_ids == 24).astype(np.uint8) * 255
    
    # Fechamento morfológico
    kernel = np.ones((9, 9), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=5)
    
    # Definir ROI
    roi_mask = np.zeros_like(mask)
    roi_x = int(WIDTH * 0.05)
    roi_width = int(WIDTH * 0.90)
    roi_y = int(HEIGHT * 0.5)
    roi_height = int(HEIGHT * 0.6)
    roi_mask[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width] = 255
    
    # Aplicar ROI
    mask = cv2.bitwise_and(mask, roi_mask)
    
    current_mask = mask

def main():
    """Configura a simulação CARLA, inicializa veículo/câmeras e gerencia navegação."""
    print("Iniciando carla_sim.py...")
    try:
        print("Conectando ao CARLA...")
        client = carla.Client('localhost', 2000)
        client.set_timeout(20.0)
        print("Carregando mundo...")
        world = client.load_world("/Game/Carla/Maps/Town05_Opt")
        world = client.get_world()
        
        world.unload_map_layer(carla.MapLayer.ParkedVehicles)
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)
        
        blueprint = world.get_blueprint_library()
        spawn_points = world.get_map().get_spawn_points()
        vehicle_bp = blueprint.find('vehicle.tesla.model3')
        spawn_point = random.choice(spawn_points)
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        
        camera_transform = carla.Transform(carla.Location(x=1.0, z=3), carla.Rotation(pitch=-30))
        
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
        
        vehicle.set_autopilot(False)
        
        cv2.namedWindow("CARLA View", cv2.WINDOW_NORMAL)
        cv2.namedWindow("CARLA Mask", cv2.WINDOW_NORMAL)
        
        for i in range(3, 0, -1):
            print(f"Starting in {i}...")
            time.sleep(1)
            world.tick()
        
        print("[INFO] Pressiona ENTER para capturar, ESC para sair")
        direction_forward = True
        current_speed = 0.5
        loop_count = 0
        
        while True:
            print("Finding road waypoints...")
            waypoints = get_road_waypoints(world, vehicle)
            
            if len(waypoints) < 10:
                print("Could not find enough waypoints. Repositioning vehicle...")
                spawn_point = random.choice(spawn_points)
                time.sleep(1)
                continue
            
            if not drive_waypoints(world, vehicle, waypoints, direction_forward, current_speed):
                break
            
            direction_forward = not direction_forward
            loop_count += 0.5
            print(f"\nRoute completed! Automatically switching direction to {('FORWARD' if direction_forward else 'REVERSE')}")
    
    except Exception as e:
        print(f"Erro: {e}")
    finally:
        try:
            camera.stop()
            seg.stop()
            camera.destroy()
            seg.destroy()
            vehicle.destroy()
        except:
            pass
        world.apply_settings(carla.WorldSettings(synchronous_mode=False))
        cv2.destroyAllWindows()
        print(f"[INFO] Sessão terminada. {frame_count} imagens processadas.")

if __name__ == "__main__":
    main()