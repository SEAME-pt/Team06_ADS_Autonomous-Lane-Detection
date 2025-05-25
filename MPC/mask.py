import carla
import cv2
import numpy as np
import time
import random
import sys
import os
import datetime
import math

# Display settings
WIDTH = 1024
HEIGHT = 720

# Dataset
dataset_dir = None
dataset_images_dir = None
frame_count = 0
current_image = None
current_mask = None
direction_forward = True
 
def get_road_waypoints(world, vehicle):
    """Gets a series of waypoints that form a complete road segment"""
    map = world.get_map()
    
    # Start from vehicle's current position
    vehicle_location = vehicle.get_location()
    current_waypoint = map.get_waypoint(vehicle_location)
    
    # Get waypoints on the same road
    waypoints = [current_waypoint]
    
    # Collect waypoints forward
    next_waypoint = current_waypoint
    for i in range(200):  # Collect up to 100 waypoints
        next_waypoints = next_waypoint.next(5.0)  # 5 meters between waypoints
        if not next_waypoints:
            break
        next_waypoint = next_waypoints[0]
        waypoints.append(next_waypoint)
    
    print(f"Collected {len(waypoints)} waypoints")
    return waypoints

def drive_waypoints(world, vehicle, waypoints, direction_forward=True, speed_factor=0.5):
    """
    Drives the vehicle following the defined waypoints.
    Data capture happens manually when user presses Enter.
    
    Parameters:
    - world: CARLA world object
    - vehicle: Vehicle actor to control
    - waypoints: List of waypoints to follow
    - direction_forward: If False, the route is followed in reverse
    - speed_factor: Controls vehicle speed (0.1-1.0)
    
    Returns:
    - True if route completed, False if user exited
    """
    global current_image
    global dataset_dir
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_dir = f"dataset/"
    dataset_images_dir = f"{dataset_dir}/images"
    dataset_mask_dir = f"{dataset_dir}/masks"
    os.makedirs(dataset_images_dir, exist_ok=True)
    os.makedirs(dataset_mask_dir, exist_ok=True)

    frame_count =0
    
    # If direction is reversed, flip the waypoints
    if not direction_forward:
        waypoints = waypoints[::-1]
    
    # Initial settings
    current_waypoint_index = 0
    total_waypoints = len(waypoints)
    vehicle_control = carla.VehicleControl()
    
    # Main loop
    running = True
    paused = False
    
    print("\n=== NAVIGATION STARTED ===")
    print(f"Direction: {'FORWARD' if direction_forward else 'REVERSE'}")
    print("ENTER: Capture frame | P: Pause | +/-: Speed | ESC: Exit")
    
    while running and current_waypoint_index < total_waypoints:
        # Update the world in synchronous mode
        world.tick()
        
        # Get the next waypoint
        target_waypoint = waypoints[current_waypoint_index]
        
        if not paused:
            # Calculate vehicle and waypoint positions
            vehicle_transform = vehicle.get_transform()
            vehicle_location = vehicle_transform.location
            target_location = target_waypoint.transform.location
            
            # Calculate direction and distance
            direction = target_location - vehicle_location
            distance = math.sqrt(direction.x**2 + direction.y**2)
            
            # If close enough, advance to the next waypoint
            if distance < 2.0:  # 2 meters tolerance
                current_waypoint_index += 1
                if current_waypoint_index % 10 == 0:
                    print(f"Waypoint {current_waypoint_index}/{total_waypoints}")
                if current_waypoint_index >= total_waypoints:
                    print("Route completed!")
                    break
                continue
            
            # Calculate angle between vehicle and waypoint
            vehicle_forward = vehicle_transform.get_forward_vector()
            dot = vehicle_forward.x * direction.x + vehicle_forward.y * direction.y
            cross = vehicle_forward.x * direction.y - vehicle_forward.y * direction.x
            angle = math.atan2(cross, dot)
            
            # Convert angle to steering value [-1, 1]
            steering = max(-1.0, min(1.0, angle * 2.0))
            
            # Apply vehicle control
            vehicle_control.throttle = speed_factor
            vehicle_control.steer = steering
            vehicle_control.brake = 0.0
            vehicle.apply_control(vehicle_control)
        
        # Show images
        if current_image is not None:
            img_display = current_image.copy()
            
            # Add information to the image
            velocity = vehicle.get_velocity()
            speed = 3.6 * np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)  # km/h
            current_steering = vehicle_control.steer
            
            # Basic information
            cv2.putText(img_display, f"Speed: {speed:.1f} km/h", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(img_display, f"Steering: {current_steering:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(img_display, f"Waypoint: {current_waypoint_index}/{total_waypoints}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(img_display, f"Direction: {'FORWARD' if direction_forward else 'REVERSE'}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            

            
            # Navigation status
            if paused:
                cv2.putText(img_display, "NAVIGATION PAUSED (press 'R' to continue)", 
                           (10, HEIGHT - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Speed factor
            cv2.putText(img_display, f"Speed factor: {speed_factor:.2f}", 
                       (WIDTH - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Commands
            cv2.putText(img_display, "ENTER: Capture | T: New session | +/-: Speed | ESC: Exit", 
                       (10, HEIGHT - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('CARLA View', img_display)
            
            # Process keys
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC
                running = False
                print("Navigation interrupted by user.")
                return False
            elif key == 13:  # ENTER
                if current_image is not None and current_mask is not None:
                    # Cria pasta se necessário
                    

                    # Gera nome base
                    timestamp = datetime.datetime.now().strftime("%d_%H%M%S")
                    fname = f"frame_{timestamp}"
                    img_path = os.path.join(dataset_images_dir, fname + ".jpg")
                    mask_path = os.path.join(dataset_mask_dir, fname + ".png")

                    # Guarda imagem e máscara
                    cv2.imwrite(img_path, current_image)
                    cv2.imwrite(mask_path, current_mask)
                    print(f"[{frame_count}] Frame e máscara guardados")
                    frame_count += 1
                else:
                    print("[WARN] Imagem ou máscara ainda não disponível.")

            elif key == ord('p') or key == ord('r'):
                # Pause/resume navigation
                paused = not paused if key == ord('p') else False
                print(f"Navigation {'paused' if paused else 'resumed'}.")
                if paused:
                    # Stop the vehicle
                    vehicle_control.throttle = 0.0
                    vehicle_control.brake = 1.0
                    vehicle.apply_control(vehicle_control)
            elif key == ord('+') or key == ord('='):
                # Increase speed
                speed_factor = min(1.0, speed_factor + 0.05)
                print(f"Speed increased to: {speed_factor:.2f}")
            elif key == ord('-') or key == ord('_'):
                # Decrease speed
                speed_factor = max(0.1, speed_factor - 0.05)
                print(f"Speed decreased to: {speed_factor:.2f}")
    
    # Stop the vehicle at the end of the route
    vehicle_control.throttle = 0.0
    vehicle_control.brake = 1.0
    vehicle.apply_control(vehicle_control)
    time.sleep(1)  # Let the vehicle come to a stop
    
    print("Vehicle stopped.")
    return True  # Route completed normally
        
def process_image(image):
    global current_image
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    current_image = np.reshape(array, (image.height, image.width, 4))[:, :, :3].copy()

def process_segmentation_a(image):
    global current_mask
    image.convert(carla.ColorConverter.CityScapesPalette)
    arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))[:, :, :3]
    mask = np.all(arr == [157, 234, 50], axis=-1).astype(np.uint8) * 255
    current_mask = mask

def process_segmentation(image):
    image.convert(carla.ColorConverter.Raw)
    arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
    class_ids = arr[:, :, 2]   
    unique_classes = np.unique(class_ids)
    print("IDs de classes na imagem:", unique_classes)


    mask = (class_ids == 24).astype(np.uint8) * 255
    global current_mask
    current_mask = mask

def custom_colored_segmentation(image):
 
    image.convert(carla.ColorConverter.Raw)
    raw = np.frombuffer(image.raw_data, dtype=np.uint8)
    seg = np.reshape(raw, (image.height, image.width, 4))[:, :, 2]  # Classe = canal azul

    mask = np.zeros((image.height, image.width, 3), dtype=np.uint8)

    color_map = {
        1:  (0, 0, 0),         # Road
        2:  (0, 0, 0),       # Sidewalk
        3:  (0, 0, 0),         # Building
        6:  (0, 0, 0),          # RoadLine
        8:  (0, 0, 0),      # Pole
        9:  (0, 0, 0),          # Traffic Light
        11: (0, 0, 0),          # Vegetation
        14: (0, 0, 0),          # Car
        18: (0, 0, 0),        # Traffic Sign
        20: (0, 0, 0),      # Wall
        21: (0, 0, 0),      # Fence
        22: (0, 0, 0),      # GuardRail
        24: (255, 255, 255),      # Road Lines ;)
    }

    for class_id, color in color_map.items():
        mask[seg == class_id] = color

 

    global current_mask
    current_mask = mask


def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    #world = client.load_world("Town10HD")
    available_maps = client.get_available_maps()
    print("Mapas disponíveis:")
    for map_name in available_maps:
        print(f"  - {map_name}")

    #world = client.load_world("/Game/Carla/Maps/Town05_Opt")
   
    world = client.load_world("/Game/Carla/Maps/Town10HD_Opt") 
    world = client.get_world()
#   - /Game/Carla/Maps/Town10HD_Opt
#   - /Game/Carla/Maps/Town05_Opt
#   - /Game/Carla/Maps/Town02
#   - /Game/Carla/Maps/Town10HD
#   - /Game/Carla/Maps/Town04_Opt
#   - /Game/Carla/Maps/Town05
#   - /Game/Carla/Maps/Town03
#   - /Game/Carla/Maps/Town03_Opt
#   - /Game/Carla/Maps/Town04
#   - /Game/Carla/Maps/Town01_Opt # 1 linha
#   - /Game/Carla/Maps/Town02_Opt #1 linha
#   - /Game/Carla/Maps/Town01

    # Minimizar ambiente
    #world.unload_map_layer(carla.MapLayer.All)
    #world.load_map_layer(carla.MapLayer.Ground)
    #world.load_map_layer(carla.MapLayer.Decals)
    # world.unload_map_layer(carla.MapLayer.Decals)
    # world.unload_map_layer(carla.MapLayer.Props)
    # world.unload_map_layer(carla.MapLayer.StreetLights)
    # world.unload_map_layer(carla.MapLayer.Foliage)
    world.unload_map_layer(carla.MapLayer.ParkedVehicles)
    # world.unload_map_layer(carla.MapLayer.Particles)
    # world.unload_map_layer(carla.MapLayer.Walls)

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    blueprint = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()
    vehicle_bp = blueprint.find('vehicle.tesla.model3')
    spawn_point = random.choice(spawn_points)
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)

    camera_transform = carla.Transform(carla.Location(x=2.0, z=1.4), carla.Rotation(pitch=-15))

    # RGB camera
    cam_rgb = blueprint.find('sensor.camera.rgb')
    cam_rgb.set_attribute('image_size_x', str(WIDTH))
    cam_rgb.set_attribute('image_size_y', str(HEIGHT))
    cam_rgb.set_attribute('fov', '90')
    camera = world.spawn_actor(cam_rgb, camera_transform, attach_to=vehicle)

    # Segmentation camera
    cam_seg = blueprint.find('sensor.camera.semantic_segmentation')
    cam_seg.set_attribute('image_size_x', str(WIDTH))
    cam_seg.set_attribute('image_size_y', str(HEIGHT))
    cam_seg.set_attribute('fov', '90')
    seg = world.spawn_actor(cam_seg, camera_transform, attach_to=vehicle)

    camera.listen(process_image)
    seg.listen(custom_colored_segmentation)

    vehicle.set_autopilot(True)


    cv2.namedWindow("CARLA View", cv2.WINDOW_NORMAL)
    for i in range(3, 0, -1):
	    print(f"Starting in {i}...")
	    time.sleep(1)
	    world.tick()

    try:
        print("[INFO] Pressiona ENTER para capturar, ESC para sair")

        direction_forward = True
        current_speed = 0.5
        loop_count = 0
        
        while True:
            # Find road waypoints from current vehicle position
            print("Finding road waypoints...")
            waypoints = get_road_waypoints(world, vehicle)
            
            if len(waypoints) < 10:
                print("Could not find enough waypoints. Repositioning vehicle...")
                # Try to reposition to a different spawn point
                spawn_point = random.choice(spawn_points)
                #vehicle.set_transform(spawn_point)
                time.sleep(1)
                continue
            
            # Drive the current route
            #print(f"\n=== {'FORWARD' if direction_forward else 'REVERSE'} DIRECTION (Loop {loop_count+1}) ===")
            
            #  returns False if the user pressed ESC
            if not drive_waypoints(world, vehicle, waypoints, direction_forward, current_speed):
                break
            
            # Toggle direction for next iteration
            direction_forward = not direction_forward
            loop_count += 0.5  # Count a complete loop as forward + reverse
            
            print(f"\nRoute completed! Automatically switching direction to {('FORWARD' if direction_forward else 'REVERSE')}")
            
            # Check if we've reached the loop limit
      

    finally:
        camera.stop()
        seg.stop()
        camera.destroy()
        seg.destroy()
        vehicle.destroy()
        world.apply_settings(carla.WorldSettings(synchronous_mode=False))
        cv2.destroyAllWindows()
        print(f"[INFO] Sessão terminada. {frame_count} imagens guardadas.")

if __name__ == "__main__":
    main()
