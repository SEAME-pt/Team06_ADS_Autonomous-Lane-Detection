import carla
import numpy as np
import cv2
import time
import os

# Connect to CARLA
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.load_world("/Game/Carla/Maps/Town10HD_Opt")
world = client.get_world()
print(f"Connected to CARLA, map: {world.get_map().name}")

# Set synchronous mode
settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = 0.05
world.apply_settings(settings)
time.sleep(2.0)

# Spawn vehicle (static)
blueprint_library = world.get_blueprint_library()
vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
spawn_points = world.get_map().get_spawn_points()
vehicle = None
for i, point in enumerate(spawn_points[:5]):
    try:
        vehicle = world.spawn_actor(vehicle_bp, point)
        print(f"Vehicle spawned at point {i}: ID={vehicle.id}")
        break
    except RuntimeError as e:
        print(f"Failed at point {i}: {e}")
if vehicle is None:
    raise RuntimeError("Could not spawn vehicle.")

# Add RGB camera
camera_bp = blueprint_library.find('sensor.camera.rgb')
camera_bp.set_attribute('image_size_x', '400')
camera_bp.set_attribute('image_size_y', '300')
camera_bp.set_attribute('fov', '90')
camera_transform = carla.Transform(carla.Location(x=2.0, z=1.4), carla.Rotation(pitch=-15))
camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
print(f"Camera spawned: ID={camera.id}")

# Initialize OpenCV window
cv2.namedWindow("Camera Feed", cv2.WINDOW_AUTOSIZE)
cv2.resizeWindow("Camera Feed", 400, 300)

# Process images
def process_image(image):
    print(f"Image: frame={image.frame}, size={image.width}x{image.height}")
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))
    array = array[:, :, :3]  # Convert to BGR
    cv2.imshow("Camera Feed", array)
    if cv2.getWindowProperty("Camera Feed", cv2.WND_PROP_VISIBLE) < 1:
        print("Window closed by user.")
        raise KeyboardInterrupt
    cv2.waitKey(10)
    os.makedirs("camera_debug", exist_ok=True)
    cv2.imwrite(f"camera_debug/frame_{image.frame}.png", array)
camera.listen(process_image)

# Main loop (no movement)
try:
    while True:
        world.tick()
        time.sleep(0.05)

except KeyboardInterrupt:
    pass

finally:
    settings.synchronous_mode = False
    world.apply_settings(settings)
    cv2.destroyAllWindows()
    camera.destroy()
    vehicle.destroy()
    print("Vehicle and camera destroyed.")