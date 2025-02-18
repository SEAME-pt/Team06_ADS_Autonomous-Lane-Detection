import cv2
import numpy as np
import time
from JetCar import JetCar


def gstreamer_pipeline(
    sensor_id=0,
    capture_width=640,
    capture_height=720,
    display_width=640,
    display_height=480,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )


class PID:
    def __init__(self, kp=0.25, ki=0.006, kd=2.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.previous_error = 0.0
        self.integral = 0.0
        self.last_time = time.time()
        self.need_reset = False

    def compute(self, error):
        if self.need_reset:
            self.previous_error = error
            self.integral = 0.0
            self.need_reset = False
            
        self.integral += error
        current_time = time.time()
        dt = current_time - self.last_time
        if dt <= 0:
            dt = 0.1
            
        derivative = (error - self.previous_error) / dt
        self.previous_error = error
        self.last_time = current_time
        
        return self.kp * error + self.ki * self.integral + self.kd * derivative

def process_image(frame):
    height, width = frame.shape[:2]
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Pode ajudar na precisão da cor
    hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)
    lower_orange = np.array([10, 100, 100])
    upper_orange = np.array([25, 255, 255])
    mask = cv2.inRange(hsv, lower_orange, upper_orange)
    
    # Apply morphological operations to clean the mask
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)
    
    # Define ROI in the lower part of the image
    roi_height = int(height * 0.3)
    roi = mask[height - roi_height:height, :]
    
    # Find contours in the ROI
    contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, mask
    
    # Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Calculate the center of the contour
    M = cv2.moments(largest_contour)
    if M["m00"] == 0:
        return None, mask
    cx = int(M["m10"] / M["m00"])
    
    # Calculate deviation from the center
    center_deviation = (cx - (width // 2)) / (width // 2)
    
    return center_deviation, mask

def interpolate(value_current, value_previous, alpha=0.5):
    return alpha * value_current + (1 - alpha) * value_previous

def main():
    car = JetCar()
    car.start()
    
    BASE_SPEED = 40
    DIFF = 160.0
    car.set_speed(0)
    

    cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    # Initialize PID controller
    #pid = PID(kp=0.25, ki=0.006, kd=2.0)
    pid = PID(kp=0.25, ki=0.005, kd=0.8)   
    
    frame_count = 0
    frame_skip = 15
    last_angle = 0
    alpha = 0.8
    last_steering_angle = 0  
    last_speed=0
    reduce = 0.55
    
    last_time = time.time()

    try:
        while True:
            current_time = time.time()
            dt = current_time - last_time
            if dt < 0.05:  # Limita a taxa de atualização a 20 FPS (~1/0.05s)
                time.sleep(0.05 - dt)  
            last_time = time.time()

            print(f"Tempo entre frames: {dt:.4f} segundos") 
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame.")
                break
            
            deviation, mask = process_image(frame)
            
            if deviation is not None:
            
                # Compute steering correction using PID
                steering_correction = pid.compute(deviation)
                
                # Convert PID output to steering angle (-90 to 90 degrees)
                steering_angle = int(steering_correction * DIFF)  
                current_steering_angle = int(steering_correction * DIFF)
                MAX_ANGLE = 120  # para evitar o turn  exagerado
                steering_angle = max(-MAX_ANGLE, min(MAX_ANGLE, steering_angle))
                
            
                smoothed_steering_angle = interpolate(current_steering_angle, last_steering_angle, alpha)
                
                last_steering_angle = smoothed_steering_angle
          
                
                # Apply steering and adjust speed
                #car.set_steering(smoothed_steering_angle)

                turn_factor = abs(steering_angle) / DIFF
                adjusted_speed = BASE_SPEED * (1 - (turn_factor *reduce))
       

                # if adjusted_speed<=-20:
                #     adjusted_speed =-20

                last_speed = adjusted_speed
                car.set_speed(adjusted_speed)
                
                # Visualize the line and center
                height, width = frame.shape[:2]
                cx = int((deviation * (width // 2)) + (width // 2))
                cv2.circle(frame, (cx, height//2), 5, (255, 0, 0), -1)
                cv2.line(frame, (width // 2, height//2), (cx, height - 50), (0, 255, 0), 2)
                cv2.putText(frame, f"Angle: {steering_angle}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Speed: {adjusted_speed:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                # Line lost, stop the car
                print(last_speed)
                car.set_speed(last_speed)
                car.set_steering(last_steering_angle)
                pid.need_reset = True
                cv2.putText(frame, "Line Lost - Stopping", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
            cv2.imshow('Line Following', frame)
            cv2.imshow('Mask', mask)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        cap.release()
        car.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
