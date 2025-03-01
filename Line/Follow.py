import socket
import cv2
import numpy as np
import threading
import time
import numpy as np
import time
import cv2
from Jetcar import JetCar
from TrackController import TrackController


def gstreamer_pipeline(
        capture_width=400,
        capture_height=400,
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
class Car:
    def __init__(self):
        self.speed_history = [] 
        self.steering_history = []  
        self.max_history = 3  
        self.smoothing_threshold = 0.15  
        self.racer = JetCar()
        self.racer.start()
        self.racer.set_speed(0)
        time.sleep(2.0)

    def smooth_value(self, history, new_value):
        if history and abs(history[-1] - new_value) < self.smoothing_threshold:
            return history[-1]   
        
        history.append(new_value)
        if len(history) > self.max_history:
            history.pop(0)   
        return sum(history) / len(history)  

    def set_steer(self, steer):
        smoothed_steer = self.smooth_value(self.steering_history, steer*100)
        self.racer.set_steering(smoothed_steer)

    def set_speed(self, speed):
        smoothed_speed = self.smooth_value(self.speed_history, speed)
        self.racer.set_speed(smoothed_speed)
    
    def start(self):
        self.racer.start()

    def stop(self):
        self.racer.stop()
       



def process_image_center(frame):
    height, width = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_orange = np.array([10, 150, 150])
    upper_orange = np.array([25, 255, 255])
    mask = cv2.inRange(hsv, lower_orange, upper_orange)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)
    roi_height = int(height * 0.5)
    roi = mask[height-roi_height:height, :]
    contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, mask
    largest_contour = max(contours, key=cv2.contourArea)
    M = cv2.moments(largest_contour)
    if M["m00"] == 0:
        return None, mask
    cx = int(M["m10"] / M["m00"])
    center_deviation = (cx - (width / 2)) / (width / 2)
    
    return center_deviation, mask



def main():
    car = Car()
    car.start()
    controller = TrackController(car)
    time.sleep(4)
  
    cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    frame_count = 0
    frame_skip = 30
   
    last_time = time.time()

    try:
        while True:
            frame_count += 1
            if frame_count % frame_skip == 0:
                continue
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame.")
                break
            
            deviation, mask =process_image_center(frame)
              	
            
            controller.update(frame)
            debugFrame = controller.visualization(frame)
            cv2.imshow('Controle', frame)
            cv2.imshow('Debug', debugFrame)
            #hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            #lower_orange = np.array([10, 150, 150])
            #upper_orange = np.array([25, 255, 255])
            #mask = cv2.inRange(hsv, lower_orange, upper_orange)
            #kernel = np.ones((3, 3), np.uint8)
            #mask = cv2.erode(mask, kernel, iterations=1)
            #mask = cv2.dilate(mask, kernel, iterations=2)
            cv2.imshow('Orange Mask', mask)

            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        cap.release()
        car.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
