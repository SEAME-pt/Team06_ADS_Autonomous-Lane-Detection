import cv2
from pathlib import Path
from linenet import create_linenet
from inference import LineNetInference
import torch
import numpy as np

class LineNetVideoInference(LineNetInference):
    def process_video(self, video_path: str, output_path: str, threshold: float = 0.5):
        cap = cv2.VideoCapture(video_path)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height), True)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            pred_mask, _ = self.predict(frame, 0.5)
            pred_mask = cv2.resize(pred_mask, (frame_width, frame_height), interpolation=cv2.INTER_NEAREST)
            color_mask = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
            color_mask[pred_mask == 255] = (255,0,255)
            kernel = np.ones((3, 3), np.uint8)
            cleaned_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel, iterations=1)
            cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

            overlay = cv2.addWeighted(frame, 0.5, color_mask, 1.0, 0)

            out.write(overlay)
    
            cv2.imshow('Video', overlay)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        out.release()
        cv2.destroyAllWindows()

 
model_path = '/home/djoker/code/conda/linenet/outputs/medium/best_model.pth'
video_path = '/home/djoker/code/cuda/estrada.mp4'
video_path = 'http://100.93.94.123:8080/stream.mjpg'
output_path = 'output.mp4'
infer = LineNetVideoInference(model_path, variant='medium', device='auto')
infer.process_video(video_path, output_path)

