import torch
import torch.nn as nn
import cv2
import numpy as np
from pathlib import Path
import argparse
import onnx
import onnxruntime as ort
from PIL import Image
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List
import time
from linenet import create_linenet

class LineNetInference:
    def __init__(self, model_path: str, variant: str = 'nano', device: str = 'auto', input_size: Tuple[int, int] = (224, 224)):
        self.variant = variant
        self.input_size = input_size
        self.device = torch.device('cuda' if torch.cuda.is_available() and device == 'auto' else device)
        
        self.model = create_linenet(variant=variant, num_classes=1, input_channels=3)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded: {variant} variant")
        print(f"Device: {self.device}")
        print(f"Input size: {input_size}")
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for inference"""
        # Resize image
        image = cv2.resize(image, self.input_size, interpolation=cv2.INTER_LINEAR)
        
        # Convert BGR to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Normalize
        image = image.astype(np.float32) / 255.0
        
        # Apply ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std
        
        # Convert to tensor and add batch dimension
        image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        image = image.unsqueeze(0)
        
        return image.to(self.device)
    
    def postprocess_prediction(self, prediction: torch.Tensor, threshold: float = 0.5) -> np.ndarray:
        """Postprocess model prediction"""
        # Convert to numpy
        pred = prediction.cpu().numpy().squeeze()
        
        # Apply threshold
        pred_binary = (pred > threshold).astype(np.uint8) * 255
        
        return pred_binary
    
    def predict(self, image: np.ndarray, threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """Run inference on a single image"""
        # Preprocess
        input_tensor = self.preprocess_image(image)
        
        # Inference
        with torch.no_grad():
            prediction = self.model(input_tensor)
        
        # Postprocess
        pred_mask = self.postprocess_prediction(prediction, threshold)
        pred_prob = prediction.cpu().numpy().squeeze()
        
        return pred_mask, pred_prob
    
  
