import numpy as np
import pycuda.driver as cuda
import cv2

print("NumPy version:", np.__version__)
cuda.init()
print("PyCUDA - GPU:", cuda.Device(0).name())
print("OpenCV version:", cv2.__version__)
print("CUDA support:", cv2.cuda.getCudaEnabledDeviceCount() > 0)
