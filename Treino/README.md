# JetCar System Documentation

## Overview

The JetCar system is a robotic platform based on Jetson Nano that enables controlling a robotic car through keyboard commands, recording videos during operation, and collecting datasets for training machine learning models for autonomous driving.

## System Requirements

- NVIDIA Jetson Nano
- GStreamer-compatible camera (CSI or USB)
- JetCar library for car hardware interface
- OpenCV 4.x
- Python 3.6+

## System Structure

The software consists of a main `Controller` class that manages the following subsystems:

1. **Vehicle Control**
   - Interface with the JetCar module to control steering and speed
   - Mapping of keyboard commands to movements

2. **Video Capture**
   - Camera initialization via GStreamer pipeline
   - Real-time frame display
   - Visual interface with status information

3. **Video Recording**
   - Creation and management of recording sessions
   - Use of multiple codecs to ensure compatibility

4. **Dataset Collection**
   - Capture of specific frames with associated steering values
   - Organization in directory structure
   - Creation of CSV for mapping images and steering values

## System Initialization

1. The `Controller` class is initialized
2. The car is initialized through the JetCar library
3. The camera is configured using the GStreamer pipeline
4. A session directory for videos is created

## Camera Pipeline

The system uses GStreamer to capture video from the camera with the following pipeline:

```
nvarguscamerasrc ! 
video/x-raw(memory:NVMM), 
width=(int)WIDTH, height=(int)HEIGHT, 
format=(string)NV12, framerate=(fraction)FPS/1 ! 
nvvidconv flip-method=FLIP ! 
video/x-raw, width=(int)DISPLAY_WIDTH, height=(int)DISPLAY_HEIGHT, format=(string)BGRx ! 
videoconvert ! 
video/x-raw, format=(string)BGR ! appsink
```

This pipeline is optimized for NVIDIA hardware and uses hardware acceleration for efficient video processing.

## Steering and Speed Control

| Key | Function |
|-----|----------|
| W | Increase forward speed |
| S | Increase backward speed |
| A | Turn left |
| D | Turn right |
| C | Center steering |
| Space | Stop vehicle (zero speed) |

- Speed is adjusted in increments of 0.02 (-1.0 to 1.0)
- Steering is adjusted in increments of 0.1 (-1.0 to 1.0)
- Maximum speed is limited to 70% (configurable via `max_speed`)

## Video Recording

| Key | Function |
|-----|----------|
| R | Start/stop video recording |

### Recording Process

1. When recording is activated:
   - A video file with timestamp is created in the current session
   - Resolution and frame rate are configured based on the camera
   - The system tries the following codecs in order: MJPG, XVID, IYUV

2. During recording, a visual indicator (red circle) appears in the window along with the recording time

3. When recording is ended:
   - The file is finalized
   - The recording duration is displayed in the terminal

### Video Location

All videos are stored in:
```
videos/session_YYYYMMDD_HHMMSS/video_YYYYMMDD_HHMMSS.avi
```

## Dataset Collection

| Key | Function |
|-----|----------|
| T | Start/stop dataset collection |
| Enter | Capture a frame for the dataset |

### Collection Process

1. When collection is activated:
   - A dataset directory with timestamp is created
   - A CSV file is initialized with headers "image_path,steering"

2. During collection:
   - A visual indicator (blue circle) appears on screen
   - The current number of collected frames is displayed
   - An instruction "ENTER to capture frame" is shown

3. When pressing ENTER:
   - The current frame is saved with a timestamp-based name
   - The current steering value is recorded in the CSV
   - A visual capture feedback is temporarily shown

4. When collection is ended:
   - The CSV file is closed
   - The total number of collected frames is displayed in the terminal

### Dataset Structure

```
dataset/
└── session_YYYYMMDD_HHMMSS/
    ├── images/
    │   ├── frame_YYYYMMDD_HHMMSS_ffffff.jpg
    │   ├── frame_YYYYMMDD_HHMMSS_ffffff.jpg
    │   └── ...
    └── steering_data.csv
```

The CSV file contains entries in the format:
```
image_path,steering
images/frame_YYYYMMDD_HHMMSS_ffffff.jpg,0.300000
images/frame_YYYYMMDD_HHMMSS_ffffff.jpg,-0.200000
...
```

## Visual Interface

The system displays a complete visual interface with:

1. **Status Information**
   - Current speed
   - Current steering
   - Recording status (if active)
   - Dataset collection status (if active)
   - FPS (frames per second)

2. **Visual Representations**
   - Horizontal bar for steering with position indicator
   - Vertical bar for speed with colored indicator (green for forward, red for backward)

3. **Instructions**
   - Control legend at the bottom of the screen

## System Termination

When the program is terminated (by pressing ESC or CTRL+C):

1. The car is stopped (speed and steering reset to zero)
2. Any ongoing recording is properly finalized
3. Any ongoing dataset collection is properly finalized
4. The camera is released
5. All windows are closed

## Using the Dataset for Training

The collected dataset can be used to train autonomous driving models, such as:

1. Convolutional neural networks that predict steering angle based on the image
2. Imitation learning models
3. Reinforcement learning systems with demonstrations