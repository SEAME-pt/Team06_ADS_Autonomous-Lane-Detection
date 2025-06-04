 

# LineNet: Deep Learning Tools for Line Detection

**LineNet** is a collection of Python tools for training, inference, and visualization of deep learning models specialized in line detection. The project includes modules for model architecture, training, inference, and real-time video processing.

## Features

- **Model Training:** Train LineNet variants (nano, lite, small, medium, strong) with customizable parameters and loss functions.
- **Inference:** Run predictions on images or video frames using trained models.
- **Video Processing:** Process videos frame-by-frame, apply masks, and overlay results with customizable colors.
- **Visualization:** Generate prediction grids and training curves for monitoring and analysis.
- **Early Stopping:** Automatic model checkpointing and early stopping to prevent overfitting.


## Project Structure

```
linenet/
│
├── linenet.py           # Model architecture and factory functions
├── train.py             # Training script with data loading, augmentation, and logging
├── inference.py         # Inference class for image and batch processing
├── video.py             # Video processing with mask overlay and cleanup
└── README.md            # Project documentation (this file)
```


## Installation

1. **Clone this repository:**

```bash
git clone https://github.com/akadjoker/linenet.git
cd linet
```

2. **Install dependencies:**

```bash
pip install torch torchvision opencv-python matplotlib albumentations tqdm numpy
```


## Usage

### 1. **Training**

Train a model using your dataset:

```bash
python train.py --train_images /path/to/images --train_masks /path/to/masks --variant small --batch_size 4 --epochs 200 --learning_rate 1e-3 --image_size 224 224
```


### 2. **Inference on Images**

Run inference on a single image:

```python
from inference import LineNetInference

infer = LineNetInference(model_path='best_model.pth', variant='small')
pred_mask, _ = infer.predict(image)
```




## Model Variants

| Variant | Use Case |
| :-- | :-- |
| nano | Edge devices, real-time |
| lite | Lightweight, efficient |
| small | Balanced performance |
| medium | Higher accuracy |
| strong | Maximum accuracy |

## Best Practices

- **Use early stopping** to avoid overfitting and save the best model.
- **Monitor training curves** (loss, IoU, Dice, etc.) for model performance.
- **Clean and smooth masks** with OpenCV for better visual results in videos.


 