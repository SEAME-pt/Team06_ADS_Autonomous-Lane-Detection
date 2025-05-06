# Team06-ADS_Autonomous-Lane-Detection

## line Dect

### onnx -> tensorrt on nano
/usr/src/tensorrt/bin/trtexec --onnx=unet_model_256.onnx --saveEngine=unet_model_256.engine --fp16



#################################

## Running YoloV5 with TensorRT Engine on Jetson

python3 -m venv yolov5_env
source yolov5_env/bin/activate

1.
git clone https://github.com/ultralytics/yolov5
cd yolov5
pip install -r requirements.txt

2.
python train.py \
  --img 320 \
  --batch 8 \
  --epochs 50 \
  --data coco128.yaml \
  --weights yolov5n.pt \
  --device 0 \
  --workers 4

3.
python export.py \
  --weights runs/train/crosswalk-pedestrian7/weights/best.pt \
  --include onnx \
  --opset 13 \
  --imgsz 320

4. (jetson nano)
/usr/src/tensorrt/bin/trtexec --onnx=yolov5_crosswalk.onnx --saveEngine=yolov5_crosswalk.engine --fp16


## Running YoloV8 with TensorRT Engine on Jetson
1. Install YOLOv8 (Ultralytics)

pip install ultralytics

2. Train with YOLOv8

yolo task=detect mode=train model=yolov8n.pt data=dataset.yaml epochs=100 imgsz=320 batch=16

opcional - "device=0"

3. 

yolo task=detect mode=val model=runs/detect/train/weights/best.pt data=dataset.yaml

4. 

yolo export model=runs/detect/train/weights/best.pt format=engine





## Support material
 ```bash
   https://github.com/mailrocketsystems/JetsonYolov5/blob/main/README.md

   https://www.youtube.com/watch?v=ErWC3nBuV6k&list=PLWw98q-Xe7iGIwrnBY_SpXHZzAZZ6944l

   https://www.youtube.com/watch?v=bcM5AQSAzUY&t=199s



python3 -m venv yolov5_env
source yolov5_env/bin/activate

1.
git clone https://github.com/ultralytics/yolov8
cd yolov5
pip install -r requirements.txt

2.
python train.py --img 320 --batch 16 --epochs 50 --data coco128.yaml --weights yolov5n.pt --name yolov5_coco128

python train.py \
  --img 320 \
  --batch 8 \
  --epochs 50 \
  --data coco128.yaml \
  --weights yolov5n.pt \
  --device 0 \
  --workers 4

3.
python export.py --weights runs/train/yolov5n_coco128/weights/best.pt --include onnx --opset 13 --simplify

python export.py \
  --weights runs/train/exp/weights/best.pt \
  --include onnx \
  --opset 13 \
  --imgsz 320

4. (jetson nano)
/usr/src/tensorrt/bin/trtexec --onnx=best_320.onnx --saveEngine=best_320.engine --fp16


