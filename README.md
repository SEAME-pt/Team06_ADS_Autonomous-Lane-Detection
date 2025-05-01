# Team06-ADS_Autonomous-Lane-Detection


## Running YoloV5 with TensorRT Engine on Jetson

#### Support material
 ```bash
   https://github.com/mailrocketsystems/JetsonYolov5/blob/main/README.md

   https://www.youtube.com/watch?v=ErWC3nBuV6k&list=PLWw98q-Xe7iGIwrnBY_SpXHZzAZZ6944l

   https://www.youtube.com/watch?v=bcM5AQSAzUY&t=199s



python3 -m venv yolov5_env
source yolov5_env/bin/activate

1.
git clone https://github.com/ultralytics/yolov5
cd yolov5
pip install -r requirements.txt

2.
python train.py --img 416 --batch 16 --epochs 3 --data coco128.yaml --weights yolov5n.pt --name yolov5_coco128

3.
python export.py --weights runs/train/yolov5n_coco128/weights/best.pt --include onnx --opset 13 --simplify

4. (jetson nano)
/usr/src/tensorrt/bin/trtexec --onnx=best_yolov5n128.onnx --saveEngine=best_yolov5n128.engine --fp16
