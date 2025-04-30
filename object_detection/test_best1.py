import cv2
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import time

# === CONFIG ===
ENGINE_PATH = "best.engine"
CONF_THRESH = 0.4
IOU_THRESH = 0.5
INPUT_WIDTH = 640
INPUT_HEIGHT = 640
LABELS = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
    "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

# === LOAD ENGINE ===
TRT_LOGGER = trt.Logger(trt.Logger.INFO)
with open(ENGINE_PATH, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())

context = engine.create_execution_context()

# === MEMORY ALLOCATION ===
input_shape = (1, 3, INPUT_HEIGHT, INPUT_WIDTH)
output_shape = (1, 25200, 85)  # 80 classes + 5 valores (x, y, w, h, conf)

input_size = np.prod(input_shape)
output_size = np.prod(output_shape)

d_input = cuda.mem_alloc(input_size * np.float32().nbytes)
d_output = cuda.mem_alloc(output_size * np.float32().nbytes)

bindings = [int(d_input), int(d_output)]

# === FUNC: PREPROCESS ===
def preprocess(image):
    image_resized = cv2.resize(image, (INPUT_WIDTH, INPUT_HEIGHT))
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    image_transposed = image_rgb.transpose((2, 0, 1))  # CHW
    image_normalized = image_transposed / 255.0
    input_data = np.expand_dims(image_normalized, axis=0).astype(np.float32)
    return input_data

# === FUNC: NMS + DRAW ===
def postprocess(output, original_image):
    output = output.reshape(-1, len(LABELS) + 5)
    boxes = []
    for det in output:
        conf = det[4]
        if conf < CONF_THRESH:
            continue
        scores = det[5:]
        class_id = np.argmax(scores)
        class_conf = scores[class_id]
        if class_conf < CONF_THRESH:
            continue
        cx, cy, w, h = det[0:4]
        x1 = int((cx - w/2) * original_image.shape[1] / INPUT_WIDTH)
        y1 = int((cy - h/2) * original_image.shape[0] / INPUT_HEIGHT)
        x2 = int((cx + w/2) * original_image.shape[1] / INPUT_WIDTH)
        y2 = int((cy + h/2) * original_image.shape[0] / INPUT_HEIGHT)
        boxes.append((x1, y1, x2, y2, class_id, float(class_conf)))

    for x1, y1, x2, y2, class_id, conf in boxes:
        label = f"{LABELS[class_id]} {conf:.2f}"
        cv2.rectangle(original_image, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(original_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    return original_image

# === MAIN LOOP ===
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    input_data = preprocess(frame)
    output_data = np.empty(output_shape, dtype=np.float32)

    # Transfer to device
    cuda.memcpy_htod(d_input, input_data)
    context.execute_v2(bindings)
    cuda.memcpy_dtoh(output_data, d_output)

    # Draw results
    result = postprocess(output_data, frame)
    cv2.imshow("YOLOv5 + TensorRT", result)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
