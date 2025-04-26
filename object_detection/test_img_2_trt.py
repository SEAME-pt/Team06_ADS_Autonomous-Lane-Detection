import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

# Função para realizar o pós-processamento
def postprocess_output(output, conf_threshold=0.5, iou_threshold=0.4):
    output = output.reshape(-1, 6)  # (cx, cy, w, h, conf, class_id)
    
    # Imprimir as detecções antes do filtro de confiança
    print("Detections (raw):")
    print(output)
    
    # Filtrando apenas caixas com confiança acima do limiar
    output = output[output[:, 4] > conf_threshold]
    
    # Imprimir as detecções após o filtro de confiança
    print("Detections after confidence threshold:")
    print(output)
    
    boxes = output[:, :4]
    scores = output[:, 4]
    class_ids = output[:, 5].astype(int)

    # Realiza a operação NMS (Non-Maximum Suppression)
    indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), conf_threshold, iou_threshold)
    
    final_boxes = []
    final_scores = []
    final_class_ids = []
    
    if len(indices) > 0:
        for i in indices.flatten():
            final_boxes.append(boxes[i])
            final_scores.append(scores[i])
            final_class_ids.append(class_ids[i])
    
    # Imprimir as caixas finais após o NMS
    print("Final detections after NMS:")
    print(final_boxes, final_scores, final_class_ids)
    
    return final_boxes, final_scores, final_class_ids

# Função para desenhar as caixas de detecção na imagem
def draw_boxes(image, boxes, class_ids, scores, class_names):
    for i, box in enumerate(boxes):
        # Verificando se class_ids[i] é válido
        if class_ids[i] >= len(class_names):
            print(f"Warning: class_id {class_ids[i]} is out of range")
            continue
        
        x, y, w, h = box
        x1, y1, x2, y2 = int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)
        
        # Desenhando a caixa
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Adicionando o label da classe e a confiança
        label = f"{class_names[class_ids[i]]} {scores[i]:.2f}"
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    return image

# Inicializar TensorRT
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
engine_file_path = "best.engine"

# Carregar o engine
with open(engine_file_path, "rb") as f:
    engine_data = f.read()

runtime = trt.Runtime(TRT_LOGGER)
engine = runtime.deserialize_cuda_engine(engine_data)
context = engine.create_execution_context()

# Descobrir input e output bindings para TensorRT 10
for idx in range(engine.num_io_tensors):
    tensor_name = engine.get_tensor_name(idx)
    tensor_shape = engine.get_tensor_shape(tensor_name)
    tensor_dtype = engine.get_tensor_dtype(tensor_name)
    tensor_mode = engine.get_tensor_mode(tensor_name)

    if tensor_mode == trt.TensorIOMode.INPUT:
        input_name = tensor_name
        input_shape = tensor_shape
    else:
        output_name = tensor_name
        output_shape = tensor_shape

# ---- PRE-PROCESSAR IMAGEM ----
img = cv2.imread('cao_homem.jpg')
img_resized = cv2.resize(img, (640, 640))  # Resize para o esperado
img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
img_normalized = img_rgb.astype(np.float32) / 255.0
img_transposed = np.transpose(img_normalized, (2, 0, 1))
img_input = np.expand_dims(img_transposed, axis=0)  # (1, 3, 640, 640)
img_input = np.ascontiguousarray(img_input)

# ---- ALOCAR MEMÓRIA CUDA ----
d_input = cuda.mem_alloc(img_input.nbytes)
d_output = cuda.mem_alloc(np.empty(output_shape, dtype=np.float32).nbytes)

# ---- PREPARAR STREAM e BINDINGS ----
stream = cuda.Stream()
cuda.memcpy_htod_async(d_input, img_input, stream)

context.set_tensor_address(input_name, int(d_input))
context.set_tensor_address(output_name, int(d_output))

context.execute_async_v3(stream.handle)

# ---- COPIAR OUTPUT para CPU ----
output = np.empty(output_shape, dtype=np.float32)
cuda.memcpy_dtoh_async(output, d_output, stream)

# ---- ESPERAR STREAM CONCLUIR ----
stream.synchronize()

# ---- OUTPUT PRONTO ----
print("Output shape:", output.shape)
print("Output sample:", output.flatten()[:10])

# Pós-processamento
boxes, scores, class_ids = postprocess_output(output)

# Names das classes (ajustar se necessário)
class_names = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'none', 'stop sign', 'parking meter', 
    'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
    'zebra', 'giraffe', 'none', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 
    'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 
    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 
    'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 
    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 
    'hair drier', 'toothbrush'
]  # As classes do COCO dataset (coco128 contém uma versão reduzida, mas é similar)

# Desenhar as caixas de detecção na imagem
result_img = draw_boxes(img.copy(), boxes, class_ids, scores, class_names)

# Exibir a imagem com as caixas desenhadas
cv2.imshow("Detections", result_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Salvar a imagem se quiser
cv2.imwrite('output_image.jpg', result_img)
