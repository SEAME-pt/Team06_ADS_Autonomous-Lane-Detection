import cv2
import numpy as np
import onnxruntime
from tqdm import tqdm

def load_onnx_model(onnx_path='pretrained/model.onnx'):
    session = onnxruntime.InferenceSession(
        onnx_path, 
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )
    return session

def process_frame_onnx(session, frame):
    frame_resized = cv2.resize(frame, (640, 360))
    img_original = frame_resized.copy()
    
    img = frame_resized[:, :, ::-1].transpose(2, 0, 1)  # BGR para RGB e HWC para CHW
    img = np.ascontiguousarray(img, dtype=np.float32)
    img = np.expand_dims(img, axis=0) / 255.0  # Adicionar batch dimension e normalizar
    
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: img})
    
    da_output = outputs[0]  # Driving area
    ll_output = outputs[1]  # Lane lines
    
    da_predict = np.argmax(da_output, axis=1)
    ll_predict = np.argmax(ll_output, axis=1)
    
    da_mask = (da_predict[0] * 255).astype(np.uint8)
    ll_mask = (ll_predict[0] * 255).astype(np.uint8)
    
    img_original[da_mask > 100] = [0, 0, 255]  # Driving area em vermelho
    img_original[ll_mask > 100] = [0, 255, 0]  # Lane lines em verde
    
    return img_original

def test_video_onnx(onnx_path, input_video, output_video):
    session = load_onnx_model(onnx_path)
    
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print(f"Erro: Não foi possível abrir o vídeo {input_video}")
        return
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"FPS: {fps}, Total frames: {total_frames}")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (640, 360))
    
    print("Processando vídeo com ONNX...")
    with tqdm(total=total_frames, desc="Frames processados") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            processed_frame = process_frame_onnx(session, frame)
            out.write(processed_frame)
            pbar.update(1)
    
    cap.release()
    out.release()
    print(f"Vídeo processado guardado em: {output_video}")

if __name__ == "__main__":
    onnx_path = "pretrained/model.onnx"
    input_video = "VID_20250527_183248.mp4"
    output_video = "output_segmented_onnx.mp4"    
    test_video_onnx(onnx_path, input_video, output_video)

