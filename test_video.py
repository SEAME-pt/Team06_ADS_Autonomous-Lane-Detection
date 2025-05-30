import torch
import cv2
import numpy as np
from tqdm import tqdm
from model import TwinLite as net

def load_model(model_path='pretrained/model.pth'):
 
    model = net.TwinLiteNet()
    #model = torch.nn.DataParallel(model)
    model = model.cuda()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def process_frame(model, frame):
 
    frame_resized = cv2.resize(frame, (640, 360))
    img_original = frame_resized.copy()
    
 
    img = frame_resized[:, :, ::-1].transpose(2, 0, 1)  # BGR para RGB e HWC para CHW
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).unsqueeze(0).cuda().float() / 255.0
    
    # Inferência
    with torch.no_grad():
        outputs = model(img)
        
        # Obter predições
        _, da_predict = torch.max(outputs[0], 1)  # Driving area
        _, ll_predict = torch.max(outputs[1], 1)  # Lane lines
        
        # Converter para numpy
        da_mask = da_predict.byte().cpu().data.numpy()[0] * 255
        ll_mask = ll_predict.byte().cpu().data.numpy()[0] * 255
        
        # Aplicar máscaras coloridas na imagem original
        img_original[da_mask > 100] = [0, 0, 255]    # Driving area em vermelho
        img_original[ll_mask > 100] = [0, 255, 0]    # Lane lines em verde
    
    return img_original

def test_video(model_path, input_video, output_video):
 
    model = load_model(model_path)
    
 
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print(f"Erro: Não foi possível abrir o vídeo {input_video}")
        return
    
 
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"FPS: {fps}, Total frames: {total_frames}")
    
 
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (640, 360))
    
 
    print("Processando vídeo...")
    with tqdm(total=total_frames, desc="Frames processados") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
 
            processed_frame = process_frame(model, frame)
            
       
            out.write(processed_frame)
            
            pbar.update(1)
 
    cap.release()
    out.release()
    
    print(f"Vídeo processado guardado em: {output_video}")

if __name__ == "__main__":
 
    model_path = "pretrained/model.pth"       
    input_video = "video_20250410_071911.avi"            
    output_video = "output_segmented.mp4"    
    
 
    test_video(model_path, input_video, output_video)

