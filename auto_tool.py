import torch
import numpy as np
import shutil
from tqdm.autonotebook import tqdm
import os
import cv2
from model import TwinLite as net

def Run(model, img):
    img = cv2.resize(img, (640, 360))
    img_rs = img.copy()
    
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img)
    img = torch.unsqueeze(img, 0)
    img = img.cuda().float() / 255.0

    with torch.no_grad():
        img_out = model(img)
        x0 = img_out[0]  # driving area segmentation
        x1 = img_out[1]  # lane line detection

        _, da_predict = torch.max(x0, 1)
        _, ll_predict = torch.max(x1, 1)

        DA = da_predict.byte().cpu().data.numpy()[0] * 255
        LL = ll_predict.byte().cpu().data.numpy()[0] * 255

        # Criar máscara binária para segmentação
        img_segmentation = np.zeros_like(img_rs)
        img_segmentation[DA > 100] = [255, 255, 255]
        
        # Criar máscara binária para linhas
        img_lines = np.zeros_like(img_rs)
        img_lines[LL > 100] = [255, 255, 255]

        return img_segmentation, img_lines


def create_output_directories():
    directories = ['results/segments', 'results/lines']
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Pasta criada: {directory}")

def main():
    model = net.TwinLiteNet()
    model = model.cuda()
    model.load_state_dict(torch.load('pretrained/model.pth'))
    model.eval()
 
    create_output_directories()
 
    input_folder = 'images'
    input_folder = '/home/djoker/code/cuda/LandDetection/bk/fotos'
    
    if not os.path.exists(input_folder):
        print(f"Pasta '{input_folder}' não encontrada!")
        return

    image_list = os.listdir(input_folder)
    
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_list = [img for img in image_list if any(img.lower().endswith(ext) for ext in valid_extensions)]

    print(f"Processando {len(image_list)} imagens...")

    for i, imgName in enumerate(tqdm(image_list, desc="Processando imagens")):
        try:
            img_path = os.path.join(input_folder, imgName)
            img = cv2.imread(img_path)
            
            if img is None:
                print(f"Erro ao carregar imagem: {imgName}")
                continue

            img_segmentation, img_lines = Run(model, img)

            name_without_ext = os.path.splitext(imgName)[0]

            segmentation_path = os.path.join('results/segments', f"{name_without_ext}.png")
            cv2.imwrite(segmentation_path, img_segmentation)

            lines_path = os.path.join('results/lines', f"{name_without_ext}.png")
            cv2.imwrite(lines_path, img_lines)

            print(f"Processada: {imgName} -> PNG")

        except Exception as e:
            print(f"Erro ao processar {imgName}: {str(e)}")

    print("Processamento concluído! Todas as imagens salvas em PNG.")

if __name__ == "__main__":
    main()
