import torch
import cv2
import torch.utils.data
import torchvision.transforms as transforms
import numpy as np
import os
import random
import albumentations as A
import math

def augment_hsv_safe(img, hgain=0.015, sgain=0.7, vgain=0.4):
    """Versão mais robusta usando numpy"""
    if img is None or len(img.shape) != 3:
        return img
    
    # Certificar formato correto
    img_work = img.astype(np.float32) / 255.0
    
    # Converter para HSV usando cv2
    hsv = cv2.cvtColor(img_work, cv2.COLOR_BGR2HSV)
    
    # Gerar fatores aleatórios
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1
    
    # Aplicar mudanças
    hsv[:, :, 0] = (hsv[:, :, 0] * r[0]) % 1.0  # Hue
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * r[1], 0, 1)  # Saturation
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * r[2], 0, 1)  # Value
    
    # Converter de volta
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    # Converter para uint8
    result = (bgr * 255).astype(np.uint8)
    
    # Copiar de volta
    img[:] = result[:]
    
    return img

def augment_hsv(img, hgain=0.015, sgain=0.7, vgain=0.4):
    """change color hue, saturation, value"""
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype  # uint8

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed
def random_perspective(combination,  degrees=10, translate=.1, scale=.1, shear=10, perspective=0.0, border=(0, 0)):
    """combination of img transform"""
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # targets = [cls, xyxy]
    img, gray, line = combination
    height = img.shape[0] + border[0] * 2  # shape(h,w,c)
    width = img.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

    # Perspective
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            img = cv2.warpPerspective(img, M, dsize=(width, height), borderValue=(114, 114, 114))
            gray = cv2.warpPerspective(gray, M, dsize=(width, height), borderValue=0)
            line = cv2.warpPerspective(line, M, dsize=(width, height), borderValue=0)
        else:  # affine
            img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(114, 114, 114))
            gray = cv2.warpAffine(gray, M[:2], dsize=(width, height), borderValue=0)
            line = cv2.warpAffine(line, M[:2], dsize=(width, height), borderValue=0)



    combination = (img, gray, line)
    return combination



       
class JetsonDataSet(torch.utils.data.Dataset):
 
    def __init__(self, root_dir="organized_dataset",  color_augmentation=True,transform=None, valid=False):
        '''
        :param root_dir: diretório raiz do dataset organizado
        :param transform: transformações a aplicar
        :param valid: True para validação, False para treino
        '''
        self.transform = transform
        self.Tensor = transforms.ToTensor()
        self.valid = valid
        self.color_augmentation = color_augmentation
     
        split = "val" if valid else "train"
        self.root_dir = os.path.join(root_dir, split)
        
 
        self.images_dir = os.path.join(self.root_dir, "images")
        self.segments_dir = os.path.join(self.root_dir, "segments")
        self.lanes_dir = os.path.join(self.root_dir, "lanes")
        
 
        if not all(os.path.exists(d) for d in [self.images_dir, self.segments_dir, self.lanes_dir]):
            raise FileNotFoundError(f"Estrutura de pastas incompleta em {self.root_dir}")
        
        # Obter lista de imagens
        self.image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            self.image_files.extend([f for f in os.listdir(self.images_dir) 
                                   if f.lower().endswith(ext.replace('*',''))])
        
        self.image_files.sort()  # garantir ordem consistente
        
    
        self._check_dataset_integrity()
        
        self.setup_color_augmentations()
        
        if self.color_augmentation and not self.valid:
            print("Augmentações de cor ativadas para treino")


    def setup_color_augmentations(self):
        if not self.valid and self.color_augmentation:
            self.color_transform = A.Compose([
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,  # Reduzido de 0.3
                    contrast_limit=0.2,    # Reduzido de 0.3
                    p=0.6                  # Reduzido de 0.7
                ),
                
                A.HueSaturationValue(
                    hue_shift_limit=15,     # Reduzido de 20
                    sat_shift_limit=20,     # Reduzido de 30
                    val_shift_limit=15,     # Reduzido de 20
                    p=0.5                   # Reduzido de 0.6
                ),
                

                A.OneOf([
                    A.RandomGamma(gamma_limit=(85, 115), p=1.0),  # Mais conservador
                    A.CLAHE(clip_limit=1.5, tile_grid_size=(8, 8), p=1.0),
                ], p=0.3),  # Reduzido de múltiplas aplicações
                
                # Condições especiais (mais raras)
                A.OneOf([
                    A.RandomFog(fog_coef_lower=0.05, fog_coef_upper=0.15, p=1.0),  # Névoa mais sutil
                    A.RandomSunFlare(
                        flare_roi=(0, 0, 1, 0.3),  # Área menor
                        num_flare_circles_lower=1,
                        num_flare_circles_upper=2,  # Menos círculos
                        p=1.0
                    ),
                ], p=0.1),  # Muito mais raro
                
            ], p=0.7)  # Reduzido de 0.8

    def apply_color_augmentation(self, image):
  
        if self.color_transform is not None:
            # Albumentations espera imagem em RGB
            if len(image.shape) == 3 and image.shape[2] == 3:
                augmented = self.color_transform(image=image)
                return augmented['image']
        return image

    def apply_pytorch_color_jitter(self, image):
     
        if not self.valid and random.random() < 0.3:
            # Converter para PIL, aplicar ColorJitter, converter de volta
            from PIL import Image
            
            # Converter numpy para PIL
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            pil_image = Image.fromarray(image)
            
            # Aplicar ColorJitter
            color_jitter = transforms.ColorJitter(
                brightness=(0.7, 1.3),    # 70% a 130% do brilho original
                contrast=(0.8, 1.2),      # 80% a 120% do contraste
                saturation=(0.5, 1.5),    # 50% a 150% da saturação
                hue=(-0.1, 0.1)           # ±10% do matiz
            )
            
            jittered = color_jitter(pil_image)
            return np.array(jittered)
        
        return image    
    def _check_dataset_integrity(self):
  
        missing_files = []
        
        for img_file in self.image_files:
            img_name = os.path.splitext(img_file)[0]
            
          
            seg_file = os.path.join(self.segments_dir, f"{img_name}.png")
            lane_file = os.path.join(self.lanes_dir, f"{img_name}.png")
            
            if not os.path.exists(seg_file):
                missing_files.append(f"SEGMENT: {seg_file}")
            if not os.path.exists(lane_file):
                missing_files.append(f"LANE: {lane_file}")
        
        if missing_files:
            print(f"Faltam ({len(missing_files)}):")
            for missing in missing_files[:5]:  # mostrar apenas os primeiros 5
                print(f"   {missing}")
            if len(missing_files) > 5:
                print(f"   ... e mais {len(missing_files) - 5}")
            
            # Remover imagens que não têm todos os arquivos correspondentes
            valid_images = []
            for img_file in self.image_files:
                img_name = os.path.splitext(img_file)[0]
                seg_file = os.path.join(self.segments_dir, f"{img_name}.png")
                lane_file = os.path.join(self.lanes_dir, f"{img_name}.png")
                
                if os.path.exists(seg_file) and os.path.exists(lane_file):
                    valid_images.append(img_file)
            
            self.image_files = valid_images
            print(f"Dataset filtrado: {len(self.image_files)} imagens válidas")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        '''
        :param idx: Index da imagem
        :return: retorna a imagem e os labels correspondentes
        '''
        W_ = 640
        H_ = 360
        
        # Obter nome do arquivo
        img_file = self.image_files[idx]
        img_name = os.path.splitext(img_file)[0]
        
        # Caminhos completos
        image_path = os.path.join(self.images_dir, img_file)
        segment_path = os.path.join(self.segments_dir, f"{img_name}.png")
        lane_path = os.path.join(self.lanes_dir, f"{img_name}.png")
        
        # Carregar arquivos
        image = cv2.imread(image_path)
        label1 = cv2.imread(segment_path, 0)  # segmentação
        label2 = cv2.imread(lane_path, 0)     # linhas
        
        # Verificar se foram carregados corretamente
        if image is None:
            raise ValueError(f"Erro ao carregar imagem: {image_path}")
        if label1 is None:
            raise ValueError(f"Erro ao carregar segmentação: {segment_path}")
        if label2 is None:
            raise ValueError(f"Erro ao carregar linhas: {lane_path}")
        
        if not self.valid:
            # FASE 1: Augmentações Geométricas (uma por vez)
            geometric_aug_prob = 0.6  # Reduzido de 0.7
            if random.random() < geometric_aug_prob:
                combination = (image, label1, label2)
                (image, label1, label2) = random_perspective(
                    combination=combination,
                    degrees=10,      
                    translate=0.1,    
                    scale=0.2,        
                    shear=0.05       
                )
            
            # FASE 2: Augmentações de Qualidade 
            quality_choice = random.random()
            if quality_choice < 0.15:  # 15% blur
                image = cv2.GaussianBlur(image, (3, 3), 0)  # Kernel menor
            elif quality_choice < 0.25:  # 10% noise
                noise = np.random.normal(0, 15, image.shape).astype(np.uint8)  # Menos ruído
                image = cv2.add(image, noise)
            
            
            if random.random() < 0.5:
                image = np.fliplr(image)
                label1 = np.fliplr(label1)
                label2 = np.fliplr(label2)
            
            # FASE 4: Augmentações de Cor  
            color_choice = random.random()
            if color_choice < 0.4:  # 40% - Albumentations (mais natural)
                image = self.apply_color_augmentation(image)
            elif color_choice < 0.6:  # 20% - ColorJitter (mais agressivo)
                image = self.apply_pytorch_color_jitter(image)
            #elif color_choice < 0.8:  # 20% - HSV customizado
            #    image=augment_hsv_safe(image, hgain=0.01, sgain=0.5, vgain=0.3)  # Mais suave
    
        # Redimensionar
        label1 = cv2.resize(label1, (W_, H_))
        label2 = cv2.resize(label2, (W_, H_))
        image = cv2.resize(image, (W_, H_))
        
        # Processamento dos labels
        _, seg_b1 = cv2.threshold(label1, 1, 255, cv2.THRESH_BINARY_INV)
        _, seg_b2 = cv2.threshold(label2, 1, 255, cv2.THRESH_BINARY_INV)
        _, seg1 = cv2.threshold(label1, 1, 255, cv2.THRESH_BINARY)
        _, seg2 = cv2.threshold(label2, 1, 255, cv2.THRESH_BINARY)
        
        # Converter para tensores
        seg1 = self.Tensor(seg1)
        seg2 = self.Tensor(seg2)
        seg_b1 = self.Tensor(seg_b1)
        seg_b2 = self.Tensor(seg_b2)
        
        # Empilhar tensores
        seg_da = torch.stack((seg_b1[0], seg1[0]), 0)
        seg_ll = torch.stack((seg_b2[0], seg2[0]), 0)
        
        # Processar imagem
        image = image[:, :, ::-1].transpose(2, 0, 1)  # BGR para RGB e HWC para CHW
        image = np.ascontiguousarray(image)
        
        return image_path, torch.from_numpy(image), (seg_da, seg_ll)
