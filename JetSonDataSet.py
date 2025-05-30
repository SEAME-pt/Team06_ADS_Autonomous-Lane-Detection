import torch
import cv2
import numpy as np
import random
import os
from pathlib import Path
from torchvision import transforms
import albumentations as A
from torch.utils.data import random_split, DataLoader
from sklearn.model_selection import train_test_split
import math

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


class CustomLaneDataset(torch.utils.data.Dataset):
 
    def __init__(self, images_dir='dataset/images', 
                 lines_dir='dataset/lines', 
                 segmentation_dir='dataset/segmentation',
                 transform=None, valid=False, img_size=(640, 360),
                 color_augmentation=True):
        
        self.images_dir = images_dir
        self.lines_dir = lines_dir
        self.segmentation_dir = segmentation_dir
        self.transform = transform
        self.Tensor = transforms.ToTensor()
        self.valid = valid
        self.W_, self.H_ = img_size
        self.color_augmentation = color_augmentation

 
        for dir_path in [images_dir, lines_dir, segmentation_dir]:
            if not os.path.exists(dir_path):
                raise FileNotFoundError(f"Pasta não encontrada: {dir_path}")

 
        img_files = sorted([f for f in os.listdir(images_dir) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        line_files = sorted([f for f in os.listdir(lines_dir) 
                            if f.lower().endswith('.png')])
        seg_files = sorted([f for f in os.listdir(segmentation_dir) 
                           if f.lower().endswith('.png')])

        
        img_bases = {os.path.splitext(f)[0]: f for f in img_files}
        line_bases = {os.path.splitext(f)[0]: f for f in line_files}
        seg_bases = {os.path.splitext(f)[0]: f for f in seg_files}
        common_keys = sorted(set(img_bases.keys()) & 
                           set(line_bases.keys()) & 
                           set(seg_bases.keys()))

        self.img_paths = [os.path.join(images_dir, img_bases[k]) for k in common_keys]
        self.line_paths = [os.path.join(lines_dir, line_bases[k]) for k in common_keys]
        self.seg_paths = [os.path.join(segmentation_dir, seg_bases[k]) for k in common_keys]
        
        # Configurar augmentações de cor usando Albumentations
        self.setup_color_augmentations()
        
        print(f"Dataset carregado: {len(self.img_paths)} triplas")
        if self.color_augmentation and not self.valid:
            print("Augmentações de cor ativadas para treino")

    def setup_color_augmentations(self):
 
        if not self.valid and self.color_augmentation:
            self.color_transform = A.Compose([
                # Mudanças de brilho e contraste (para diferentes condições de luz)
                A.RandomBrightnessContrast(
                    brightness_limit=0.3,  # ±30% de brilho
                    contrast_limit=0.3,    # ±30% de contraste
                    p=0.7
                ),
                
                # Mudanças de matiz e saturação (para diferentes cores de vidro)
                A.HueSaturationValue(
                    hue_shift_limit=20,     # ±20 graus de matiz
                    sat_shift_limit=30,     # ±30 de saturação
                    val_shift_limit=20,     # ±20 de valor
                    p=0.6
                ),
                
                # Gamma correction (para simular diferentes exposições)
                A.RandomGamma(
                    gamma_limit=(80, 120),  # Gamma entre 0.8 e 1.2
                    p=0.4
                ),
                
                # CLAHE para melhorar contraste local
                A.CLAHE(
                    clip_limit=2.0,
                    tile_grid_size=(8, 8),
                    p=0.3
                ),
                
                # Color jittering para variações sutis
                A.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1,
                    p=0.5
                ),
                
                # Mudanças de temperatura de cor
                A.OneOf([
                    A.ToSepia(p=1.0),  # Tom sépia
                    A.ToGray(p=1.0),   # Escala de cinza
                    A.ChannelShuffle(p=1.0),  # Embaralhar canais
                ], p=0.2),
                
                # Simulação de diferentes condições atmosféricas
                A.OneOf([
                    A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=1.0),
                    A.RandomSunFlare(
                        flare_roi=(0, 0, 1, 0.5),
                        angle_lower=0,
                        angle_upper=1,
                        num_flare_circles_lower=1,
                        num_flare_circles_upper=3,
                        p=1.0
                    ),
                ], p=0.15),
            ], p=0.8)  # 80% chance de aplicar alguma augmentação
        else:
            self.color_transform = None

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

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
     
        image = cv2.imread(self.img_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR -> RGB
        
        line_mask = cv2.imread(self.line_paths[idx], 0)
        seg_mask = cv2.imread(self.seg_paths[idx], 0)
        
        image_name = os.path.splitext(os.path.basename(self.img_paths[idx]))[0]

        # Augmentações geométricas (apenas no treino)
        if not self.valid:
            # Flip horizontal
            if random.random() < 0.5:
                image = np.fliplr(image)
                line_mask = np.fliplr(line_mask)
                seg_mask = np.fliplr(seg_mask)
            
 
            if random.random()<0.5:
                combination = (image, line_mask, seg_mask)
                (image, label1, label2)= random_perspective(
                    combination=combination,
                    degrees=10,
                    translate=0.1,
                    scale=0.25,
                    shear=0.0
                )
            if random.random()<0.5:
                augment_hsv(image)
            if random.random() < 0.5:
                image = np.fliplr(image)
                label1 = np.fliplr(label1)
                label2 = np.fliplr(label2)
        # Aplicar augmentações de cor (antes do redimensionamento)
        if not self.valid:
            # Aplicar Albumentations
            image = self.apply_color_augmentation(image)
            
            # Aplicar PyTorch ColorJitter ocasionalmente
            image = self.apply_pytorch_color_jitter(image)

 
        image = cv2.resize(image, (self.W_, self.H_))
        line_mask = cv2.resize(line_mask, (self.W_, self.H_))
        seg_mask = cv2.resize(seg_mask, (self.W_, self.H_))

 
        _, seg_bg_da = cv2.threshold(seg_mask, 1, 255, cv2.THRESH_BINARY_INV)
        _, seg_fg_da = cv2.threshold(seg_mask, 1, 255, cv2.THRESH_BINARY)
        seg_bg_da = self.Tensor(seg_bg_da)
        seg_fg_da = self.Tensor(seg_fg_da)
        seg_da = torch.stack((seg_bg_da[0], seg_fg_da[0]), 0)

        _, seg_bg_ll = cv2.threshold(line_mask, 1, 255, cv2.THRESH_BINARY_INV)
        _, seg_fg_ll = cv2.threshold(line_mask, 1, 255, cv2.THRESH_BINARY)
        seg_bg_ll = self.Tensor(seg_bg_ll)
        seg_fg_ll = self.Tensor(seg_fg_ll)
        seg_ll = torch.stack((seg_bg_ll[0], seg_fg_ll[0]), 0)

        # Processar imagem (RGB -> CHW)
        image = image.transpose(2, 0, 1)
        image = np.ascontiguousarray(image)

        return image_name, torch.from_numpy(image), (seg_da, seg_ll)
 

    @staticmethod
    def create_train_val_split(dataset, val_split=0.2,   random_seed=42):
        """
        Cria split de treino, validação e teste
        """
        total_size = len(dataset)
        

    
        val_size = int(total_size * val_split)
        train_size = total_size - val_size  
        

        generator = torch.Generator().manual_seed(random_seed)
        
    
        train_dataset, val_dataset = random_split(
            dataset,
            [train_size, val_size],
            generator=generator
        )
        return train_dataset, val_dataset