import os
import random
import shutil

# Diretórios de origem
images_dir = '/home/djoker/code/cuda/mixdataset/images'
masks_dir = '/home/djoker/code/cuda/mixdataset/masks'

# Diretórios de destino
train_images_dir = './dataset/train/images'
train_masks_dir = './dataset/train/masks'
val_images_dir = './dataset/val/images'
val_masks_dir = './dataset/val/masks'

# Criar pastas de destino
os.makedirs(train_images_dir, exist_ok=True)
os.makedirs(train_masks_dir, exist_ok=True)
os.makedirs(val_images_dir, exist_ok=True)
os.makedirs(val_masks_dir, exist_ok=True)


image_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
mask_files = {os.path.splitext(f)[0]: f for f in os.listdir(masks_dir) if f.lower().endswith('.png')}

# Filtrar apenas imagens com máscara correspondente
valid_pairs = []
for img in image_files:
    base = os.path.splitext(img)[0]
    if base in mask_files:
        valid_pairs.append((img, mask_files[base]))

random.shuffle(valid_pairs)

# Separar 20% para validação
split_idx = int(0.8 * len(valid_pairs))
train_pairs = valid_pairs[:split_idx]
val_pairs = valid_pairs[split_idx:]

# Copiar para treino
for img, mask in train_pairs:
    shutil.copy(os.path.join(images_dir, img), os.path.join(train_images_dir, img))
    shutil.copy(os.path.join(masks_dir, mask), os.path.join(train_masks_dir, mask))

# Copiar para validação
for img, mask in val_pairs:
    shutil.copy(os.path.join(images_dir, img), os.path.join(val_images_dir, img))
    shutil.copy(os.path.join(masks_dir, mask), os.path.join(val_masks_dir, mask))

print(f"Treino: {len(train_pairs)} pares")
print(f"Validação: {len(val_pairs)} pares")

