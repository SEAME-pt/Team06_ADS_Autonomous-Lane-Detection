import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from tqdm import tqdm
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score

# ================================
# DATASET PERSONALIZADO
# ================================
class RoadDataset(Dataset):
    def __init__(self, images_dir, masks_dir, img_size=(224, 224), augment=False):
        self.img_size = img_size
        self.augment = augment
        
        # Corrigir caminhos e emparelhar arquivos
        img_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith('.jpg')])
        mask_files = sorted([f for f in os.listdir(masks_dir) if f.lower().endswith('.png')])
        
        img_bases = {os.path.splitext(f)[0]: f for f in img_files}
        mask_bases = {os.path.splitext(f)[0]: f for f in mask_files}
        common_keys = sorted(set(img_bases.keys()) & set(mask_bases.keys()))
        
        self.img_paths = [os.path.join(images_dir, img_bases[k]) for k in common_keys]
        self.mask_paths = [os.path.join(masks_dir, mask_bases[k]) for k in common_keys]
        
        print(f"Dataset carregado: {len(self.img_paths)} pares válidos")
        
        # Transformações para imagem
        self.transform_img = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Transformações para máscara (sem normalização)
        self.transform_mask = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.img_size, interpolation=0),  # NEAREST
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # Carregar imagem
        img = cv2.imread(self.img_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Carregar máscara
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        
        # Augmentação simples
        if self.augment and np.random.rand() > 0.5:
            img = np.fliplr(img).copy()
            mask = np.fliplr(mask).copy()
        
        # Aplicar transformações
        img = self.transform_img(img)
        mask = self.transform_mask(mask)
        
        # Binarizar máscara
        mask = (mask > 0.5).float().squeeze(0)  # Remove canal extra
        
        return img, mask

# ================================
# MODELO TWINLITENET SIMPLES
# ================================
class TwinLiteNet(nn.Module):
    def __init__(self, num_classes=1):
        super(TwinLiteNet, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, 2, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, num_classes, 1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# ================================
# MÉTRICAS
# ================================
def calculate_iou(pred, target, threshold=0.5):
    pred = (torch.sigmoid(pred) > threshold).float()
    target = (target > threshold).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return (intersection / union).item()

def calculate_accuracy(pred, target, threshold=0.5):
    pred = (torch.sigmoid(pred) > threshold).float()
    target = (target > threshold).float()
    correct = (pred == target).float().sum()
    total = target.numel()
    return (correct / total).item()

# ================================
# FUNÇÃO PARA SALVAR PREVISÕES
# ================================
def save_predictions(model, val_loader, device, epoch, save_dir="predictions"):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    
    # Pegar um batch para visualização
    imgs, masks = next(iter(val_loader))
    imgs, masks = imgs.to(device), masks.to(device)
    
    with torch.no_grad():
        outputs = model(imgs)
        preds = torch.sigmoid(outputs)
    
    # Salvar visualizações
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    
    for i in range(min(4, len(imgs))):
        # Desnormalizar imagem para visualização
        img = imgs[i].cpu()
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = img * std + mean
        img = torch.clamp(img, 0, 1)
        
        # Original
        axes[0, i].imshow(img.permute(1, 2, 0))
        axes[0, i].set_title(f'Original {i+1}')
        axes[0, i].axis('off')
        
        # Ground Truth
        axes[1, i].imshow(masks[i].cpu(), cmap='gray')
        axes[1, i].set_title('Ground Truth')
        axes[1, i].axis('off')
        
        # Predição
        axes[2, i].imshow(preds[i, 0].cpu() > 0.5, cmap='gray')
        axes[2, i].set_title(f'Predição (E{epoch})')
        axes[2, i].axis('off')
    
    plt.suptitle(f'Época {epoch} - Comparação', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/predictions_epoch_{epoch:03d}.png', dpi=150, bbox_inches='tight')
    plt.close()

# ================================
# FUNÇÃO PRINCIPAL DE TREINO
# ================================
def train_model():
    # Configurações
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-3
    EPOCHS = 50
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Usando device: {DEVICE}")
    
    # Dataset - CAMINHOS CORRIGIDOS
    dataset = RoadDataset(
        images_dir='/home/djoker/code/cuda/mixdataset/images',  # Sem aspas extras
        masks_dir='/home/djoker/code/cuda/mixdataset/masks',    # Sem aspas extras
        img_size=(224, 224),
        augment=True
    )
    
    # Divisão treino/validação
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"Treino: {train_size} amostras | Validação: {val_size} amostras")
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Modelo
    model = TwinLiteNet(num_classes=1).to(DEVICE)  # 1 classe: segmentação binária
    
    # Loss e otimizador - CORRIGIDO PARA SEGMENTAÇÃO BINÁRIA
    criterion = nn.BCEWithLogitsLoss()  # Não CrossEntropyLoss
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)
    
    # Histórico
    history = {
        'train_loss': [], 'val_loss': [],
        'train_iou': [], 'val_iou': [],
        'train_acc': [], 'val_acc': []
    }
    
    best_val_loss = float('inf')
    
    print("Iniciando treino...")
    
    for epoch in range(EPOCHS):
        # ==================
        # TREINO
        # ==================
        model.train()
        train_loss = 0.0
        train_iou = 0.0
        train_acc = 0.0
        
        for batch_idx, (images, masks) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS}')):
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs.squeeze(1), masks)  # Squeeze para remover dimensão extra
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_iou += calculate_iou(outputs.squeeze(1), masks)
            train_acc += calculate_accuracy(outputs.squeeze(1), masks)
        
        # ==================
        # VALIDAÇÃO
        # ==================
        model.eval()
        val_loss = 0.0
        val_iou = 0.0
        val_acc = 0.0
        
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(DEVICE), masks.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs.squeeze(1), masks)
                
                val_loss += loss.item()
                val_iou += calculate_iou(outputs.squeeze(1), masks)
                val_acc += calculate_accuracy(outputs.squeeze(1), masks)
        
        # Médias
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        avg_train_iou = train_iou / len(train_loader)
        avg_val_iou = val_iou / len(val_loader)
        avg_train_acc = train_acc / len(train_loader)
        avg_val_acc = val_acc / len(val_loader)
        
        # Salvar histórico
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_iou'].append(avg_train_iou)
        history['val_iou'].append(avg_val_iou)
        history['train_acc'].append(avg_train_acc)
        history['val_acc'].append(avg_val_acc)
        
        print(f'Época {epoch+1}/{EPOCHS}:')
        print(f'  Train - Loss: {avg_train_loss:.4f}, IoU: {avg_train_iou:.4f}, Acc: {avg_train_acc:.4f}')
        print(f'  Val   - Loss: {avg_val_loss:.4f}, IoU: {avg_val_iou:.4f}, Acc: {avg_val_acc:.4f}')
        
        # Salvar melhor modelo
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_twinlitenet_road.pth')
            print('  → Melhor modelo salvo!')
        
        # Salvar previsões a cada 5 épocas
        if (epoch + 1) % 5 == 0:
            save_predictions(model, val_loader, DEVICE, epoch + 1)
            print(f'  → Previsões salvas para época {epoch + 1}')
        
        scheduler.step(avg_val_loss)
    
    # Salvar modelo final
    torch.save(model.state_dict(), 'final_twinlitenet_road.pth')
    
    # Plotar histórico
    plot_training_history(history)
    
    print("Treino concluído!")
    print(f"Melhor Val Loss: {best_val_loss:.4f}")
    print(f"Melhor Val IoU: {max(history['val_iou']):.4f}")

def plot_training_history(history):
    """Plotar gráficos do treino"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Loss
    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'], label='Validation')
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Época')
    axes[0].legend()
    axes[0].grid(True)
    
    # IoU
    axes[1].plot(history['train_iou'], label='Train')
    axes[1].plot(history['val_iou'], label='Validation')
    axes[1].set_title('IoU')
    axes[1].set_xlabel('Época')
    axes[1].legend()
    axes[1].grid(True)
    
    # Accuracy
    axes[2].plot(history['train_acc'], label='Train')
    axes[2].plot(history['val_acc'], label='Validation')
    axes[2].set_title('Accuracy')
    axes[2].set_xlabel('Época')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    train_model()

