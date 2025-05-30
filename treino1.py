import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from model import TwinLite as net
import torchvision.transforms as transforms
from tqdm import tqdm
from DataSet import RoadDataset


def train_model():
    # Configurações
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-3
    EPOCHS = 100
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Transformações de dados
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Dataset
    dataset = RoadDataset('/home/djoker/code/cuda/mixdataset/images', '/home/djoker/code/cuda/mixdataset/masks', transform=transform)
    
    # Divisão treino/validação
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    print(train_size)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    
    # Modelo
    model =  net.TwinLiteNet()
    model = model.cuda()
    
    # Loss e otimizador
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
    # Treinamento
    best_val_loss = float('inf')
    
    for epoch in range(EPOCHS):
        # Treino
        model.train()
        train_loss = 0.0
        
        for batch_idx, (images, masks) in enumerate(tqdm(train_loader)):
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validação
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(DEVICE), masks.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f'Epoch {epoch+1}/{EPOCHS}:')
        print(f'Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        
        # Salvar melhor modelo
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_twinlitenet_road.pth')
            print('Modelo salvo!')
        
        scheduler.step()

if __name__ == '__main__':
    train_model()
