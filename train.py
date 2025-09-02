import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import cv2
import numpy as np
import os
import json
import matplotlib
matplotlib.use('Agg')   
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import logging
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import argparse
from pathlib import Path
import wandb   

from linenet import create_linenet
 

def create_error_visualizations(images, masks, predictions, num_samples=4):
    """Mostra imagem, GT, predição e erro absoluto"""
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples))
    
    for i in range(min(num_samples, len(images))):
        img = images[i].cpu().numpy().transpose(1, 2, 0)
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)

        mask = masks[i].cpu().numpy().squeeze()
        pred = predictions[i].cpu().numpy().squeeze()
        error = np.abs(mask - pred)
        
        axes[i, 0].imshow(img)
        axes[i, 0].set_title('Input Image')
        axes[i, 1].imshow(mask, cmap='gray')
        axes[i, 1].set_title('Ground Truth')
        axes[i, 2].imshow(pred, cmap='gray')
        axes[i, 2].set_title('Prediction')
        axes[i, 3].imshow(error, cmap='hot')
        axes[i, 3].set_title('Error Map')
        
        for ax in axes[i]:
            ax.axis('off')
    
    plt.tight_layout()
    return fig


class LineDataset(Dataset):
    """Dataset for line detection with masks"""
    
    def __init__(self, images_dir: str, masks_dir: str, transform=None, 
                 image_size: Tuple[int, int] = (224, 224)):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.transform = transform
        self.image_size = image_size
        
        # Get all image files
        self.image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            self.image_files.extend(list(self.images_dir.glob(ext)))
        
        # Filter images that have corresponding masks
        self.valid_files = []
        for img_file in self.image_files:
            mask_file = self.masks_dir / f"{img_file.stem}.png"
            if mask_file.exists():
                self.valid_files.append((img_file, mask_file))
        
        print(f"Found {len(self.valid_files)} valid image-mask pairs")
    
    def __len__(self):
        return len(self.valid_files)


    
    def __getitem__(self, idx):
        img_path, mask_path = self.valid_files[idx]
        
        # Load image and mask
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #image = apply_clahe(image)
        
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        # Resize
        image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, self.image_size, interpolation=cv2.INTER_NEAREST)
        
        # Normalize mask to [0, 1]
        mask = mask.astype(np.float32) / 255.0
        mask = np.expand_dims(mask, axis=0)  # Add channel dimension
        
        if self.transform:
            # Apply transforms using albumentations
            transformed = self.transform(image=image, mask=mask.squeeze(0))
            image = transformed['image']
            mask = transformed['mask']
            mask = np.expand_dims(mask, axis=0)
        else:
            # Convert to tensor manually
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            mask = torch.from_numpy(mask).float()
        
        return image, mask

def get_transforms_original(image_size: Tuple[int, int], is_training: bool = True):
    """Get data augmentation transforms"""
    
    if is_training:
        transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1, 
                scale_limit=0.1, 
                rotate_limit=15, 
                p=0.5
            ),
            A.OneOf([
                A.GaussNoise(var_limit=0.1),
                A.GaussianBlur(blur_limit=3),
                A.MotionBlur(blur_limit=3),
            ], p=0.3),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            ], p=0.5),
            #novo
            A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.7),
            A.CLAHE(p=0.3),  # Realce local de contraste
            A.RandomShadow(p=0.3),  # Simula sombras
            A.RandomSunFlare(p=0.3),  # Simula reflexos solares
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        transform = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    return transform

def get_transforms(image_size=(224, 224), is_training=True):
    if is_training:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.4),
            A.RandomBrightnessContrast(p=0.3),
            A.GaussNoise(var_limit=(5.0, 20.0), p=0.2),
            A.HueSaturationValue(p=0.2),
            A.Resize(*image_size),
            A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(*image_size),
            A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
            ToTensorV2()
        ])

def get_transforms_2(image_size: Tuple[int, int], is_training: bool = True):
 
    
    if is_training:
        transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            #A.VerticalFlip(p=0.2),
            #A.RandomRotate90(p=0.5),
            #A.ShiftScaleRotate(
            #    shift_limit=0.1, 
            #    scale_limit=0.1, 
            #    rotate_limit=15, 
            #    p=0.5
            #)
            
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                #A.GaussianBlur(blur_limit=3, p=0.3),
                #A.MotionBlur(blur_limit=3, p=0.3),
            ], p=0.5),
            A.OneOf([
                #A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.7),
                #A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.5),
                A.CLAHE(clip_limit=4.0, p=0.3),
                #A.RandomShadow(p=0.3),
                #A.RandomSunFlare(src_radius=80, p=0.3),
            ], p=0.7),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        transform = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    return transform


class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def save_checkpoint(self, model):
        self.best_weights = model.state_dict()


class CombinedLoss(nn.Module):
    """Combined loss function for line detection"""
    
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.3, 
                 focal_weight: float = 0.2, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.alpha = alpha
        self.gamma = gamma
        
        self.bce_loss = nn.BCELoss()
    
    def dice_loss(self, pred, target, smooth=1e-6):
        pred = pred.view(-1)
        target = target.view(-1)
        intersection = (pred * target).sum()
        dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
        return 1 - dice
    
    def focal_loss(self, pred, target):
        bce_loss = F.binary_cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()
    
    def forward(self, pred, target):
        bce = self.bce_loss(pred, target)
        dice = self.dice_loss(pred, target)
        focal = self.focal_loss(pred, target)
        
        total_loss = (self.bce_weight * bce + 
                     self.dice_weight * dice + 
                     self.focal_weight * focal)
        
        return total_loss, {
            'bce': bce.item(),
            'dice': dice.item(),
            'focal': focal.item(),
            'total': total_loss.item()
        }

def calculate_metrics(pred, target, threshold=0.5):
    """Calculate evaluation metrics"""
    pred_binary = (pred > threshold).float()
    target_binary = target
    
    # Flatten tensors
    pred_flat = pred_binary.view(-1)
    target_flat = target_binary.view(-1)
    
    # Calculate metrics
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum() - intersection
    
    # IoU
    iou = (intersection + 1e-6) / (union + 1e-6)
    
    # Dice coefficient
    dice = (2 * intersection + 1e-6) / (pred_flat.sum() + target_flat.sum() + 1e-6)
    
    # Precision and Recall
    tp = intersection
    fp = pred_flat.sum() - intersection
    fn = target_flat.sum() - intersection
    
    precision = (tp + 1e-6) / (tp + fp + 1e-6)
    recall = (tp + 1e-6) / (tp + fn + 1e-6)
    
    # F1 Score
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    
    return {
        'iou': iou.item(),
        'dice': dice.item(),
        'precision': precision.item(),
        'recall': recall.item(),
        'f1': f1.item()
    }

def create_prediction_grid(images, masks, predictions, num_samples=4):
    """Create a grid of predictions for visualization"""
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))
    
    for i in range(min(num_samples, len(images))):
        # Denormalize image
        img = images[i].cpu().numpy().transpose(1, 2, 0)
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        
        mask = masks[i].cpu().numpy().squeeze()
        pred = predictions[i].cpu().numpy().squeeze()
        
        axes[i, 0].imshow(img)
        axes[i, 0].set_title('Input Image')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(mask, cmap='gray')
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(pred, cmap='gray')
        axes[i, 2].set_title('Prediction')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    return fig

def train_epoch(model, dataloader, optimizer, criterion, device, scheduler=None):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    running_metrics = {'bce': 0.0, 'dice': 0.0, 'focal': 0.0}
    running_eval_metrics = {'iou': 0.0, 'dice': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    
    pbar = tqdm(dataloader, desc='Training')
    
    for batch_idx, (images, masks) in enumerate(pbar):
        images, masks = images.to(device), masks.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(images)
        loss, loss_components = criterion(outputs, masks)
        
        loss.backward()
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
        
        # Update metrics
        running_loss += loss.item()
        for key in running_metrics:
            running_metrics[key] += loss_components[key]
        
        # Calculate evaluation metrics
        eval_metrics = calculate_metrics(outputs, masks)
        for key in running_eval_metrics:
            running_eval_metrics[key] += eval_metrics[key]
        
        # Update progress bar
        avg_loss = running_loss / (batch_idx + 1)
        avg_iou = running_eval_metrics['iou'] / (batch_idx + 1)
        pbar.set_postfix({'Loss': f'{avg_loss:.4f}', 'IoU': f'{avg_iou:.4f}'})
    
    # Calculate averages
    epoch_loss = running_loss / len(dataloader)
    epoch_metrics = {key: val / len(dataloader) for key, val in running_metrics.items()}
    epoch_eval_metrics = {key: val / len(dataloader) for key, val in running_eval_metrics.items()}
    
    return epoch_loss, epoch_metrics, epoch_eval_metrics

def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    running_loss = 0.0
    running_metrics = {'bce': 0.0, 'dice': 0.0, 'focal': 0.0}
    running_eval_metrics = {'iou': 0.0, 'dice': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    
    all_predictions = []
    all_images = []
    all_masks = []
    
    pbar = tqdm(dataloader, desc='Validation')
    
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(pbar):
            images, masks = images.to(device), masks.to(device)
            
            outputs = model(images)
            loss, loss_components = criterion(outputs, masks)
            
            # Update metrics
            running_loss += loss.item()
            for key in running_metrics:
                running_metrics[key] += loss_components[key]
            
            # Calculate evaluation metrics
            eval_metrics = calculate_metrics(outputs, masks)
            for key in running_eval_metrics:
                running_eval_metrics[key] += eval_metrics[key]
            
            # Store for visualization (only first batch)
            if batch_idx == 0:
                all_images = images[:8]  # Store up to 8 images
                all_masks = masks[:8]
                all_predictions = outputs[:8]
            
            # Update progress bar
            avg_loss = running_loss / (batch_idx + 1)
            avg_iou = running_eval_metrics['iou'] / (batch_idx + 1)
            pbar.set_postfix({'Loss': f'{avg_loss:.4f}', 'IoU': f'{avg_iou:.4f}'})
    
    # Calculate averages
    epoch_loss = running_loss / len(dataloader)
    epoch_metrics = {key: val / len(dataloader) for key, val in running_metrics.items()}
    epoch_eval_metrics = {key: val / len(dataloader) for key, val in running_eval_metrics.items()}
    
    return epoch_loss, epoch_metrics, epoch_eval_metrics, (all_images, all_masks, all_predictions)

def save_checkpoint(model, optimizer, scheduler, epoch, best_metric, save_path):
    """Save model checkpoint"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'best_metric': best_metric,
    }, save_path)

def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint['epoch'], checkpoint['best_metric']

def plot_training_curves(train_losses, val_losses, train_metrics, val_metrics, save_path):
    """Plot and save training curves"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Loss curves
    axes[0, 0].plot(train_losses, label='Train Loss', color='blue')
    axes[0, 0].plot(val_losses, label='Val Loss', color='red')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # IoU curves
    train_iou = [m['iou'] for m in train_metrics]
    val_iou = [m['iou'] for m in val_metrics]
    axes[0, 1].plot(train_iou, label='Train IoU', color='blue')
    axes[0, 1].plot(val_iou, label='Val IoU', color='red')
    axes[0, 1].set_title('IoU Score')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('IoU')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Dice curves
    train_dice = [m['dice'] for m in train_metrics]
    val_dice = [m['dice'] for m in val_metrics]
    axes[0, 2].plot(train_dice, label='Train Dice', color='blue')
    axes[0, 2].plot(val_dice, label='Val Dice', color='red')
    axes[0, 2].set_title('Dice Coefficient')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Dice')
    axes[0, 2].legend()
    axes[0, 2].grid(True)
    
    # Precision curves
    train_precision = [m['precision'] for m in train_metrics]
    val_precision = [m['precision'] for m in val_metrics]
    axes[1, 0].plot(train_precision, label='Train Precision', color='blue')
    axes[1, 0].plot(val_precision, label='Val Precision', color='red')
    axes[1, 0].set_title('Precision')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Recall curves
    train_recall = [m['recall'] for m in train_metrics]
    val_recall = [m['recall'] for m in val_metrics]
    axes[1, 1].plot(train_recall, label='Train Recall', color='blue')
    axes[1, 1].plot(val_recall, label='Val Recall', color='red')
    axes[1, 1].set_title('Recall')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    # F1 curves
    train_f1 = [m['f1'] for m in train_metrics]
    val_f1 = [m['f1'] for m in val_metrics]
    axes[1, 2].plot(train_f1, label='Train F1', color='blue')
    axes[1, 2].plot(val_f1, label='Val F1', color='red')
    axes[1, 2].set_title('F1 Score')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('F1')
    axes[1, 2].legend()
    axes[1, 2].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def setup_logging(log_dir):
    """Setup logging configuration"""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'training.log'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Train LineNet for line detection')
    
    # Data arguments
    parser.add_argument('--train_images', type=str, required=True, help='Path to training images')
    parser.add_argument('--train_masks', type=str, required=True, help='Path to training masks')
    parser.add_argument('--val_images', type=str, help='Path to validation images')
    parser.add_argument('--val_masks', type=str, help='Path to validation masks')
    parser.add_argument('--val_split', type=float, default=0.2, help='Validation split ratio if no val data provided')
    
    # Model arguments
    parser.add_argument('--variant', type=str, default='small', 
                       choices=['nano', 'lite', 'small', 'medium', 'strong'],
                       help='Model variant')
    parser.add_argument('--input_channels', type=int, default=3, help='Input channels')
    parser.add_argument('--num_classes', type=int, default=1, help='Number of output classes')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--image_size', type=int, nargs=2, default=[224, 224], help='Image size (H W)')
    
    # Loss arguments
    parser.add_argument('--bce_weight', type=float, default=0.5, help='BCE loss weight')
    parser.add_argument('--dice_weight', type=float, default=0.3, help='Dice loss weight')
    parser.add_argument('--focal_weight', type=float, default=0.2, help='Focal loss weight')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Output directory')
    parser.add_argument('--experiment_name', type=str, default='linenet_experiment', help='Experiment name')
    parser.add_argument('--save_every', type=int, default=5, help='Save checkpoint every N epochs')
    parser.add_argument('--visualize_every', type=int, default=2, help='Create visualizations every N epochs')
    
    # Resume training
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    
    # Device
    parser.add_argument('--device', type=str, default='auto', help='Device to use (auto, cpu, cuda)')
    
    # Wandb logging (optional)
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='linenet', help='Wandb project name')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Setup output directory
    output_dir = Path(args.output_dir) / args.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(output_dir)
    logger.info(f"Starting training experiment: {args.experiment_name}")
    logger.info(f"Arguments: {vars(args)}")
    
    # Save arguments
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Initialize wandb if requested
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.experiment_name,
            config=vars(args)
        )
    
    # Create model
    model = create_linenet(
        variant=args.variant,
        num_classes=args.num_classes,
        input_channels=args.input_channels
    ).to(device)
    
    # Log model info
    model_info = model.get_model_info()
    logger.info(f"Model info: {model_info}")
    
    # Create datasets and dataloaders
    train_transform = get_transforms(tuple(args.image_size), is_training=True)
    val_transform = get_transforms(tuple(args.image_size), is_training=False)
    
    # Check if separate validation data is provided
    if args.val_images and args.val_masks:
        train_dataset = LineDataset(args.train_images, args.train_masks, 
                                   transform=train_transform, image_size=tuple(args.image_size))
        val_dataset = LineDataset(args.val_images, args.val_masks,
                                 transform=val_transform, image_size=tuple(args.image_size))
    else:
        # Split training data
        full_dataset = LineDataset(args.train_images, args.train_masks,
                                  transform=None, image_size=tuple(args.image_size))
        
        # Split indices
        dataset_size = len(full_dataset)
        val_size = int(args.val_split * dataset_size)
        train_size = dataset_size - val_size
        
        indices = torch.randperm(dataset_size).tolist()
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        # Create subset datasets
        from torch.utils.data import Subset
        train_dataset = Subset(full_dataset, train_indices)
        val_dataset = Subset(full_dataset, val_indices)
        
        # Apply transforms
        train_dataset.dataset.transform = train_transform
        val_dataset.dataset.transform = val_transform
    
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Create loss function
    criterion = CombinedLoss(
        bce_weight=args.bce_weight,
        dice_weight=args.dice_weight,
        focal_weight=args.focal_weight
    )
    
    # Create optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    early_stopping = EarlyStopping(patience=20, min_delta=0.001)

    # Training state
    start_epoch = 0
    best_iou = 0.0
    train_losses = []
    val_losses = []
    train_metrics = []
    val_metrics = []

    
    # Resume from checkpoint if provided
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        start_epoch, best_iou = load_checkpoint(args.resume, model, optimizer, scheduler)
        logger.info(f"Resumed from epoch {start_epoch}, best IoU: {best_iou:.4f}")
    
    # Training loop
    logger.info("Starting training...")
    
    for epoch in range(start_epoch, args.epochs):
        logger.info(f"\nEpoch {epoch+1}/{args.epochs}")
        logger.info(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Train
        train_loss, train_loss_components, train_eval_metrics = train_epoch(
            model, train_loader, optimizer, criterion, device, scheduler
        )
        
        # Validate
        val_loss, val_loss_components, val_eval_metrics, viz_data = validate_epoch(
            model, val_loader, criterion, device
        )
        
        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_metrics.append(train_eval_metrics)
        val_metrics.append(val_eval_metrics)
        val_dice = val_eval_metrics["dice"]
        
        # Log metrics
        logger.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        logger.info(f"Train IoU: {train_eval_metrics['iou']:.4f}, Val IoU: {val_eval_metrics['iou']:.4f}")
        logger.info(f"Train Dice: {train_eval_metrics['dice']:.4f}, Val Dice: {val_eval_metrics['dice']:.4f}")
        logger.info(f"Train F1: {train_eval_metrics['f1']:.4f}, Val F1: {val_eval_metrics['f1']:.4f}")
        
        # Wandb logging
        if args.use_wandb:
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_iou': train_eval_metrics['iou'],
                'val_iou': val_eval_metrics['iou'],
                'train_dice': train_eval_metrics['dice'],
                'val_dice': val_eval_metrics['dice'],
                'train_f1': train_eval_metrics['f1'],
                'val_f1': val_eval_metrics['f1'],
                'learning_rate': optimizer.param_groups[0]['lr']
            })
        
        #if early_stopping(val_loss, model):
        if early_stopping(val_dice, model):

            logger.info(f"Early stopping triggered at epoch {epoch+1}")
            break

        # Save best model
        current_iou = val_eval_metrics['iou']
        if current_iou > best_iou:
            best_iou = current_iou
            best_model_path = output_dir / 'best_model.pth'
            save_checkpoint(model, optimizer, scheduler, epoch, best_iou, best_model_path)
            logger.info(f"New best model saved! IoU: {best_iou:.4f}")
        
        # Save checkpoint every N epochs
        if (epoch + 1) % args.save_every == 0:
            checkpoint_path = output_dir / f'checkpoint_epoch_{epoch+1}.pth'
            save_checkpoint(model, optimizer, scheduler, epoch, best_iou, checkpoint_path)
            logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        # Create visualizations every N epochs
        if (epoch + 1) % args.visualize_every == 0:
            images, masks, predictions = viz_data

            if len(images) > 0:
                viz_samples = min(4, len(images))


                # Erro visual
                fig_err = create_error_visualizations(images, masks, predictions, viz_samples)
                err_path = output_dir / f'errors_epoch_{epoch+1}.png'
                fig_err.savefig(err_path, dpi=300, bbox_inches='tight')
                plt.close(fig_err)
                logger.info(f"Erro visual guardado: {err_path}")

                fig = create_prediction_grid(images, masks, predictions, viz_samples)
                
                viz_path = output_dir / f'predictions_epoch_{epoch+1}.png'
                fig.savefig(viz_path, dpi=300, bbox_inches='tight')
                plt.close(fig)


                
                if args.use_wandb:
                    wandb.log({f'predictions_epoch_{epoch+1}': wandb.Image(str(viz_path))})
                
                logger.info(f"Visualizations saved: {viz_path}")
        
        # Plot training curves
        if (epoch + 1) % 5 == 0:
            curves_path = output_dir / f'training_curves_epoch_{epoch+1}.png'
            plot_training_curves(train_losses, val_losses, train_metrics, val_metrics, curves_path)
            
            if args.use_wandb:
                wandb.log({'training_curves': wandb.Image(str(curves_path))})
    
    # Final save
    final_model_path = output_dir / 'final_model.pth'
    save_checkpoint(model, optimizer, scheduler, args.epochs-1, best_iou, final_model_path)
    
    # Final training curves
    final_curves_path = output_dir / 'final_training_curves.png'
    plot_training_curves(train_losses, val_losses, train_metrics, val_metrics, final_curves_path)
    
    logger.info("Training completed!")
    logger.info(f"Best IoU achieved: {best_iou:.4f}")
    logger.info(f"Models saved in: {output_dir}")
    

    if args.use_wandb:
        wandb.finish()
#python train.py   --train_images /home/djoker/code/cuda/mixdataset/images   --train_masks /home/djoker/code/cuda/mixdataset/masks   --variant small   --batch_size 4  --epochs 200   --learning_rate 1e-3   --image_size 224 224   --visualize_every 2  --experiment_name small

#python train.py --train_images /home/djoker/code/cuda/mixdataset/images   --train_masks /home/djoker/code/cuda/mixdataset/masks  --image_size 224 224   --visualize_every 2  --variant nano   --batch_size 2  --epochs 50   --learning_rate 1e-3   --experiment_name nano

#python3 train.py --train_images /home/djoker/code/cuda/RoadSegmentation/dataset/images   --train_masks /home/djoker/code/cuda/mixdataset/masks   --variant small   --batch_size 4   --epochs 100   --learning_rate 1e-3   #--image_size 224 224   --visualize_every 5  --experiment_name small

#python3 train.py --train_images /home/djoker/code/cuda/mixdataset/images   --train_masks /home/djoker/code/cuda/RoadSegmentation/dataset/masks   --variant nano   --batch_size 2   --epochs 100   --learning_rate 1e-3   --image_size 160 160   --visualize_every 5  --experiment_name nano

#python3 train.py --train_images /home/djoker/code/cuda/mixdataset/images   --train_masks /home/djoker/code/cuda/mixdataset/masks  --variant nano   --batch_size 2   --epochs 100   --learning_rate 1e-3   --image_size 160 160   --visualize_every 5  --experiment_name nano

#python3 train.py --train_images /home/djoker/code/cuda/mixdataset/images   --train_masks /home/djoker/code/cuda/mixdataset/masks  --variant small   --batch_size 2   --epochs 100   --learning_rate 1e-3   --image_size 224 224   --visualize_every 5  --experiment_name small


if __name__ == '__main__':
    main()
