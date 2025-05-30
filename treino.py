import os

import matplotlib
matplotlib.use('Agg')  # Backend não-interativo
os.environ['MPLBACKEND'] = 'Agg'
os.environ['DISPLAY'] = ''

# Desabilitar GUI warnings
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from model import TwinLite as net
import torch.backends.cudnn as cudnn
#from DataSet import CustomLaneDataset
#from JetSonDataSet import CustomLaneDataset
from DataSet import JetsonDataSet
from argparse import ArgumentParser
from utils import train, val, netParams, save_checkpoint, poly_lr_scheduler
import torch.optim.lr_scheduler

from loss import TotalLoss

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
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
        elif val_loss < self.best_loss - self.min_delta:
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
        """Save model when validation loss decreases."""
        self.best_weights = model.state_dict().copy()
def create_prediction_visualization(model, val_dataset, device, epoch, save_dir, num_samples=8):
    """Create a visualization showing original images, ground truth, and predictions"""
    model.eval()
    
    # Create directory for visualizations
    vis_dir = os.path.join(save_dir, 'visualizations')
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    
    # Randomly sample images from validation set
    indices = np.random.choice(len(val_dataset), min(num_samples, len(val_dataset)), replace=False)
    
    fig, axes = plt.subplots(3, num_samples, figsize=(2*num_samples, 6))
    if num_samples == 1:
        axes = axes.reshape(-1, 1)
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            # Get sample - handle the 3-value return (image_name, image, (seg_da, seg_ll))
            sample = val_dataset[idx]
            if len(sample) == 3:
                image_name, image, mask = sample
            else:
                image, mask = sample
            
            # Convert image to float and normalize
            if image.dtype == torch.uint8:
                image = image.float() / 255.0
            
            image_batch = image.unsqueeze(0).to(device)
            
            # Get prediction
            with torch.no_grad():
                da_predict, ll_predict = model(image_batch)
                
            # Convert tensors to numpy for visualization
            image_np = image.cpu().numpy().transpose(1, 2, 0)
            # Ensure image is in [0,1] range for display
            if image_np.max() > 1:
                image_np = image_np / 255.0
            image_np = np.clip(image_np, 0, 1)
            
            # Handle mask - your dataset returns (seg_da, seg_ll) tuple
            if isinstance(mask, (list, tuple)):
                seg_da, seg_ll = mask
                
                # Convert ground truth masks to visualization format
                # seg_da and seg_ll are 2-channel tensors (background, foreground)
                da_gt = seg_da[1].cpu().numpy()  # foreground channel
                ll_gt = seg_ll[1].cpu().numpy()  # foreground channel
                
                # Create RGB ground truth visualization
                gt_mask = np.zeros((da_gt.shape[0], da_gt.shape[1], 3))
                gt_mask[:, :, 0] = da_gt  # Red channel for driving area
                gt_mask[:, :, 1] = ll_gt  # Green channel for lane lines
            else:
                # Handle single mask case
                gt_mask = mask.cpu().numpy()
                if len(gt_mask.shape) == 3 and gt_mask.shape[0] > 1:
                    gt_mask = gt_mask.transpose(1, 2, 0)
                elif len(gt_mask.shape) == 2:
                    gt_mask = np.stack([gt_mask, gt_mask, gt_mask], axis=-1)
            
            # Convert predictions to numpy
            # Apply softmax and get the foreground probability
            da_pred = torch.softmax(da_predict, dim=1)[0, 1].cpu().numpy()  # Foreground probability
            ll_pred = torch.softmax(ll_predict, dim=1)[0, 1].cpu().numpy()  # Foreground probability
            
            # Create RGB prediction visualization
            pred_mask = np.zeros((da_pred.shape[0], da_pred.shape[1], 3))
            pred_mask[:, :, 0] = da_pred > 0.5  # Red channel for driving area
            pred_mask[:, :, 1] = ll_pred > 0.5  # Green channel for lane lines
            
            # Plot
            axes[0, i].imshow(image_np)
            axes[0, i].set_title(f'Original {i+1}')
            axes[0, i].axis('off')
            
            axes[1, i].imshow(gt_mask)
            axes[1, i].set_title('Ground Truth')
            axes[1, i].axis('off')
            
            axes[2, i].imshow(pred_mask)
            axes[2, i].set_title('Prediction')
            axes[2, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, f'predictions_epoch_{epoch:03d}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved for epoch {epoch}")
def gradual_unfreezing(model, epoch, unfreeze_epoch):
    """
    Gradually unfreeze layers during training
    Args:
        model: The model
        epoch: Current epoch
        unfreeze_epoch: Epoch at which to start unfreezing
    """
    if epoch == unfreeze_epoch:
        print(f"=> Unfreezing all layers at epoch {epoch}")
        for param in model.parameters():
            param.requires_grad = True
        return True
    return False

def load_pretrained_weights(model, pretrained_path, strict=False):
    """
    Load pretrained weights from a .pth file
    Args:
        model: The model to load weights into
        pretrained_path: Path to the .pth file
        strict: Whether to strictly enforce that the keys in state_dict match the keys returned by this module's state_dict()
    """
    if os.path.isfile(pretrained_path):
        print(f"=> Loading pretrained weights from '{pretrained_path}'")
        
        # Load the state dict
        pretrained_dict = torch.load(pretrained_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'state_dict' in pretrained_dict:
            pretrained_dict = pretrained_dict['state_dict']
        elif 'model' in pretrained_dict:
            pretrained_dict = pretrained_dict['model']
        
        # Get current model state dict
        model_dict = model.state_dict()
        
        if not strict:
            # Filter out unnecessary keys and size mismatches
            filtered_dict = {}
            for k, v in pretrained_dict.items():
                # Remove 'module.' prefix if present (from DataParallel)
                key = k.replace('module.', '') if k.startswith('module.') else k
                
                if key in model_dict:
                    if model_dict[key].shape == v.shape:
                        filtered_dict[key] = v
                        print(f"✓ Loaded: {key} {v.shape}")
                    else:
                        print(f"✗ Shape mismatch for {key}: model={model_dict[key].shape}, pretrained={v.shape}")
                else:
                    print(f"✗ Key not found in model: {key}")
            
            pretrained_dict = filtered_dict
            
            # Update model dict
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            print(f"=> Loaded {len(pretrained_dict)}/{len(model_dict)} layers from pretrained weights")
        else:
            model.load_state_dict(pretrained_dict, strict=True)
            print("=> Loaded all pretrained weights (strict mode)")
    else:
        raise FileNotFoundError(f"=> No pretrained weights found at '{pretrained_path}'")

def train_net(args):
    # load the model
    cuda_available = torch.cuda.is_available()
    num_gpus = torch.cuda.device_count()
    model = net.TwinLiteNet()
    
    # Load pretrained weights if specified
    if args.pretrained:
        load_pretrained_weights(model, args.pretrained, strict=args.strict_loading)

    if num_gpus > 1:
        model = torch.nn.DataParallel(model)

    args.savedir = args.savedir + '/'

    # create the directory if not exist
    if not os.path.exists(args.savedir):

        os.mkdir(args.savedir)
 

    train_dataset = JetsonDataSet(valid=False,color_augmentation=True)
    val_dataset = JetsonDataSet(valid=True,color_augmentation=False)


    trainLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    valLoader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    if cuda_available:
        args.onGPU = True
        model = model.cuda()
        cudnn.benchmark = True
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    total_paramters = netParams(model)
    print('Total network parameters: ' + str(total_paramters))

    criteria = TotalLoss()

    start_epoch = 0
    lr = args.lr


    # Freeze layers if specified for transfer learning
    if args.freeze_backbone and args.pretrained:
        print("=> Freezing backbone layers for transfer learning")
        for name, param in model.named_parameters():
            # Ajustar nomes conforme sua arquitetura específica
            if any(layer in name for layer in ['encoder', 'backbone', 'features', 'conv1', 'conv2', 'conv3']):
                if not any(head in name for head in ['classifier', 'segmentation_head', 'decode', 'head', 'final']):
                    param.requires_grad = False
                    print(f"Frozen: {name}")

    # if args.freeze_backbone and args.pretrained:
    #     print("=> Freezing backbone layers for transfer learning")
    #     for name, param in model.named_parameters():
    #         if 'classifier' not in name and 'segmentation_head' not in name and 'decode' not in name:
    #             param.requires_grad = False
    #             print(f"Frozen: {name}")
        
        # Use different learning rates for frozen and unfrozen parts
        unfrozen_params = [p for p in model.parameters() if p.requires_grad]
        frozen_params = [p for p in model.parameters() if not p.requires_grad]
        
        print(f"Unfrozen parameters: {sum(p.numel() for p in unfrozen_params)}")
        print(f"Frozen parameters: {sum(p.numel() for p in frozen_params)}")
        
        optimizer = torch.optim.Adam(unfrozen_params, lr, (0.9, 0.999), eps=1e-08, weight_decay=5e-4)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr, (0.9, 0.999), eps=1e-08, weight_decay=5e-4)
    
    # Initialize Early Stopping
    early_stopping = EarlyStopping(patience=args.patience, min_delta=args.min_delta)
    
    # Lists to store metrics for plotting
    train_losses = []
    val_losses = []
    da_ious = []
    ll_accs = []
    ll_ious = []

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    for epoch in range(start_epoch, args.max_epochs):
        model_file_name = args.savedir + os.sep + 'model_{}.pth'.format(epoch)
        
        # Gradual unfreezing if specified
        if args.freeze_backbone and args.unfreeze_epoch > 0:
            if gradual_unfreezing(model, epoch, args.unfreeze_epoch):
                # Recreate optimizer with all parameters
                optimizer = torch.optim.Adam(model.parameters(), args.lr * 0.1, (0.9, 0.999), eps=1e-08, weight_decay=5e-4)
                print("=> Recreated optimizer with lower learning rate for unfrozen layers")
        
        poly_lr_scheduler(args, optimizer, epoch)
        
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
        print("Learning rate: " + str(lr))
        
        # train for one epoch
        model.train()
        train_loss = train(args, trainLoader, model, criteria, optimizer, epoch)
        
        # validation
        model.eval()
        da_segment_results, ll_segment_results = val(valLoader, model)
        
        # Calculate validation loss (you might need to modify the val function to return loss)
        val_loss = 1 - da_segment_results[2]  # Using 1 - mIOU as a proxy for loss
        
        # Store metrics
        train_losses.append(train_loss if train_loss is not None else 0)
        val_losses.append(val_loss)
        da_ious.append(da_segment_results[2])
        ll_accs.append(ll_segment_results[0])
        ll_ious.append(ll_segment_results[1])

        print(f"Epoch [{epoch+1}/{args.max_epochs}]")
        print(f"Driving Area Segment: mIOU({da_segment_results[2]:.3f})")
        print(f"Lane Line Segment: Acc({ll_segment_results[0]:.3f}) IOU({ll_segment_results[1]:.3f})")
        
        # Save model
        torch.save(model.state_dict(), model_file_name)
        
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr': lr,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'da_ious': da_ious,
            'll_accs': ll_accs,
            'll_ious': ll_ious
        }, args.savedir + 'checkpoint.pth.tar')
        
        # Create visualization every 2 epochs
        if epoch % 2 == 0:
            try:
                create_prediction_visualization(model, val_dataset, device, epoch, args.savedir, args.vis_samples)
            except Exception as e:
                print(f"Warning: Could not create visualization for epoch {epoch}: {e}")
        
        # Plot training curves
        if epoch > 0:
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            plt.plot(train_losses, label='Train Loss')
            plt.plot(val_losses, label='Val Loss')
            plt.title('Training and Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            
            plt.subplot(1, 3, 2)
            plt.plot(da_ious, label='Driving Area mIOU')
            plt.title('Driving Area Segmentation Performance')
            plt.xlabel('Epoch')
            plt.ylabel('mIOU')
            plt.legend()
            plt.grid(True)
            
            plt.subplot(1, 3, 3)
            plt.plot(ll_accs, label='Lane Line Accuracy')
            plt.plot(ll_ious, label='Lane Line IOU')
            plt.title('Lane Line Segmentation Performance')
            plt.xlabel('Epoch')
            plt.ylabel('Score')
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(args.savedir, 'training_curves.png'), dpi=150, bbox_inches='tight')
            plt.close()
        
        # Early stopping check
        if early_stopping(val_loss, model):
            print(f"Early stopping triggered at epoch {epoch + 1}")
            print(f"Best validation loss: {early_stopping.best_loss:.4f}")
            break
    
    print("Training completed!")
    if early_stopping.counter >= early_stopping.patience:
        print(f"Stopped early due to no improvement for {early_stopping.patience} epochs")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--max_epochs', type=int, default=300, help='Max. number of epochs')
    parser.add_argument('--num_workers', type=int, default=6, help='No. of parallel threads')
    parser.add_argument('--batch_size', type=int, default=6, help='Batch size. 12 for ESPNet-C and 6 for ESPNet. Change as per the GPU memory')
    parser.add_argument('--step_loss', type=int, default=100, help='Decrease learning rate after how many epochs.')
    parser.add_argument('--lr', type=float, default=5e-4, help='Initial learning rate')
    parser.add_argument('--savedir', default='./test_', help='directory to save the results')
    parser.add_argument('--resume', type=str, default='', help='Use this flag to load last checkpoint for training')
    parser.add_argument('--pretrained', default='./pretrained/best.pth', help='Path to pretrained .pth weights file')
    parser.add_argument('--strict_loading', action='store_true', help='Use strict loading for pretrained weights')
    parser.add_argument('--freeze_backbone', action='store_true', help='Freeze backbone layers for transfer learning')
    
    # Dataset paths
    parser.add_argument('--train_images', default='dataset/train/images', help='Path to training images')
    parser.add_argument('--train_masks', default='dataset/train/masks', help='Path to training masks')
    parser.add_argument('--val_images', default='dataset/val/images', help='Path to validation images')
    parser.add_argument('--val_masks', default='dataset/val/masks', help='Path to validation masks')
    
    # Early stopping parameters
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience (number of epochs without improvement)')
    parser.add_argument('--min_delta', type=float, default=0.001, help='Minimum change to qualify as improvement')
    
    # Visualization parameters
    parser.add_argument('--vis_samples', type=int, default=8, help='Number of samples to visualize (5 or 8 recommended)')
    
    # Transfer learning parameters
    parser.add_argument('--unfreeze_epoch', type=int, default=0, help='Epoch at which to unfreeze all layers (0 = no gradual unfreezing)')

    train_net(parser.parse_args())


# python train.py     --pretrained ./pretrained/best.pth     --lr 5e-5  

# python treino.py --pretrained ./pretrained/best.pth    --freeze_backbone     --unfreeze_epoch 15     --lr 1e-4     --patience 20