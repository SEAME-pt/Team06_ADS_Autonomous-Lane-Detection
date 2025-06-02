import torch
import os
from model import TwinLite as net

def extract_best_model(checkpoint_path, output_path):
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at '{checkpoint_path}'")

    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    
    model = net.TwinLiteNet()
    
    if 'best_weights' in checkpoint:
        print("Found 'best_weights' in checkpoint. Loading best model weights.")
        model.load_state_dict(checkpoint['best_weights'])
    else:
        print("No 'best_weights' found. Loading 'state_dict' as fallback.")
        model.load_state_dict(checkpoint['state_dict'])
    
    model.eval()
    
    torch.save(model.state_dict(), output_path)
    print(f"Best model weights saved to '{output_path}'")

if __name__ == '__main__':
    checkpoint_path = './test_/checkpoint.pth.tar'  # Adjust path as needed
    output_path = './test_/model.pth'         # Adjust output path as needed
    extract_best_model(checkpoint_path, output_path)
