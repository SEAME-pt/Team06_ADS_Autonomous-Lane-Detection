import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional

class DepthwiseSeparableConv(nn.Module):
    """Efficient depthwise separable convolution"""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, 
                 stride: int = 1, padding: int = 1, bias: bool = False):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, 
                                 stride, padding, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return x

class InvertedResidual(nn.Module):
    """MobileNetV2 style inverted residual block"""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, 
                 expand_ratio: int = 6, use_se: bool = False):
        super().__init__()
        self.stride = stride
        self.use_residual = stride == 1 and in_channels == out_channels
        hidden_dim = in_channels * expand_ratio
        
        layers = []
        # Expand
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True)
            ])
        
        # Depthwise
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True)
        ])
        
        # SE block for stronger variants
        if use_se:
            layers.append(SEBlock(hidden_dim))
        
        # Project
        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])
        
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        return self.conv(x)

class SEBlock(nn.Module):
    """Squeeze-and-Excitation block"""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling for multi-scale feature extraction"""
    def __init__(self, in_channels: int, out_channels: int, rates: List[int] = [1, 6, 12, 18]):
        super().__init__()
        self.convs = nn.ModuleList()
        
        # 1x1 conv
        self.convs.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))
        
        # Atrous convs
        for rate in rates[1:]:
            self.convs.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=rate, dilation=rate, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))
        
        # Global average pooling - FIXED: Removed BatchNorm after 1x1 spatial dims
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.ReLU(inplace=True)  # Removed BatchNorm2d here
        )
        
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * (len(rates) + 1), out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
    
    def forward(self, x):
        h, w = x.size()[2:]
        features = []
        
        for conv in self.convs:
            features.append(conv(x))
        
        # Global pooling branch
        global_feat = self.global_pool(x)
        global_feat = F.interpolate(global_feat, size=(h, w), mode='bilinear', align_corners=False)
        features.append(global_feat)
        
        x = torch.cat(features, dim=1)
        return self.project(x)

class LineNet(nn.Module):
    """
    Modular Line Detection Network
    
    Variants:
    - nano: Ultra-lightweight for edge devices
    - lite: Lightweight with good performance
    - small: Balanced performance/efficiency
    - medium: Higher accuracy
    - strong: Maximum accuracy with advanced features
    """
    
    CONFIGS = {
        'nano': {
            'channels': [16, 24, 32, 64, 96],
            'depths': [1, 1, 2, 2, 1],
            'use_se': False,
            'use_aspp': False,
            'expand_ratio': 3,
            'decoder_channels': 32
        },
        'lite': {
            'channels': [24, 32, 48, 96, 160],
            'depths': [1, 2, 2, 3, 1],
            'use_se': False,
            'use_aspp': False,
            'expand_ratio': 4,
            'decoder_channels': 48
        },
        'small': {
            'channels': [32, 48, 64, 128, 192],
            'depths': [2, 2, 3, 4, 2],
            'use_se': True,
            'use_aspp': False,
            'expand_ratio': 4,
            'decoder_channels': 64
        },
        'medium': {
            'channels': [48, 64, 96, 160, 256],
            'depths': [2, 3, 4, 5, 3],
            'use_se': True,
            'use_aspp': True,
            'expand_ratio': 6,
            'decoder_channels': 96
        },
        'strong': {
            'channels': [64, 96, 128, 192, 320],
            'depths': [3, 4, 6, 8, 4],
            'use_se': True,
            'use_aspp': True,
            'expand_ratio': 6,
            'decoder_channels': 128
        }
    }
    
    def __init__(self, variant: str = 'small', num_classes: int = 1, input_channels: int = 3):
        super().__init__()
        
        if variant not in self.CONFIGS:
            raise ValueError(f"Variant {variant} not supported. Choose from {list(self.CONFIGS.keys())}")
        
        config = self.CONFIGS[variant]
        self.variant = variant
        
        # Initial stem
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, config['channels'][0], 3, 2, 1, bias=False),
            nn.BatchNorm2d(config['channels'][0]),
            nn.ReLU6(inplace=True)
        )
        
        # Encoder blocks
        self.encoder = nn.ModuleList()
        in_channels = config['channels'][0]
        
        for i, (out_channels, depth) in enumerate(zip(config['channels'][1:], config['depths'][1:])):
            stage = nn.ModuleList()
            
            # First block with stride 2 (downsampling)
            stage.append(InvertedResidual(
                in_channels, out_channels, stride=2, 
                expand_ratio=config['expand_ratio'], 
                use_se=config['use_se'] and i >= 2  # SE only in later stages
            ))
            
            # Remaining blocks
            for _ in range(depth - 1):
                stage.append(InvertedResidual(
                    out_channels, out_channels, stride=1,
                    expand_ratio=config['expand_ratio'],
                    use_se=config['use_se'] and i >= 2
                ))
            
            self.encoder.append(stage)
            in_channels = out_channels
        
        # ASPP for stronger variants
        if config['use_aspp']:
            self.aspp = ASPP(config['channels'][-1], config['decoder_channels'])
            bottleneck_channels = config['decoder_channels']
        else:
            self.aspp = None
            bottleneck_channels = config['channels'][-1]
        
        # Decoder
        decoder_channels = config['decoder_channels']
        self.decoder = nn.ModuleList()
        
        # Decoder blocks
        for i in range(len(config['channels']) - 1):
            skip_channels = config['channels'][-(i+2)]
            
            if i == 0:
                in_ch = bottleneck_channels
            else:
                in_ch = decoder_channels
            
            self.decoder.append(nn.Sequential(
                nn.ConvTranspose2d(in_ch, decoder_channels, 4, 2, 1, bias=False),
                nn.BatchNorm2d(decoder_channels),
                nn.ReLU(inplace=True),
                DepthwiseSeparableConv(decoder_channels + skip_channels, decoder_channels),
                nn.ReLU(inplace=True)
            ))
        
        # Final upsampling and output
        self.final_upsample = nn.ConvTranspose2d(decoder_channels, decoder_channels, 4, 2, 1)
        self.output = nn.Conv2d(decoder_channels, num_classes, 1)
        
        # Noise reduction module for lab conditions
        self.noise_reduction = nn.Sequential(
            nn.Conv2d(decoder_channels, decoder_channels, 3, 1, 1, groups=decoder_channels),
            nn.Conv2d(decoder_channels, decoder_channels, 1),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # Store input size for final upsampling
        input_size = x.size()[2:]
        
        # Stem
        x = self.stem(x)
        
        # Encoder with skip connections
        skip_connections = [x]
        
        for stage in self.encoder:
            for block in stage:
                x = block(x)
            skip_connections.append(x)
        
        # ASPP if available
        if self.aspp is not None:
            x = self.aspp(x)
        
        # Decoder
        for i, decoder_block in enumerate(self.decoder):
            # Upsample and concatenate with skip connection
            x = decoder_block[0](x)  # Transpose conv
            x = decoder_block[1](x)  # BatchNorm
            x = decoder_block[2](x)  # ReLU
            
            # Skip connection
            skip = skip_connections[-(i+2)]
            if x.size()[2:] != skip.size()[2:]:
                x = F.interpolate(x, size=skip.size()[2:], mode='bilinear', align_corners=False)
            
            x = torch.cat([x, skip], dim=1)
            x = decoder_block[3](x)  # DepthwiseSeparableConv
            x = decoder_block[4](x)  # ReLU
        
        # Final upsampling to input resolution
        x = self.final_upsample(x)
        if x.size()[2:] != input_size:
            x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)
        
        # Noise reduction
        x = self.noise_reduction(x)
        
        # Output
        x = self.output(x)
        
        return torch.sigmoid(x)
    
    def get_model_info(self):
        """Get model information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'variant': self.variant,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024)  # Assuming float32
        }

def create_linenet(variant: str = 'small', num_classes: int = 1, input_channels: int = 3) -> LineNet:
    """Factory function to create LineNet models"""
    return LineNet(variant=variant, num_classes=num_classes, input_channels=input_channels)

# Example usage and model comparison
if __name__ == "__main__":
    # Test all variants
    variants = ['nano', 'lite', 'small', 'medium', 'strong']
    input_tensor = torch.randn(1, 3, 224, 224)
    
    print("LineNet Model Comparison:")
    print("-" * 60)
    
    for variant in variants:
        model = create_linenet(variant=variant)
        model.eval()  # Set to eval mode to avoid BatchNorm issues
        info = model.get_model_info()
        
        with torch.no_grad():
            output = model(input_tensor)
        
        print(f"{variant.upper():>8} | Params: {info['total_params']:>8,} | "
              f"Size: {info['model_size_mb']:>6.2f}MB | "
              f"Output: {output.shape}")
    
    print("-" * 60)
    print("Recommended usage:")
    print("- nano/lite: Edge devices, real-time inference")
    print("- small: Good balance for most applications")
    print("- medium/strong: High accuracy requirements")
    
    # Test with different input sizes
    print("\nTesting different input sizes:")
    model = create_linenet(variant='small')
    model.eval()
    
    for size in [128, 256, 512]:
        test_input = torch.randn(1, 3, size, size)
        with torch.no_grad():
            output = model(test_input)
        print(f"Input: {test_input.shape} -> Output: {output.shape}")
