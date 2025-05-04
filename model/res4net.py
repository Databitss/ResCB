import torch
import torch.nn as nn
import torch.nn.functional as F
from .cbam import *

# Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, block_type='basic'):
        super(ResidualBlock , self).__init__()
        self.block_type = block_type
        # Branch layers - Struktur umum untuk Block 1 - 4
        self.leaky_relu1_branch = nn.LeakyReLU(negative_slope=0.01) # Diubah
        self.conv1_branch = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0 ,bias=False) # Padding corrected to 0
        self.bn1_branch = nn.BatchNorm2d(out_channels)
        self.leaky_relu2_branch = nn.LeakyReLU(negative_slope=0.01) # Diubah
        self.conv2_branch = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2_branch = nn.BatchNorm2d(out_channels)
        self.leaky_relu3_branch = nn.LeakyReLU(negative_slope=0.01) # Diubah
        self.conv3_branch = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding = 0, bias=False) # Padding corrected to 0
        self.bn3_branch = nn.BatchNorm2d(out_channels)

        # Skip Connection layer (Setup)
        # For complex block, stride in skip connection should be 2 to match main branch downsampling
        if block_type == 'complex': 
            self.conv_skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, bias=False) # Stride set to 2 for complex block
        # For basic block, stride 1 if channels are same, else stride 2 if channels change to align with main path stride
        else: 
            if in_channels != out_channels:
                self.conv_skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, bias=False) # Stride 2 if channels change
            else:
                self.conv_skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False) # Stride 1 if channels are same for basic block
        self.bn_skip = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x # Store input for skip connection
        out = self.leaky_relu1_branch(x)
        out = self.conv1_branch(out)
        out = self.bn1_branch(out)
        out = self.leaky_relu2_branch(out)
        out = self.conv2_branch(out)
        out = self.bn2_branch(out)
        out = self.leaky_relu3_branch(out)
        out = self.conv3_branch(out)
        out = self.bn3_branch(out)

        # Skip Connection (Res-block 1 dan Res-block 3)
        if self.block_type == 'complex' or self.conv_skip.stride[0] == 2:
            residual = self.conv_skip(x)
            residual = self.bn_skip(residual)
        else:
            if residual.shape != out.shape:
              residual = F.interpolate(residual, size=out.shape[2:], mode='bilinear', align_corners=False)
        
        out += residual # Jumlah hasil pembobotan residual dijumlahkan
        return out

class Res4Net_CBAM(nn.Module):
    def __init__(self, num_classes=100):
        super(Res4Net_CBAM, self).__init__()

        # Initial Block Input
        self.initial_conv = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False) # Conv
        self.initial_bn = nn.BatchNorm2d(64) # BatchNorm
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # Max-Pooling

        # Residual Blocks with CBAM
        self.residual_block_cbam1 = ResidualBlock(64, 128, block_type='complex')  # ResBlock 1
        self.cbam1 = CBAM(128) # CBAM Block
        self.residual_block_cbam2 = ResidualBlock(128, 256, block_type='basic')   # ResBlock 2
        self.cbam2 = CBAM(256) # CBAM Block
        self.residual_block_cbam3 = ResidualBlock(256, 512, block_type='complex') # ResBlock 3
        self.cbam3 = CBAM(512) # CBAM Block
        self.residual_block_cbam4 = ResidualBlock(512, 512, block_type='basic')   # ResBlock 4
        self.cbam4 = CBAM(512) # CBAM Block

        # Neural Network Layer and Classification
        self.leakyReLu = nn.LeakyReLU(negative_slope=0.01)  # LeakyReLU
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # GAP (Global Adaptive Pooling)
        self.dense = nn.Linear(512, 1024)  # Fully Connected Layer
        self.dropout = nn.Dropout(0.3) # Dropout 30%
        self.dense_output = nn.Linear(1024, num_classes)  # Output layer

    def forward(self, x):
        out = self.initial_conv(x)  # Initial Convolutional Layer
        out = self.initial_bn(out) # Initial BN Layer
        out = self.maxpool(out)  # Initial Max-Pooling layer
        out = self.residual_block_cbam1(out) # ResBlock 1
        out = self.cbam1(out) # CBAM Block 1
        out = self.residual_block_cbam2(out) # ResBlock 2
        out = self.cbam2(out) # CBAM Block 2
        out = self.residual_block_cbam3(out) # ResBlock 3
        out = self.cbam3(out) # CBAM Block 3
        out = self.residual_block_cbam4(out) # ResBlock 4
        out = self.cbam4(out) # CBAM Block 4
        out = self.leakyReLu(out) # LeakyReLu
        out = self.global_avg_pool(out)  # Global Average Pooling Output => shape: (batch, 512, 1, 1)
        out = torch.flatten(out, 1)  # Flatten jadi (batch, 512)
        out = self.dense(out)  # Fully Connected Layers
        out = self.dropout(out) # Dropout 0.3
        out = self.dense_output(out)  # Output layer (Softmax)
        return out
