import torch
import torch.nn as nn 
# torch.nn is a fundamental module within the PyTorch deep learning framework, 
# providing the core components and functionalities for building and training neural networks.

class BrainAutoEncoder3D(nn.Module):  # nn.Module: This is the base class for all neural network modules in PyTorch.
    def __init__(self):
        super(BrainAutoEncoder3D, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, stride=2, padding=1),  # [16, 32, 32, 32]
            nn.GroupNorm(8, 16),
            nn.ReLU(True),
            
            nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1), # [32, 16, 16, 16]
            nn.GroupNorm(8, 32),
            nn.ReLU(True),
            
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1), # [64, 8, 8, 8]
            nn.GroupNorm(8, 64),
            nn.ReLU(True)
        )

        # Decoder 
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1), # [32, 16, 16, 16]
            nn.GroupNorm(8, 32),
            nn.ReLU(True),
            
            nn.ConvTranspose3d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1), # [16, 32, 32, 32]
            nn.GroupNorm(8, 16),
            nn.ReLU(True),
            
            nn.ConvTranspose3d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # [1, 64, 64, 64]
            nn.Sigmoid()  # because voxel intensities are normalized between 0 and 1
        )
    
    def forward(self, x):                   # Actual implementation  
        encoded = self.encoder(x)           # takes x -> input brain volume
        decoded = self.decoder(encoded)     # encodes it then decodes using encoded data
        return decoded                      # and returns reconstructed output