# test_model.py

import torch
import matplotlib.pyplot as plt
from dataset import BrainDataset
from model import BrainAutoEncoder3D

# ------------------------------
# Load one sample from dataset
# ------------------------------
dataset = BrainDataset("data")
x, _ = dataset[0]           # get the first brain volume (ignore dummy label)
x = x.unsqueeze(0)          # add batch dimension -> [1, 1, 64, 64, 64]

# ------------------------------
# Load the model
# ------------------------------
model = BrainAutoEncoder3D()
model.eval()                # evaluation mode (no dropout/batchnorm updates)

# Forward pass (no gradient tracking)
with torch.no_grad():
    output = model(x)

print("✅ Input shape :", x.shape)
print("✅ Output shape:", output.shape)

# ------------------------------
# Visualization
# ------------------------------

def show_slice(volume, title):
    """Display the middle slice of a 3D volume"""
    slice_idx = volume.shape[-1] // 2  # middle slice
    plt.imshow(volume[0, 0, :, :, slice_idx].cpu(), cmap='gray')
    plt.title(title)
    plt.axis('off')

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
show_slice(x, "Original Slice")
plt.subplot(1, 2, 2)
show_slice(output, "Reconstructed Slice")
plt.tight_layout()
plt.show()
