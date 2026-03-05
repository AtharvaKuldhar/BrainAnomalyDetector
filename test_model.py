import torch
import matplotlib.pyplot as plt
from model import BrainAutoEncoder3D
from dataset import BrainDataset
import numpy as np
from skimage.metrics import structural_similarity as ssim

# ---------------- Load dataset ---------------- #
dataset = BrainDataset("data")     # path to your .nii.gz files

# ---------------- Load trained model ---------------- #
model = BrainAutoEncoder3D()
model.load_state_dict(torch.load("100epoch.pth", map_location="cpu"))
model.eval()

# ---------------- Pick one sample ---------------- #
sample, _ = dataset[0]     # first brain volume
sample = sample.unsqueeze(0)  # add batch dimension [1, 1, 64, 64, 64]

# ---------------- Run through autoencoder ---------------- #
with torch.no_grad():
    reconstructed = model(sample)

# ---------------- Convert to numpy ---------------- #
orig = sample[0, 0].cpu().numpy()         # shape [64, 64, 64]
recon = reconstructed[0, 0].cpu().numpy() # same shape

# ---------------- Visualize Original vs Reconstructed ---------------- #
slice_idx = orig.shape[0] // 2 # middle slice
plt.figure(figsize=(10, 4))

plt.subplot(1, 3, 1)
plt.imshow(orig[slice_idx], cmap="gray")
plt.title("Original Slice")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(recon[slice_idx], cmap="gray")
plt.title("Reconstructed Slice")
plt.axis("off")

# ---------------- Compute & Visualize Error Map ---------------- #
error_map = np.abs(orig - recon)
plt.subplot(1, 3, 3)
plt.imshow(error_map[slice_idx], cmap="hot")
plt.title("Reconstruction Error Map")
plt.axis("off")

plt.tight_layout()
plt.show()

# ----------------- Evaluation -----------------
mae = np.mean(np.abs(orig - recon))
rmse = np.sqrt(np.mean((orig - recon)**2))
similarity = ssim(orig[slice_idx], recon[slice_idx], data_range=1.0)

threshold = 0.2
abnormal_ratio = np.mean((np.abs(orig - recon) > threshold))

print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"SSIM: {similarity:.4f}")
print(f"Abnormal voxel ratio: {abnormal_ratio*100:.2f}%")
print(orig.mean(), orig.std(), recon.mean(), recon.std())

