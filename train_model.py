import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from dataset import BrainDataset
from model import BrainAutoEncoder3D

# ------------------------------------------------------------ #
# TRAINING SCRIPT — 3D Brain Autoencoder
# ------------------------------------------------------------ #

# 1 Load dataset
dataset = BrainDataset("data")        # path to folder with .nii.gz files
loader = DataLoader(dataset, batch_size=2, shuffle=True)

# 2 Setup model, loss, optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = BrainAutoEncoder3D().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 3 training loop
num_epochs = 100
for epoch in range(num_epochs):
    total_loss = 0.0
    model.train()

    for imgs, _ in loader:
        imgs = imgs.to(device)

        # forward pass
        reconstructed = model(imgs)
        loss = criterion(reconstructed, imgs)

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}")

# 4️⃣ Save trained model
torch.save(model.state_dict(), "100epoch.pth")
print("✅ Model training complete and saved as brain_autoencoder_trained.pth")
