🧠 3D Brain MRI Anomaly Detection using a 3D Convolutional Autoencoder

I built my first Neural Network by exploring the complexity of the human brain through deep learning.

This project implements a 3D Convolutional Autoencoder capable of reconstructing brain MRI volumes and identifying potential abnormalities through reconstruction error.

The key idea:

Train the model on healthy brain MRI scans

Let the network learn normal anatomical structure

During inference, large reconstruction errors indicate anomalies

This approach can help in early detection of neurodevelopmental disorders by identifying irregularities in brain scans.

🧩 Working of the Autoencoder

The system uses a 3D Convolutional Autoencoder consisting of two parts:

Encoder

Compresses the 3D MRI volume into a latent vector representation.

Decoder

Attempts to reconstruct the original brain scan from the compressed vector.

The model learns by minimizing the difference between the original and reconstructed scan.

🏗 Model Architecture

Architecture Highlights

3D Convolution Layers

Latent Bottleneck Representation

Group Normalization

ReLU Activation

Sigmoid Output Layer

Loss Function

The network is optimized using Mean Squared Error (MSE) during backpropagation.

⚙️ Project Pipeline

The project pipeline consists of:

1️⃣ MRI Dataset Loader

Loads 3D NIfTI (.nii.gz) brain scans.

2️⃣ Preprocessing

Percentile normalization

Resizing MRI volumes

Conversion to PyTorch tensors

3️⃣ Model Training

3D convolutional encoder–decoder

Latent bottleneck representation

Backpropagation using MSE loss

4️⃣ Evaluation Metrics

Model performance evaluated using:

MAE (Mean Absolute Error)

RMSE (Root Mean Squared Error)

SSIM (Structural Similarity Index)

Voxel-wise reconstruction error maps