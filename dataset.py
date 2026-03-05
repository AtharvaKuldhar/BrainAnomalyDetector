import nibabel as nib                   # reads .nii and .nii.gz medical images
import numpy as np                      # array operations
import torch                            # pytorch deep learning framework
from torch.utils.data import Dataset
import glob                             # used to read file via paths
import torch.nn.functional as F

# ------------------------------------------------------------ #
# All the data preprocessing is done here
# Brain scans are loaded cleaned resized and returned as PyTorch Tensors for training
# ------------------------------------------------------------ #

class BrainDataset(Dataset):
    def __init__(self, path):
        self.files = sorted(glob.glob(path + "/*.nii.gz"))
        # validation
        if(len(self.files) == 0):
            raise FileNotFoundError("No .nii.gz files")
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        # load one file
        file = self.files[idx]
        img = nib.load(file).get_fdata()

        # replace NaNs if any 
        img = np.nan_to_num(img)

        # normalise intensity values
        # img = img / np.max(img)
        p1, p99 = np.percentile(img, [1, 99])
        img = np.clip(img, p1, p99)
        img = (img - p1) / (p99 - p1)

        # convert to pytorch tensor
        img = torch.tensor(img, dtype = torch.float32).unsqueeze(0)

        # resize to (64, 64, 64) for training
        img = F.interpolate(
            img.unsqueeze(0),
            size = (64,64,64),
            mode = 'trilinear',
            align_corners = False
        ).squeeze(0)

        # return (image, dummyLabel) because it is unsupervised learning
        return img, 0
