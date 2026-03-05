from dataset import BrainDataset

ds = BrainDataset("data")
print(len(ds))

img, label = ds[0]
print(type(img))
print(img.shape)
print(label)
