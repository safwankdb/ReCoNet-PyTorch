import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np 
from totaldata import *
from skimage import io, transform

data=ConsolidatedDataset(MPI_path="../MPI_Data/" ,FC_path="../FlyingChairs2/")
print(len(data))
img1, img2, mask, flow= data[230]
print(img1.size(), img2.size(), mask.size(), flow.size())
io.imsave("img1.png", img1.squeeze().permute(1,2,0).cpu().numpy())
io.imsave("img2.png", img2.squeeze().permute(1,2,0).cpu().numpy())
# io.imsave("mask.png", mask.squeeze().permute(1,2,0).cpu().numpy())
