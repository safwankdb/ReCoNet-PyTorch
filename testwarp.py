from utilities import *
from dataset import MPIDataset
from skimage import io, transform
from torch.utils.data import DataLoader
import torch

path="../MPI_Data/"
dataloader = DataLoader(MPIDataset("../MPI_Data/"), batch_size=1)

for itr, (img1, img2, mask, flow) in enumerate(dataloader):
	if(itr==1):
		break
	warped=warp(img1,flow)
	print(img1.squeeze().permute(1,2,0).size())
	io.imsave("warped.png", warped.squeeze().permute(1,2,0).cpu().numpy())
	io.imsave("img1.png", img1.squeeze().permute(1,2,0).cpu().numpy())
	io.imsave("img2.png", img2.squeeze().permute(1,2,0).cpu().numpy())
	