"""
Chaanges by Kushagra :- commented 53 line,
Does the style tensor have batch size =2??? Or is that a mistake?
"""

import torch
import torch.functional as F
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import matplotlib.pyplot as plt
import flowlib
from PIL import Image

from utilities import *
from network import *
from totaldata import *

LR = 1e-3
epochs = 10
device = 'cuda'

dataloader = DataLoader(FlyingChairsDataset("../FlyingChairs2/"), batch_size=1)
model = ReCoNet().to(device)

resume = input('Resume training? y/n: ').lower() == 'y'
if resume:
	model_name = input('Model Name: ')
	model.load_state_dict(torch.load('runs/output/' + model_name))

adam = optim.Adam(model.parameters(), lr=LR)
L2distance = nn.MSELoss().to(device)
L2distancematrix = nn.MSELoss(reduction='none').to(device)
Vgg16 = Vgg16().to(device)

style_names = ('autoportrait', 'candy', 'composition',
			   'edtaonisl', 'udnie')
style_model_path = './models/weights/'
#Changed excessive min, max operations in next line
style_img_path = './models/style/'+style_names[2]
style = transform1(Image.open(style_img_path+'.jpg'))
# print(style.size())
style = style.unsqueeze(0).expand(1, 3, IMG_SIZE[0], IMG_SIZE[1]).to(device)

for param in Vgg16.parameters():
	param.requires_grad = False

STYLE_WEIGHTS = [1e-1, 1e0, 1e1, 5e0] #[1e-1, 1e0, 1e1, 5e0, 1e1] not sure about what value to be deleted 
# print(style.size())
styled_featuresR = Vgg16(normalize(style))
# print(styled_featuresR[1].size())
style_GM = [gram_matrix(f) for f in styled_featuresR]
# print(len(style_GM))

for epoch in range(epochs):
	for itr, (img1, img2, mask, flow) in enumerate(dataloader):
		flow=-flow
		adam.zero_grad()
		# print(img1.size())
		if (itr+1) % 500 == 0:
			for param in adam.param_groups:
				param['lr'] = max(param['lr']/1.2, 1e-4)
	

		feature_map1, styled_img1 = model(img1)
		feature_map2, styled_img2 = model(img2)
		styled_img1 = normalize(styled_img1)
		styled_img2 = normalize(styled_img2)
		img1, img2 = normalize(img1), normalize(img2)

		styled_features1 = Vgg16(styled_img1)
		styled_features2 = Vgg16(styled_img2)
		img_features1 = Vgg16(img1)
		img_features2 = Vgg16(img2)

		feature_flow = nn.functional.interpolate(
			flow, size=feature_map1.shape[2:], mode='bilinear')
		feature_flow[0,0, :, :] *= float(feature_map1.shape[2])/flow.shape[2]
		feature_flow[0,1, :, :] *= float(feature_map1.shape[3])/flow.shape[3]
		# print(flow.size(), feature_map1.shape[2:],feature_flow.size())
		feature_mask = nn.functional.interpolate(
			mask.view(1,1,640,360), size=feature_map1.shape[2:], mode='bilinear')
		# print(feature_map1.size(), feature_flow.size())
		warped_fmap = warp(feature_map1, feature_flow)

		# #Changed by KJ to multiply with feature mask
		# # print(L2distancematrix(feature_map2, warped_fmap).size()) #Should be a matrix not number
		# # mean replaced sum 
		f_temporal_loss = torch.sum(feature_mask*(L2distancematrix(feature_map2, warped_fmap)))
		f_temporal_loss *= LAMBDA_F
		f_temporal_loss *= 1/(feature_map2.shape[1]*feature_map2.shape[2]*feature_map2.shape[3])

		# # print(styled_img1.size(), flow.size())
		# # Removed unsqueeze methods in both styled_img1,flow in next line since already 4 dimensional
		warped_style = warp(styled_img1, flow)
		warped_image = warp(img1, flow)

		# print(img2.size())
		output_term = styled_img2[0] - warped_style[0]
		# print(output_term.shape, styled_img2.shape, warped_style.shape)
		input_term = img2[0] - warped_image[0]
		# print(input_term.size())
		# Changed the next few lines since dimension is 4 instead of 3 with batch size=1
		input_term = 0.2126 * input_term[0, :, :] + 0.7152 * \
			input_term[1, :, :] + 0.0722 * input_term[2, :, :]
		input_term = input_term.expand(output_term.size())

		o_temporal_loss = torch.sum(mask * (L2distancematrix(output_term, input_term)))
		o_temporal_loss *= LAMBDA_O
		o_temporal_loss *= 1/(img1.shape[2]*img1.shape[3])

		content_loss = 0
		content_loss += L2distance(styled_features1[2],
								   img_features1[2].expand(styled_features1[2].size()))
		content_loss += L2distance(styled_features2[2],
								   img_features2[2].expand(styled_features2[2].size()))
		content_loss *= ALPHA/(styled_features1[2].shape[1] * styled_features1[2].shape[2] * styled_features1[2].shape[3])

		style_loss = 0
		for i, weight in enumerate(STYLE_WEIGHTS):
			gram_s = style_GM[i]
			# print(styled_features1[i].size())
			gram_img1 = gram_matrix(styled_features1[i])
			gram_img2 = gram_matrix(styled_features2[i])
			# print(gram_img1.size(), gram_s.size())
			style_loss += float(weight) * (L2distance(gram_img1, gram_s.expand(
				gram_img1.size())) + L2distance(gram_img2, gram_s.expand(gram_img2.size())))
		style_loss *= BETA

		reg_loss = GAMMA * \
			(torch.sum(torch.abs(styled_img1[:, :, :, :-1] - styled_img1[:, :, :, 1:])) +
			 torch.sum(torch.abs(styled_img1[:, :, :-1, :] - styled_img1[:, :, 1:, :])))

		reg_loss += GAMMA * \
			 (torch.sum(torch.abs(styled_img2[:, :, :, :-1] - styled_img2[:, :, :, 1:])) +
			 torch.sum(torch.abs(styled_img2[:, :, :-1, :] - styled_img2[:, :, 1:, :])))

		# print(f_temporal_loss.size(), o_temporal_loss.size(), content_loss.size(), style_loss.size(), reg_loss.size())
		loss = f_temporal_loss + o_temporal_loss + content_loss + style_loss + reg_loss
		# loss = content_loss + style_loss
		loss.backward()
		adam.step()
		#
		#
		if (itr+1)%1000 ==0 :
			torch.save(model.state_dict(), '%s/final_reconet_epoch_%d_itr_%d.pth' % ("runs/output", epoch, itr//1000))


		print('[%d/%d][%d/%d] SL: %.4f CL: %.4f FTL: %.4f OTL: %.4f RL: %.4f'
						% (epoch, epochs, itr, len(dataloader),
							style_loss, content_loss , f_temporal_loss, o_temporal_loss, reg_loss))
	torch.save(model.state_dict(), '%s/reconet_epoch_%d.pth' % ("runs/output", epoch))
