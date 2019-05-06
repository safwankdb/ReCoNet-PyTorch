import os
import argparse
import torch
from torchvision import transforms, utils
from skimage import io, transform
from PIL import Image
import cv2
import numpy as np
from network import ReCoNet
from utilities import *

parser = argparse.ArgumentParser()
parser.add_argument('--source', required=True, help='Video file to process')
# parser.add_argument('--target', required=True, help='Output file')
parser.add_argument('--model', required=True, help='Model state_dict file')
args = parser.parse_args()
device = 'cuda'
video_capture = cv2.VideoCapture(args.source) 
model = ReCoNet().to(device)
model.load_state_dict(torch.load(args.model))

images = os.listdir('alley_2')
images.sort()
# for i in images:
#     frame = cv2.imread('alley_2/'+i)
while(True):
    ret, frame = video_capture.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = torch.from_numpy(transform.resize(frame, (360, 640))).to(device).permute(2, 0, 1).float()
    # frame = normalize(frame)
    features, styled_frame = model(frame.unsqueeze(0))
#     styled_frame -= 127.5
#     styled_frame = styled_frame.cpu().clamp(0, 255).data.squeeze(0).numpy().transpose(1, 2, 0).astype('uint8')
    styled_frame = transforms.ToPILImage()(styled_frame[0].detach().cpu())
    styled_frame = np.array(styled_frame)
    styled_frame = styled_frame[:, :,::-1]
    cv2.imshow('frame', styled_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
