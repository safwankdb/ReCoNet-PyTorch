# Real-time-Coherent-Style-Transfer-For-Videos
This is a PyTorch implementation of the paper ReCoNet.
## Abstract
We aim to build a generalisable neural style transfer network for videos with temporal consistency and efficient real time style transfer using any modern GPU. All the past techniques haven't been able to accomplish real-time efficient style transfer either lacking in temporal consistency, nice perceptual style quality or fast processing. Here we have used ReCoNet, which tried to mitigate all these problems.
## ReCoNet
ReCoNet is a feed forward neural network, which stylises videos frame by frame through an encoder and subsequently a decoder, and a VGG loss network to capture the perceptual style of the transfer target.\s\s
The temporal loss is guided by occlusion masks and optical flow.\s\s
Only the encoder and decoder run during inference which makes ReCoNet very efficient, running above real-time standards on modern GPUs.\s\s
The network is illustrated in the figure below:\s\s
