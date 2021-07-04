import argparse
import os
import numpy as np
import itertools
import sys
from tqdm import tqdm

# from torch.autograd import Variable
# from models import *
# import torch.nn as nn
# import torch
# import pickle

import librosa
from utils import ls, preprocess_wav, melspectrogram, to_numpy

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="saved version based on epoch to test from")
parser.add_argument("--model_name", type=str, help="name of the model")
parser.add_argument("--gen_id", type=str, help="id of the generator for the target domain")
parser.add_argument("--wav", type=str, help="path to wav file for input to transfer")
parser.add_argument("--outdir", type=str, default='out_infer', help="path to output directory for the conversion")
parser.add_argument("--n_overlap", type=int, default=4, help="number of overlaps per slice")
parser.add_argument("--img_height", type=int, default=128, help="size of image height")
parser.add_argument("--img_width", type=int, default=128, help="size of image width")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--n_downsample", type=int, default=2, help="number downsampling layers in encoder")
parser.add_argument("--dim", type=int, default=32, help="number of filters in first encoder layer")

opt = parser.parse_args()
print(opt)

os.makedirs(opt.outdir, exist_ok=True)

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
input_shape = (opt.channels, opt.img_height, opt.img_width)

# Dimensionality (channel-wise) of image embedding
shared_dim = opt.dim * 2 ** opt.n_downsample

# Initialize generator and discriminator
encoder = Encoder(dim=opt.dim, in_channels=opt.channels, n_downsample=opt.n_downsample)
G = Generator(dim=opt.dim, out_channels=opt.channels, n_upsample=opt.n_downsample, shared_block=ResidualBlock(features=shared_dim))

if cuda:
    encoder = encoder.cuda()
    G = G1.cuda()

assert os.path.exists("saved_models/%s/encoder_%02d.pth" % (opt.model_name, opt.epoch)), 'Check that trained encoder exists'
assert os.path.exists("saved_models/%s/G%s_%02d.pth" % (opt.model_name, opt.gen_id, opt.epoch)), 'Check that trained generator exists'
    
# Load pretrained models
encoder.load_state_dict(torch.load("saved_models/%s/encoder_%02d.pth" % (opt.model_name, opt.epoch)))
G.load_state_dict(torch.load("saved_models/%s/G%s_%02d.pth" % (opt.model_name, opt.gen_id, opt.epoch)))

# Set to eval mode 
encoder.eval()
G.eval()

# Load audio and preprocess
sample = preprocess_wav(opt.wav)
spect_src = melspectrogram(sample)

# --------------------
#  Local Inference
# --------------------

def infer(S):
    """Takes in a standard sized spectrogram, returns converted version"""
    X = Variable(S.type(Tensor))
    mu, Z = encoder(X)  # Get shared latent representation
    fake_X = G(Z)  # Translate speech
    retrun to_numpy(fake_X)

# ------------------------------------------
#  Global Inference (w/ a sliding window)
# ------------------------------------------

spect_src = np.pad(spect_src, ((0,0),(opt.img_width,opt.img_width)), 'constant')  # padding for consistent overlap
length = spect_src.shape[1]
spect_trg = np.zeros(spect_src.shape)
hop = opt.img_width // opt.n_overlap

for i in range(0, length, hop):
    x = i + opt.img_width
         
    # Get cropped spectro of right dims
    if x <= length:
        S = spect_src[:, i:x]
    else:  # pad sub on right if includes sub segments out of range
        S = spect_src[:, i:]
        S = np.pad(S, ((0,0),(x-length,0)), 'constant') 
        
    # Perform inference from trained model
    T = infer(S)
    
    # Add parts of target spectrogram with an average across overlapping segments    
    for j in range(0, opt.img_width, hop):
        y = j + hop
        if i+y > length: break  # neglect sub segments out of range
        t = T[:, j:y]
        spect_trg[:, i+j:i+y] += t/opt.n_overlap  # add average element

# Removing initial left pad
spect_trg = spect_trg[:, opt.img_width:-opt.img_width]


# progress = tqdm(enumerate(dataloader),desc='',total=len(dataloader))
# for i, batch in progress:


        
# # Save converted output in pickle format
# pickle.dump(feats,open('out_eval/%s/out_%s.pickle'%(opt.model_name, opt.epoch),'wb'))