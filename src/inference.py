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

# to delete
import matplotlib.pyplot as plt
def plot(f, mel):
    plt.figure()
    plt.imshow(np.rot90(mel, 2), interpolation="None")
    plt.ylabel('Mels')
    plt.ylabel('Frames')
    plt.title('Melspectrogram')
    plt.tight_layout()
    plt.savefig(f)
    plt.close()

# os.makedirs(opt.outdir, exist_ok=True)

# cuda = True if torch.cuda.is_available() else False
# Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
# input_shape = (opt.channels, opt.img_height, opt.img_width)

# # Dimensionality (channel-wise) of image embedding
# shared_dim = opt.dim * 2 ** opt.n_downsample

# # Initialize generator and discriminator
# encoder = Encoder(dim=opt.dim, in_channels=opt.channels, n_downsample=opt.n_downsample)
# shared_G = ResidualBlock(features=shared_dim)
# G1 = Generator(dim=opt.dim, out_channels=opt.channels, n_upsample=opt.n_downsample, shared_block=shared_G)
# G2 = Generator(dim=opt.dim, out_channels=opt.channels, n_upsample=opt.n_downsample, shared_block=shared_G)

# if cuda:
#     encoder = encoder.cuda()
#     G1 = G1.cuda()
#     G2 = G2.cuda()

# assert os.path.exists("saved_models/%s/encoder_%02d.pth" % (opt.model_name, opt.epoch))  # check that trained encoder exists
# assert os.path.exists("saved_models/%s/G1_%02d.pth" % (opt.model_name, opt.epoch))  # check that trained G1 exists
# assert os.path.exists("saved_models/%s/G2_%02d.pth" % (opt.model_name, opt.epoch))  # check that trained G2 exists
    
# # Load pretrained models
# encoder.load_state_dict(torch.load("saved_models/%s/encoder_%02d.pth" % (opt.model_name, opt.epoch)))
# G1.load_state_dict(torch.load("saved_models/%s/G1_%02d.pth" % (opt.model_name, opt.epoch)))
# G2.load_state_dict(torch.load("saved_models/%s/G2_%02d.pth" % (opt.model_name, opt.epoch)))

# # Set to eval mode 
# encoder.eval()
# G1.eval()
# G2.eval()

# Prepare input wav for inference
sample = preprocess_wav(opt.wav)
spect_src = melspectrogram(sample)
spect_src = np.pad(spect_src, ((0,0),(opt.img_width,opt.img_width)), 'constant')  # pad left to start a bit before

length = spect_src.shape[1]
spect_trg = np.zeros(spect_src.shape)
hop = opt.img_width // opt.n_overlap

# ------------------------------------
#  Infering (w/ a sliding window)
# ------------------------------------

for i in range(0, length, hop):
    x = i + opt.img_width
         
    # Get cropped spectro of right dims
    if x <= length:
        S = spect_src[:, i:x]
    else:  # pad sub on right if includes sub segments out of range
        S = spect_src[:, i:]
        S = np.pad(S, ((0,0),(x-length,0)), 'constant') 
        
    # Perform inference from trained model
    # TODO
    T = S
    
    # Add parts of target spectrogram with an average across overlapping segments    
    for j in range(0, opt.img_width, hop):
        y = j + hop
        if i+y > length: break  # neglect sub segments out of range
        t = T[:, j:y]
        spect_trg[:, i+j:i+y] += t/opt.n_overlap  # add average element

# Removing initial left pad
spect_src = spect_src[:, opt.img_width:-opt.img_width]
spect_trg = spect_trg[:, opt.img_width:-opt.img_width]

print(spect_src.shape)
print(spect_trg.shape)
plot('src_after.png', spect_src)    
plot('trg_after.png', spect_trg) 
    
comparison = spect_src == spect_trg
print(comparison.all())

# progress = tqdm(enumerate(dataloader),desc='',total=len(dataloader))
# for i, batch in progress:

#     # Set model input
#     X1 = Variable(batch["A"].type(Tensor))
#     X2 = Variable(batch["B"].type(Tensor))

#     # -------------------------------
#     #  Infer with Encoder and Generators
#     # -------------------------------

#     # Get shared latent representation
#     mu1, Z1 = encoder(X1)
#     mu2, Z2 = encoder(X2)

#     # Translate speech
#     fake_X1 = G1(Z2)
#     fake_X2 = G2(Z1)
        
#     # Plot batch every couple batch intervals
#     if opt.plot_interval != -1 and i % opt.plot_interval == 0:
#         plot_batch_eval(opt.model_name, 'plot_A2B_%02d'%opt.epoch, i, X1, fake_X2)
#         plot_batch_eval(opt.model_name, 'plot_B2A_%02d'%opt.epoch, i, X2, fake_X1)
        
#     # Vocode batch every couple batch intervals    
#     if opt.wav_interval != -1 and i % opt.wav_interval == 0:
#         wav_batch_eval(opt.model_name, 'wav_A2B_%02d'%opt.epoch, i, X1, fake_X2)
#         wav_batch_eval(opt.model_name, 'wav_B2A_%02d'%opt.epoch, i, X2, fake_X1)
        
#     # Append batch output to features dictionary
#     feats['A2B'].append([spect for spect in to_numpy(fake_X2)])
#     feats['B2A'].append([spect for spect in to_numpy(fake_X1)])
        
# # Save converted output in pickle format
# pickle.dump(feats,open('out_eval/%s/out_%s.pickle'%(opt.model_name, opt.epoch),'wb'))