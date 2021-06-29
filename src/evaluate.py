import argparse
import os
import numpy as np
import itertools
import sys
from tqdm import tqdm

from torch.utils.data import DataLoader
from torch.autograd import Variable

from models import *
from data_proc import DataProc

import torch.nn as nn
import torch.nn.functional as F
import torch

from utils import plot_batch_eval


parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="saved version based on epoch to test from")
parser.add_argument("--model_name", type=str, help="name of the model")
parser.add_argument("--dataset", type=str, help="path to dataset for testing")
parser.add_argument("--n_spkrs", type=int, default=2, help="number of speakers for conversion")
parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=128, help="size of image height")
parser.add_argument("--img_width", type=int, default=128, help="size of image width")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--plot_interval", type=int, default=500, help="batch interval between saving generator sample visualisations")
parser.add_argument("--n_downsample", type=int, default=2, help="number downsampling layers in encoder")
parser.add_argument("--dim", type=int, default=32, help="number of filters in first encoder layer")

opt = parser.parse_args()
print(opt)

cuda = True if torch.cuda.is_available() else False

# Create inference output directories for both transfer directions
os.makedirs("out_eval/%s" % opt.model_name, exist_ok=True)
os.makedirs("out_eval/%s/plot_A2B/" % opt.model_name, exist_ok=True)
os.makedirs("out_eval/%s/plot_B2A/" % opt.model_name, exist_ok=True)

input_shape = (opt.channels, opt.img_height, opt.img_width)

# Dimensionality (channel-wise) of image embedding
shared_dim = opt.dim * 2 ** opt.n_downsample

# Initialize generator and discriminator
encoder = Encoder(dim=opt.dim, in_channels=opt.channels, n_downsample=opt.n_downsample)
shared_G = ResidualBlock(features=shared_dim)
G1 = Generator(dim=opt.dim, out_channels=opt.channels, n_upsample=opt.n_downsample, shared_block=shared_G)
G2 = Generator(dim=opt.dim, out_channels=opt.channels, n_upsample=opt.n_downsample, shared_block=shared_G)

if cuda:
    encoder = encoder.cuda()
    G1 = G1.cuda()
    G2 = G2.cuda()

assert os.path.exists("saved_models/%s/encoder_%d.pth" % (opt.model_name, opt.epoch))  # check that trained encoder exists
assert os.path.exists("saved_models/%s/G1_%d.pth" % (opt.model_name, opt.epoch))  # check that trained G1 exists
assert os.path.exists("saved_models/%s/G2_%d.pth" % (opt.model_name, opt.epoch))  # check that trained G2 exists
    
# Load pretrained models
encoder.load_state_dict(torch.load("saved_models/%s/encoder_%d.pth" % (opt.model_name, opt.epoch)))
G1.load_state_dict(torch.load("saved_models/%s/G1_%d.pth" % (opt.model_name, opt.epoch)))
G2.load_state_dict(torch.load("saved_models/%s/G2_%d.pth" % (opt.model_name, opt.epoch)))

# Set to eval mode 
encoder.eval()
G1.eval()
G2.eval()

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

# Prepare dataloader
# TODO: Make it so it doesnt do it randomly (for now will just do it randomly)
dataloader = torch.utils.data.DataLoader(
	DataProc(opt),
	batch_size=opt.batch_size,
	shuffle=True,
	num_workers=opt.n_cpu,
    pin_memory=True
)


# ----------
#  Testing
# ----------

progress = tqdm(enumerate(dataloader),desc='',total=len(dataloader))
for i, batch in progress:

    # Set model input
    X1 = Variable(batch["A"].type(Tensor))
    X2 = Variable(batch["B"].type(Tensor))

    # -------------------------------
    #  Infer with Encoder and Generators
    # -------------------------------

    # Get shared latent representation
    mu1, Z1 = encoder(X1)
    mu2, Z2 = encoder(X2)

    # Reconstruct speech
    recon_X1 = G1(Z1)
    recon_X2 = G2(Z2)

    # Translate speech
    fake_X1 = G1(Z2)
    fake_X2 = G2(Z1)

    # Cycle translation
    mu1_, Z1_ = encoder(fake_X1)
    mu2_, Z2_ = encoder(fake_X2)
    cycle_X1 = G1(Z2_)
    cycle_X2 = G2(Z1_)
        
    # Plot batch every couple batch intervals
    if i % opt.plot_interval == 0:
        plot_batch_eval(opt.model_name, 'plot_A2B', i, X1, recon_X1, fake_X2)
        plot_batch_eval(opt.model_name, 'plot_B2A', i, X2, recon_X2, fake_X1)
