import argparse
import os
import glob
import numpy as np
import itertools
import sys
from tqdm import tqdm

from torch.autograd import Variable
from models import *
import torch.nn as nn
import torch
import pickle

import librosa
from utils import ls, preprocess_wav, melspectrogram, to_numpy, plot_mel_transfer_infer, reconstruct_waveform
from params import sample_rate
import soundfile as sf

import skimage.metrics
from statistics import mean
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=99, help="saved version based on epoch to test from")
parser.add_argument("--model_name", type=str, help="name of the model")
parser.add_argument("--trg_id", type=str, help="id of the generator for the target domain")
parser.add_argument("--src_id", type=str, default=None, help="id of the generator for the source domain (Specify for a recon/cyclic evaluation with SSIM)")

parser.add_argument("--wav", type=str, default=None, help="path to wav file for input to transfer")
parser.add_argument("--wavdir", type=str, default=None, help="path to directory of wav files for input to transfer")

parser.add_argument("--plot", type=int, default=1, help="plot the spectrograms before and after (disable with -1)")
parser.add_argument("--n_overlap", type=int, default=4, help="number of overlaps per slice")
parser.add_argument("--img_height", type=int, default=128, help="size of image height")
parser.add_argument("--img_width", type=int, default=128, help="size of image width")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--n_downsample", type=int, default=2, help="number downsampling layers in encoder")
parser.add_argument("--dim", type=int, default=32, help="number of filters in first encoder layer")

opt = parser.parse_args()
print(opt) 

assert opt.wav or opt.wavdir , 'Please specify an input wav file or directory'
assert not opt.wav or not opt.wavdir, 'Cannot specify both wav and wavdir, choose one'

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
input_shape = (opt.channels, opt.img_height, opt.img_width)

# Dimensionality (channel-wise) of image embedding
shared_dim = opt.dim * 2 ** opt.n_downsample

assert os.path.exists("saved_models/%s/encoder_%02d.pth" % (opt.model_name, opt.epoch)), 'Check that trained encoder exists'
assert os.path.exists("saved_models/%s/G%s_%02d.pth" % (opt.model_name, opt.trg_id, opt.epoch)), 'Check that trained generator exists'
if opt.src_id: assert os.path.exists("saved_models/%s/G%s_%02d.pth" % (opt.model_name, opt.src_id, opt.epoch)), 'Check that trained generator exists'

# Prepare directories    
root = 'out_infer/%s_%d_G%s'% (opt.model_name, opt.epoch, opt.trg_id)
if opt.src_id: root += '_S%s'%opt.src_id  # includes src_id if specified
os.makedirs(root+'/gen/', exist_ok=True)
os.makedirs(root+'/ref/', exist_ok=True)

# Initialize encoder and decoder
encoder = Encoder(dim=opt.dim, in_channels=opt.channels, n_downsample=opt.n_downsample)
G_trg= Generator(dim=opt.dim, out_channels=opt.channels, n_upsample=opt.n_downsample, shared_block=ResidualBlock(features=shared_dim))
if opt.src_id: G_src = Generator(dim=opt.dim, out_channels=opt.channels, n_upsample=opt.n_downsample, shared_block=ResidualBlock(features=shared_dim))

if cuda:
    encoder = encoder.cuda()
    G_trg= G_trg.cuda()
    if opt.src_id: G_src = G_src.cuda()

# Load pretrained models
encoder.load_state_dict(torch.load("saved_models/%s/encoder_%02d.pth" % (opt.model_name, opt.epoch), map_location=device))
G_trg.load_state_dict(torch.load("saved_models/%s/G%s_%02d.pth" % (opt.model_name, opt.trg_id, opt.epoch), map_location=device))
if opt.src_id: G_src.load_state_dict(torch.load("saved_models/%s/G%s_%02d.pth" % (opt.model_name, opt.src_id, opt.epoch)))

# Set to eval mode 
encoder.eval()
G_trg.eval()
if opt.src_id: G_src.eval()

# Initialising arrays to store SSIM (will average at end)
if opt.src_id: 
    ssim_recon = []
    ssim_cyclic = []

# ----------------------------------------------------
#  SSIM Computation (Evaluating reconstruction)
# ----------------------------------------------------

def ssim(spect_src, spect_recon):
    return skimage.metrics.structural_similarity(spect_src, spect_recon, data_range=1)

# -----------------------------------------
#  Local Inference and SSIM evaluation
# -----------------------------------------

def infer(S):
    """Takes in a standard sized spectrogram, returns timbre converted version"""
    S = torch.from_numpy(S)
    S = S.view(1, 1, opt.img_height, opt.img_width)
    X = Variable(S.type(Tensor))
    
    ret = {} # just stores inference output
    
    mu, Z = encoder(X)
    fake_X = G_trg(Z)
    ret['fake'] = to_numpy(fake_X)
    
    
    if opt.src_id: 
        recon_X = G_src(Z)
        ret['recon'] = to_numpy(recon_X)
        
        mu_, Z_ = encoder(fake_X)
        cyclic_X = G_src(Z_)
        ret['cyclic'] = to_numpy(cyclic_X)
    
    return ret

# ------------------------------------------
#  Global Inference (w/ a sliding window)
# ------------------------------------------

def audio_infer(wav):
    
    # Load audio and preprocess
    sample = preprocess_wav(wav)
    spect_src = melspectrogram(sample)
    
    #print(spect_src.max(), spect_src.min())

    spect_src = np.pad(spect_src, ((0,0),(opt.img_width,opt.img_width)), 'constant')  # padding for consistent overlap
    spect_trg = np.zeros(spect_src.shape)
    spect_recon = np.zeros(spect_src.shape)
    spect_cyclic = np.zeros(spect_src.shape)
    
    length = spect_src.shape[1]
    hop = opt.img_width // opt.n_overlap

    for i in tqdm(range(0, length, hop)):
        x = i + opt.img_width

        # Get cropped spectro of right dims
        if x <= length:
            S = spect_src[:, i:x]
        else:  # pad sub on right if includes sub segments out of range
            S = spect_src[:, i:]
            S = np.pad(S, ((0,0),(x-length,0)), 'constant') 

        ret = infer(S) # perform inference from trained model
        T = ret['fake']
        if opt.src_id:
            R = ret['recon']
            C = ret['cyclic']

        # Add parts of target spectrogram with an average across overlapping segments    
        for j in range(0, opt.img_width, hop):
            y = j + hop
            if i+y > length: break  # neglect sub segments out of range
                
            # select subsegments to consider for overlap
            t = T[:, j:y]
            if opt.src_id:
                r = R[:, j:y]
                c = C[:, j:y]
            
            # add average element
            spect_trg[:, i+j:i+y] += t/opt.n_overlap
            if opt.src_id:
                spect_recon[:, i+j:i+y] += r/opt.n_overlap
                spect_cyclic[:, i+j:i+y] += c/opt.n_overlap


    # remove initial padding
    spect_src = spect_src[:, opt.img_width:-opt.img_width]
    spect_trg = spect_trg[:, opt.img_width:-opt.img_width]
    if opt.src_id:
        spect_recon = spect_recon[:, opt.img_width:-opt.img_width] 
        spect_cyclic = spect_cyclic[:, opt.img_width:-opt.img_width] 
                                                 
    # Compute and append SSIM
    if opt.src_id:
        ssim_recon.append(ssim(spect_src, spect_recon))
        ssim_cyclic.append(ssim(spect_src, spect_cyclic))

    # prepare file name for saving
    f = wav.split('/')[-1]
    wavname = f.split('.')[0]
    fname = 'G%s_%s' % (opt.trg_id, wavname)

    # plot transfer if specified
    if opt.plot != -1:
        os.makedirs(root+'/plots/', exist_ok=True)
        plot_mel_transfer_infer(root+'/plots/%s.png' % fname, spect_src, spect_trg)  

    # reconstruct with Griffin Lim (takes a while, later feed this wav as input to vocoder)
    print('Reconstructing with Griffin Lim...')
    x = reconstruct_waveform(spect_trg)
    
    sf.write(root+'/gen/%s_gen.wav'%fname, x, sample_rate)  # generated output
    sf.write(root+'/ref/%s_ref.wav'%fname, sample, sample_rate)  # input reference (for convenience)
    
    
if opt.wav:
    audio_infer(opt.wav)

if opt.wavdir:
    audio_files = glob.glob(os.path.join(opt.wavdir, '*.wav'))
    for i, wav in enumerate(audio_files):
        print('[File %d/%d]' % (i+1, len(audio_files)))
        audio_infer(wav)
        
# Display average SSIM
if opt.src_id:
    print('Average SSIM for recon: %0.2f'%mean(ssim_recon))
    print('Average SSIM for cyclic: %0.2f'%mean(ssim_cyclic))
