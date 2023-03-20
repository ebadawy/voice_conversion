# SOURCE:
# - https://github.com/CorentinJ/Real-Time-Voice-Cloning
# - https://github.com/r9y9/wavenet_vocoder

from scipy.ndimage.morphology import binary_dilation
import os
import math
import numpy as np
from pathlib import Path
from typing import Optional, Union
import librosa
import struct
from params import *
from scipy.signal import lfilter
import soundfile as sf
import matplotlib.pyplot as plt

try:
    import webrtcvad
except:
    warn("Unable to import 'webrtcvad'. This package enables noise removal and is recommended.")
    webrtcvad=None

int16_max = (2 ** 15) - 1

def preprocess_wav(fpath_or_wav: Union[str, Path, np.ndarray],
                   source_sr: Optional[int] = None):
    """
    Applies the preprocessing operations used in training the Speaker Encoder to a waveform
    either on disk or in memory. The waveform will be resampled to match the data hyperparameters.

    :param fpath_or_wav: either a filepath to an audio file (many extensions are supported, not
    just .wav), either the waveform as a numpy array of floats.
    :param source_sr: if passing an audio waveform, the sampling rate of the waveform before
    preprocessing. After preprocessing, the waveform's sampling rate will match the data
    hyperparameters. If passing a filepath, the sampling rate will be automatically detected and
    this argument will be ignored.
    """
    # Load the wav from disk if needed
    if isinstance(fpath_or_wav, str) or isinstance(fpath_or_wav, Path):
        wav, source_sr = librosa.load(str(fpath_or_wav), sr=None)
    else:
        wav = fpath_or_wav

    # Resample the wav if needed
    if source_sr is not None and source_sr != sample_rate:
        wav = librosa.resample(wav, orig_sr=source_sr, target_sr=sample_rate)

    # Apply the preprocessing: normalize volume and shorten long silences
    wav = normalize_volume(wav, audio_norm_target_dBFS, increase_only=True)
    if webrtcvad:
        wav = trim_long_silences(wav)

    return wav

def trim_long_silences(wav):
    """
    Ensures that segments without voice in the waveform remain no longer than a
    threshold determined by the VAD parameters in params.py.

    :param wav: the raw waveform as a numpy array of floats
    :return: the same waveform with silences trimmed away (length <= original wav length)
    """
    # Compute the voice detection window size
    samples_per_window = (vad_window_length * sample_rate) // 1000

    # Trim the end of the audio to have a multiple of the window size
    wav = wav[:len(wav) - (len(wav) % samples_per_window)]

    # Convert the float waveform to 16-bit mono PCM
    pcm_wave = struct.pack("%dh" % len(wav), *(np.round(wav * int16_max)).astype(np.int16))

    # Perform voice activation detection
    voice_flags = []
    vad = webrtcvad.Vad(mode=3)
    for window_start in range(0, len(wav), samples_per_window):
        window_end = window_start + samples_per_window
        voice_flags.append(vad.is_speech(pcm_wave[window_start * 2:window_end * 2],
                                         sample_rate=sample_rate))
    voice_flags = np.array(voice_flags)

    # Smooth the voice detection with a moving average
    def moving_average(array, width):
        array_padded = np.concatenate((np.zeros((width - 1) // 2), array, np.zeros(width // 2)))
        ret = np.cumsum(array_padded, dtype=float)
        ret[width:] = ret[width:] - ret[:-width]
        return ret[width - 1:] / width

    audio_mask = moving_average(voice_flags, vad_moving_average_width)
    audio_mask = np.round(audio_mask).astype(np.bool)

    # Dilate the voiced regions
    audio_mask = binary_dilation(audio_mask, np.ones(vad_max_silence_length + 1))
    audio_mask = np.repeat(audio_mask, samples_per_window)

    return wav[audio_mask == True]


def normalize_volume(wav, target_dBFS, increase_only=False, decrease_only=False):
    if increase_only and decrease_only:
        raise ValueError("Both increase only and decrease only are set")
    dBFS_change = target_dBFS - 10 * np.log10(np.mean(wav ** 2))
    if (dBFS_change < 0 and increase_only) or (dBFS_change > 0 and decrease_only):
        return wav
    return wav * (10 ** (dBFS_change / 20))


def ls(path):
    # return os.popen('ls %s'%path).read().split('\n')[:-1]
    if os.name == 'posix':  # for Linux and macOS
        return os.popen('ls %s' %path).read().split('\n')[:-1]
    elif os.name == 'nt':  # for Windows
        return os.popen('dir /b %s' %path).read().split('\n')[:-1]
    else:
        raise OSError('Unsupported operating system: %s' % os.name)

def label_2_float(x, bits):
    return 2 * x / (2**bits - 1.) - 1.


def float_2_label(x, bits):
    assert abs(x).max() <= 1.0
    x = (x + 1.) * (2**bits - 1) / 2
    return x.clip(0, 2**bits - 1)


def load_wav(path):
    return librosa.load(path, sr=sample_rate)[0]


def save_wav(x, path):
    sf.write(path, x.astype(np.float32), sample_rate)


def split_signal(x):
    unsigned = x + 2**15
    coarse = unsigned // 256
    fine = unsigned % 256
    return coarse, fine


def combine_signal(coarse, fine):
    return coarse * 256 + fine - 2**15


def encode_16bits(x):
    return np.clip(x * 2**15, -2**15, 2**15 - 1).astype(np.int16)


def linear_to_mel(spectrogram):
    return librosa.feature.melspectrogram(
        S=spectrogram, sr=sample_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin)

def normalize(S):
    return np.clip((S - min_level_db) / -min_level_db, 0, 1)


def denormalize(S):
    return (np.clip(S, 0, 1) * -min_level_db) + min_level_db


def amp_to_db(x):
    return 20 * np.log10(np.maximum(1e-5, x))


def db_to_amp(x):
    return np.power(10.0, x * 0.05)


def spectrogram(y):
    D = stft(y)
    S = amp_to_db(np.abs(D)) - ref_level_db
    return normalize(S)


def melspectrogram(y):
    D = stft(y)
    S = amp_to_db(linear_to_mel(np.abs(D)))
    return normalize(S)


def stft(y):
    return librosa.stft(
        y=y,
        n_fft=n_fft, hop_length=hop_length, win_length=win_length)


def pre_emphasis(x):
    return lfilter([1, -preemphasis], [1], x)


def de_emphasis(x):
    return lfilter([1], [1, -preemphasis], x)


def encode_mu_law(x, mu):
    mu = mu - 1
    fx = np.sign(x) * np.log(1 + mu * np.abs(x)) / np.log(1 + mu)
    return np.floor((fx + 1) / 2 * mu + 0.5)


def decode_mu_law(y, mu, from_labels=True):
    # TODO: get rid of log2 - makes no sense
    if from_labels: y = label_2_float(y, math.log2(mu))
    mu = mu - 1
    x = np.sign(y) / mu * ((1 + mu) ** np.abs(y) - 1)
    return x

def reconstruct_waveform(mel, n_iter=32):
    """Uses Griffin-Lim phase reconstruction to convert from a normalized
    mel spectrogram back into a waveform."""
    denormalized = denormalize(mel)
    amp_mel = db_to_amp(denormalized)
    S = librosa.feature.inverse.mel_to_stft(
        amp_mel, power=1, sr=sample_rate,
        n_fft=n_fft, fmin=fmin)
    wav = librosa.core.griffinlim(
        S, n_iter=n_iter,
        hop_length=hop_length, win_length=win_length)
    return wav

def to_numpy(batch):
    batch = batch.detach().cpu().numpy()
    batch = np.squeeze(batch)
    return batch

def plot_mel_transfer_train(save_path, curr_epoch, mel_in, mel_cyclic, mel_out, mel_target):
    """Visualises melspectrogram style transfer in training, with target specified"""
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(6, 6))
    
    ax[0,0].imshow(mel_in, interpolation="None")
    ax[0,0].invert_yaxis()
    ax[0,0].set(title='Input')
    ax[0,0].set_ylabel('Mels')
    ax[0,0].axes.xaxis.set_ticks([])
    ax[0,0].axes.xaxis.set_ticks([])
    
    ax[1,0].imshow(mel_cyclic, interpolation="None")
    ax[1,0].invert_yaxis()
    ax[1,0].set(title='Cyclic Reconstruction')
    ax[1,0].set_xlabel('Frames')
    ax[1,0].set_ylabel('Mels')

    ax[0,1].imshow(mel_out, interpolation="None")
    ax[0,1].invert_yaxis()
    ax[0,1].set(title='Output')
    ax[0,1].axes.yaxis.set_ticks([])
    ax[0,1].axes.xaxis.set_ticks([])
    
    ax[1,1].imshow(mel_target, interpolation="None")
    ax[1,1].invert_yaxis()
    ax[1,1].set(title='Target')
    ax[1,1].set_xlabel('Frames')
    ax[1,1].axes.yaxis.set_ticks([])
    
    fig.suptitle('Epoch ' + str(curr_epoch))
    plt.savefig(save_path)
    plt.close()
    
def plot_batch_train(modelname, direction, curr_epoch, SRC, cyclic_SRC, fake_TRGT, real_TRGT):
    SRC, cyclic_SRC, fake_TRGT, real_TRGT = to_numpy(SRC), to_numpy(cyclic_SRC), to_numpy(fake_TRGT), to_numpy(real_TRGT)
    i = 1
    for src, cyclic_src, fake_target, real_target in zip(SRC, cyclic_SRC, fake_TRGT, real_TRGT):
        fname = "out_train/%s/%s/%s_%02d_%s.png"%(modelname, direction, direction, curr_epoch, i)
        plot_mel_transfer_train(fname, curr_epoch, src, cyclic_src, fake_target, real_target)
        i += 1
    
def plot_mel_transfer_eval(save_path, mel_in, mel_out):
    """Visualises melspectrogram style transfer in testing, only shows input and output"""
    fig, ax = plt.subplots(nrows=1, ncols=2, sharex=True, figsize=(5,3))
    
    ax[0].imshow(mel_in, interpolation="None")
    ax[0].invert_yaxis()
    ax[0].set(title='Input')
    ax[0].set_ylabel('Mels')
    ax[0].set_xlabel('Frames')

    ax[1].imshow(mel_out, interpolation="None")
    ax[1].invert_yaxis()
    ax[1].set(title='Output')
    ax[1].set_xlabel('Frames')
    ax[1].axes.yaxis.set_ticks([])

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
   
    
def plot_batch_eval(modelname, direction, batchno, SRC, fake_TRGT):
    SRC, fake_TRGT = to_numpy(SRC), to_numpy(fake_TRGT)
    i = 1
    for src, fake_target in zip(SRC, fake_TRGT):
        fname = "out_eval/%s/%s/%s_%04d_%s.png"%(modelname, direction, direction, batchno, i)
        plot_mel_transfer_eval(fname, src, fake_target)
        i += 1
        
        
def wav_batch_eval(modelname, direction, batchno, SRC, fake_TRGT):
    SRC, fake_TRGT = to_numpy(SRC), to_numpy(fake_TRGT)
    i = 1
    for src, fake_target in zip(SRC, fake_TRGT):
        name = "out_eval/%s/%s/%s_%04d_%s"%(modelname, direction, direction, batchno, i)
        
        ref = reconstruct_waveform(src)
        ref_fname = name + '_ref.wav'
        sf.write(ref_fname, ref, sample_rate)
        
        out = reconstruct_waveform(fake_target)
        out_fname = name + '_out.wav'
        sf.write(out_fname, out, sample_rate)
        i += 1
        

def plot_mel_transfer_infer(save_path, mel_in, mel_out):
    """Visualises melspectrogram style transfer in inference, shows total input and output"""
    fig, ax = plt.subplots(nrows=2, ncols=1, sharey=True)

    ax[0].imshow(mel_in, interpolation="None", aspect='auto')
    ax[0].set(title='Input')
    ax[0].set_ylabel('Mels')
    ax[0].axes.xaxis.set_ticks([])

    ax[1].imshow(mel_out, interpolation="None", aspect='auto')
    ax[1].set(title='Output')
    ax[1].set_ylabel('Mels')
    ax[1].set_xlabel('Frames')
    
    ax[0].invert_yaxis()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    
