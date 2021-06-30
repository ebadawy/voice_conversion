import argparse
import pickle
from tqdm import tqdm
from params import num_samples
from utils import ls, preprocess_wav, melspectrogram
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, help="name of the model")
parser.add_argument("--dataset", type=str, help="path to dataset")
parser.add_argument("--n_spkrs", type=int, default=2, help="number of speakers for conversion")

feats = defaultdict(list)
opt = parser.parse_args()
print(opt)

for spkr in range(opt.n_spkrs):
    wavs = ls('%s/spkr_%s | grep .wav'%(opt.dataset, spkr+1))
    for i, wav in tqdm(enumerate(wavs), total=len(wavs), desc="spkr_%d"%(spkr+1)):
        sample = preprocess_wav('%s/spkr_%s/%s'%(opt.dataset, spkr+1, wav))
        spect = melspectrogram(sample)
        if spect.shape[1] >= num_samples:
            feats[spkr].append(spect)

pickle.dump(feats,open('%s/data.pickle'%(opt.dataset),'wb'))
