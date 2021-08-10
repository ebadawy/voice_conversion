
##  Voice Conversion Using Speech-to-Speech Neuro-Style Transfer

This repo contains the official implementation of the VAE-GAN from the INTERSPEECH 2020 paper [Voice Conversion Using Speech-to-Speech Neuro-Style Transfer](http://www.interspeech2020.org/uploadfile/pdf/Thu-3-4-11.pdf).


[![](https://ebadawy.github.io/post/speech_style_transfer/synthesis_pipeline.png)](https://youtu.be/zbVQwqx-kYk)


Examples of generated audio using the Flickr8k Audio Corpus: https://ebadawy.github.io/post/speech_style_transfer. Note that these examples are a result of feeding audio reconstructions of this VAE-GAN to [an implementation of WaveNet](https://github.com/r9y9/wavenet_vocoder).

## 1. Data Preperation

Dataset file structure:

```bash
/path/to/database
├── spkr_1
│   ├── sample.wav
├── spkr_2
│   ├── sample.wav
│   ...
└── spkr_N
    ├── sample.wav
    ...
# The directory under each speaker cannot be nested.
```

[Here](https://github.com/RussellSB/tt-vae-gan/blob/e530888af4841cba78a77cda08f8b9dd33dfbd0b/data_prep/flickr.py) is an example script for setting up data preparation from the Flickr8k Audio Corpus. The speakers of interest are the same as in the paper, but may be modified to other speakers if desirable.

## 2. Data Preprocessing

The prepared dataset is organised into a train/eval/test split, the audio is preprocessed and melspectrograms are computed. 

```bash
python preprocess.py --dataset [path/to/dataset] --test-size [float] --eval-size [float]
```

## 3. Training

The VAE-GAN model uses the melspectrograms to learn style transfer between two speakers.

```bash
python train.py --model_name [name of the model] --dataset [path/to/dataset]
```

#### 3.1. Visualization
By default, the code plots a batch of input and output melspectrograms every epoch.  You may add `--plot-interval -1` to the above command to disable it. Alternatively you may add `--plot-interval 20` to plot every 20 epochs.

#### 3.2. Saving Models
By default, models are saved every epoch. With smaller datasets than Flickr8k it may be more appropriate to save less frequently by adding `--checkpoint_interval 20` for 20 epochs.

#### 3.3. Epochs
The max number of epochs may be set with `--n_epochs`. For smaller datasets, you may want to increase this to more than the default 100. To load a pretrained model you can use `--epoch` and set it to the epoch number of the saved model.

#### 3.4. Pretrained Model

You can access pretrained model files [here](https://drive.google.com/drive/folders/1Wui2Pt4sOBl71exRh49GX_JEBpFv_vNg?usp=sharing). By downloading and storing them in a directory `src/saved_models/pretrained`, you may call it for training or inference with:

`--model_name pretrained --epoch 99`

Note that for inference the discriminator files D1 and D2 are not required (meanwhile for training further they are). Also here, G1 refers to the decoding generator for speaker 1 (female) and G2 for speaker 2 (male).

## 4. Inference

The trained VAE-GAN is used for inference on a specified audio file. It works by; sliding a window over a full melspectrogram, locally inferring melspectrogram subsamples, and averaging the overlap. The script then uses Griffin-Lim to reconstruct audio from the generated melspectrogram. 

```bash
python inference.py --model_name [name of the model] --epoch [epoch number] --trg_id [id of target generator] --wav [path/to/source_audio.wav]
```

For achieving high quality results like the paper you can feed the reconstructed audio to trained vocoders such as WaveNet. An example pipeline of using this model with wavenet can be found [here](https://github.com/RussellSB/tt-vae-gan). 

#### 4.1. Directory Input

Instead of a single .wav as input you may specify a whole directory of .wav files by using `--wavdir` instead of `--wav`. 

#### 4.2. Visualization

By default, plotting input and output melspectrograms is enabled. This is useful for a visual comparison between trained models. To disable set `--plot -1` 

#### 4.3. Reconstructive Evaluation

Alongside the process of generating, components for reconstruction and cyclic reconstruction may be enabled by specifying the generator id of the source audio `--src_id [id of source generator]`. 

When set, SSIM metrics for reconstructed melspectrograms and cyclically reconstructed melspectrograms are computed and printed at the end of inference. 

This is an extra feature to help with comparing the reconstructive capabilities of different models. The higher the SSIM, the higher quality the reconstruction.


## References

- [Real-Time-Voice-Cloning](https://github.com/CorentinJ/Real-Time-Voice-Cloning)
- [UNIT](https://github.com/mingyuliutw/UNIT)
- [wavenet_vocoder](https://github.com/r9y9/wavenet_vocoder)
- [Flickr8k Audio Corpus](https://groups.csail.mit.edu/sls/downloads/flickraudio/)


## Citation

If you find this code useful please cite us in your work:
```
@inproceedings{AlBadawy2020,
  author={Ehab A. AlBadawy and Siwei Lyu},
  title={{Voice Conversion Using Speech-to-Speech Neuro-Style Transfer}},
  year=2020,
  booktitle={Proc. Interspeech 2020},
  pages={4726--4730},
  doi={10.21437/Interspeech.2020-3056},
  url={http://dx.doi.org/10.21437/Interspeech.2020-3056}
}
```

## TODO:

- Rewrite `preprocess.py` to handle:
  - multi-process feature extraction
  - display error messages for failed cases
- Create:
  - Notebook for data visualisation
- Want to add something else? Please feel free to submit a PR with your changes or open an issue for that.
