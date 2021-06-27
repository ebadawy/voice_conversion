## Voice Conversion Using Speech-to-Speech Neuro-Style Transfer

This repo contains the official implementation for the INTERSPEECH 2020 paper [Voice Conversion Using Speech-to-Speech Neuro-Style Transfer](http://www.interspeech2020.org/uploadfile/pdf/Thu-3-4-11.pdf).


[![](https://ebadawy.github.io/post/speech_style_transfer/synthesis_pipeline.png)](https://youtu.be/zbVQwqx-kYk)

## Feature Extraction

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

example

```bash
python preprocess.py --model_name [name of the model] --dataset [path/to/dataset]
```

## Training

```bash
python train.py --model_name [name of the model] --dataset [path/to/dataset]
```

### Generated audios

Examples of generated audios using flicker8k audio dataset https://ebadawy.github.io/post/speech_style_transfer.

## TODO:

- Rewrite `preprocess.py` to handle:
  - multi-process feature extraction
  - create train/test/val split
  - display error messages for faild cases
- Create:
  - `inference.py`
  - `requirements.txt`
  - Notebook for data visualisation
- Upload pre-trained models
- Want to add something else? Please feel free to submit a PR with your changes or open an issue for that.


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
