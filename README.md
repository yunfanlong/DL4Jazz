# DL4Jazz
LSTM, Bi-LSTM, and VAE-LSTM for generating Jazz music.

This project is dedicated to comparing various generative music models. I employed the DeepJazz framework, including its preprocessing steps, grammar, and generation pipeline, along with specific input/output dimensions and selected hyperparameters. This approach facilitates a clear and direct comparison among diverse models.

I extend my gratitude to the DeepJazz and the jazzml teams for their foundational work, which has enabled me to concentrate on comparing and selecting high-level models.

## Usage

Use `generator.py` for public interface

E.g. use `lstm` model and train 8 epochs

```bash
python generator.py --model-choice "lstm" --epochs 8
```

E.g. use `vae-lstm` model and train 1 epochs

```bash
python generator.py --model-choice "vae-lstm" --epochs 1
```

E.g. use `bi-lstm` model and train 256 epochs

```bash
python generator.py --model-choice "bi-lstm" --epochs 256
```

E.g. use `bi-lstm` model and train 5 epochs, with diversity 0.8

```bash
python generator.py --model-choice "bi-lstm" --epochs 5 --diversity 0.8
```

## Model Architechture

### Bi-LSTM
![VAE-LSTM Architecture](image\Bidirectional-LSTM.jpg)
- **Input Layer**
- **Bi-LSTM Layer 1**
  - LSTM Forward 1
  - LSTM Backward 1
  - Dropout 1
- **Bi-LSTM Layer 2**
  - LSTM Forward 2
  - LSTM Backward 2
  - Dropout 2
- **...**
- **Bi-LSTM Layer 8**
  - LSTM Forward 8
  - LSTM Backward 8
  - Dropout 8
- **Output Layer**
  - Dense
  - Softmax

### VAE-LSTM
![VAE-LSTM Architecture](image\vae-lstm-arch.png)
- **Input Layer**
- **LSTM Layer 1 (Split)**
  - LSTM Forward 1
  - Dropout 1
- **Dense Layer 1 - Dense Layer 2**
- **Lambda Layer 1 - RepeatVector**
- **LSTM Layer 2**
  - LSTM Forward 1
  - Dropout 1
- **LSTM Layer 2**
  - LSTM Forward 2
  - Dropout 2
- **Output Layer**
  - Softmax

## Reference

### Code reference and baseline template

+ Deep Jazz  
https://github.com/jisungk/deepjazz

### Paper reference in implementation

+ Chen, K., Zhang, W., Dubnov, S., Xia, G., & Li, W. (2019, January). The effect of explicit structure encoding of deep neural networks for symbolic music generation. In 2019 International Workshop on Multilayer Music Representation and Processing (MMRP) (pp. 77-84). IEEE.  
https://arxiv.org/pdf/1811.08380.pdf

+ Performance-RNN by Margenta  
https://magenta.tensorflow.org/performance-rnn
  + Dynamics
  + Temperature and randomness
  + Volumes

+ Building Autoencoders in Keras  
https://blog.keras.io/building-autoencoders-in-keras.html

+ Generating sentences from a continuous space  
https://arxiv.org/abs/1511.06349

## Additional Dataset

To download the larger Yamaha e-Piano Competition Dataset, from:
+ Malik, I., & Ek, C. H. (2017). Neural translation of musical style. arXiv preprint arXiv:1708.03535.
Use this link: http://imanmalik.com/assets/dataset/TPD.zip

```

## Requirement

```bash
python 2.7
keras
tensorflow
music21
numpy
```