# Sentiment Analysis PyTorch
I would suggest to Run in Colab and you don't need to downlaod dataset seperatly 

This repo contains tutorials covering how to do sentiment analysis for moview review using [PyTorch](https://github.com/pytorch/pytorch) 1.8 and [torchtext](https://github.com/pytorch/text) 0.9 using Python 3.8.


**If you find any mistakes or disagree with any of the explanations, please do not hesitate to contact me.**

## Getting Started

To install PyTorch, see installation instructions on the [PyTorch website](https://pytorch.org/get-started/locally).

To install torchtext:

``` bash
pip install torchtext
```

We'll also make use of spaCy to tokenize our data. To install spaCy, follow the instructions [here](https://spacy.io/usage/) making sure to install the English models with:

``` bash
python -m spacy download en_core_web_sm
```

install transformers with below pip command

```bash
pip install transformers
```

## Tutorials

* 1 - [Sentiment Analysis](https://github.com/razacode/Sentiment-Analysis-Movie-Review/blob/main/SA_2%20with%2010%20epocs.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bentrevett/pytorch-sentiment-analysis/blob/master/2%20-%20Upgraded%20Sentiment%20Analysis.ipynb)

    Now we have the basic workflow covered, this tutorial will focus on improving our results. We'll cover: using packed padded sequences, loading and using pre-trained word embeddings, different optimizers, different RNN architectures, bi-directional RNNs, multi-layer (aka deep) RNNs and regularization.

* 2 - [Convolutional Sentiment Analysis](https://github.com/razacode/Sentiment-Analysis-Movie-Review/blob/main/SA_4_with%201d%20cnn.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bentrevett/pytorch-sentiment-analysis/blob/master/4%20-%20Convolutional%20Sentiment%20Analysis.ipynb)

    Next, we'll cover convolutional neural networks (CNNs) for sentiment analysis. This model will be an implementation of [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882).

## References

Here are some things I looked at while making these tutorials. Some of it may be out of date.

* http://anie.me/On-Torchtext/
* http://mlexplained.com/2018/02/08/a-comprehensive-tutorial-to-torchtext/
* https://github.com/spro/practical-pytorch
* https://github.com/bentrevett/pytorch-sentiment-analysis
* https://github.com/keras-team/keras/blob/master/examples/imdb_fasttext.py
* https://github.com/Shawn1993/cnn-text-classification-pytorch

