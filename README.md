# Long-term dependencies in RNNs
The aim of this project is to compare how different recurrent, neural network architectures cope with long-term dependencies.

## Experiment
We decided to recreate one of the experiments poposed in the original LSTM paper[[1]](#1):
```Task 6a: two relevant, widely separated symbols. The goal is to classify sequences.
Elements and targets are represented locally (input vectors with only one non-zero bit). The
sequence starts with an E, ends with a B (the "trigger symbol") and otherwise consists of randomly
chosen symbols from the set {a; b; c; d} except for two elements at positions t1 and t2 that are either
X or Y . The sequence length is randomly chosen between 100 and 110, t1 is randomly chosen
between 10 and 20, and t2 is randomly chosen between 50 and 60. There are 4 sequence classes
Q; R; S; U which depend on the temporal order of X and Y . The rules are: X; X -> Q; X; Y ->
R; Y ; X -> S; Y ; Y -> U.
```
The task is fairly simple classification - two symbols determine one of four classes. 
There are two main difficulties:
* input sequence is very noisy - class is based on only 2 out of about a 100 symbols
* important symbols are widely separated (at least 30 symbol gap)

## Implementation
### Data
Training dataset has been replaced by random sequence generator based on [tf.keras.utils.Sequence](https://www.tensorflow.org/api_docs/python/tf/keras/utils/Sequence).
### Sequence coding
Sequences have been padded with 'E' symbol to maximum available length. Both sequence symbols and classes have been one hot encoded.
### Architectures
Three different recurrent architectures have been tested:
* [SimpleRNN](https://www.tensorflow.org/api_docs/python/tf/keras/layers/SimpleRNN)
* [LSTM](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM)
* [GRU](https://www.tensorflow.org/api_docs/python/tf/keras/layers/GRU)
### Training paramteres
* **Loss function**: binary crossentropy
* **Optimizer**: Adam
* **Metrics**: categorical accuracy
* **Batch size**: 64

## Bibliography
1. <a id="1"></a>Hochreiter, S. & Schmidhuber, Jü. (1997). Long short-term memory. Neural computation, 9, 1735--1780
2. <a id="2"></a>[https://colah.github.io/posts/2015-08-Understanding-LSTMs/](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
3. <a id="3"></a>[https://www.tensorflow.org/guide/keras/rnn](https://www.tensorflow.org/guide/keras/rnn)
4. <a id="4"></a>[https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly](https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly)
5. <a id="5"></a>[https://towardsdatascience.com/illustrated-guide-to-lstms-and-gru-s-a-step-by-step-explanation-44e9eb85bf21](https://towardsdatascience.com/illustrated-guide-to-lstms-and-gru-s-a-step-by-step-explanation-44e9eb85bf21)
