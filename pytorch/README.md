# MissingLink.AI PyTorch Model Zoo

In this folder you can find PyTorch examples integrated with the MissingLink.AI SDK.
The examples are taken from the following sources:
* `pytorch-examples` - [pytorch/examples](https://github.com/pytorch/examples)
* `OpenNMT-py` - [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py)
* `kuangliu/pytorch-cifar` - [pytorch-cifar](https://github.com/kuangliu/pytorch-cifar)
* `kuangliu/pytorch-agender` - [pytorch-agender](https://github.com/kuangliu/pytorch-agender)
* `lstm-sentence-classifier` - [lstm_sentence_classifier](https://github.com/yuchenlin/lstm_sentence_classifier)
* `char-rnn-pytorch` - [char-rnn.pytorch](https://github.com/spro/char-rnn.pytorch)

Every example comes with an original `README` file, that contains explenation about the model, installation instructions and usage instructions.

## Usage

* Clone the repository.
* Choose the example you want to run.
* [Install PyTorch](http://pytorch.org/)
* Install the MissingLink.AI SDK:
```
pip install missinglink-sdk
```
**Note** that some example might require additional packages to run properly.
* Log in to the MissingLink.AI website and create a new project.
* In the training file of the example, find the following lines:
```python
OWNER_ID = 'your_owner_id'
PROJECT_TOKEN = 'your_project_token'
```
* Find your owner ID and project token:

![missinglink](https://user-images.githubusercontent.com/30972111/33119952-44d0313c-cf79-11e7-8be3-091eca2e9e57.png)
![missinglink](https://user-images.githubusercontent.com/30972111/33120206-01428874-cf7a-11e7-8441-3e7b1f860845.png)
* Insert your owner ID and the project token in the appropriate places.
* Run the model:
```
# from the 'model-zoo/pytorch/<example-name>' directory
python <train-file>.py
```

## Main Integration Files

A list of all the integrated files.

- pytorch-examples
  - [dcgan]
    - pytorch/pytorch-examples/dcgan/main.py
  - [fast_neural_style]
    - pytorch/pytorch-examples/fast_neural_style/neural_style/neural_style.py
  - [imagenet]
    - pytorch/pytorch-examples/imagenet/main.py
  - [mnist]
    - pytorch/pytorch-examples/mnist/main.py
  - [regression]
    - pytorch/pytorch-examples/regression/main.py
  - [reinforcement_learning]
    - pytorch/pytorch-examples/reinforcement_learning/actor_critic.py
    - pytorch/pytorch-examples/reinforcement_learning/reinforce.py
  - [snli]
    - pytorch/pytorch-examples/snli/train.py
  - [super_resolution]
    - pytorch/pytorch-examples/super_resolution/main.py
  - [vae]
    - pytorch/pytorch-examples/vae/main.py
  - [word_language_model]
    - pytorch/pytorch-examples/word_language_model/main.py
- [OpenNMT-py]
  - pytorch/OpenNMT-py/train.py
- kuangliu
  - [pytorch-cifar]
    - pytorch/kuangliu/pytorch-agender/train.py
  - [pytorch-agender]
    - pytorch/kuangliu/pytorch-cifar/main.py
- [lstm_sentence_classifier]
  - pytorch/lstm_sentence_classifier/LSTM_sentence_classifier.py
  - pytorch/lstm_sentence_classifier/LSTM_sentence_classifier_minibatch.py
- [char-rnn-pytorch]
  - pytorch/char-rnn-pytorch/train.py

