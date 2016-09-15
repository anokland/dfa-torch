Training neural networks with back-prop, feedback-alignment and direct feedback-alignment
==========================================================================================

This repo contains code to reproduce experiments in paper
"Direct Feedback Alignment Provides Learning in Deep Neural Networks"
(https://arxiv.org/abs/1609.01596)

This code and readme is copied and modified based on https://github.com/eladhoffer/ConvNet-torch

Deep Networks on classification tasks using Torch
=================================================
This is a complete training example for {Cifar10/100, STL10, SVHN, MNIST} tasks

##Data
You can get the needed data using @soumith's repo: https://github.com/soumith/cifar.torch.git

##Dependencies
* Torch (http://torch.ch)
* "DataProvider.torch" (https://github.com/eladhoffer/DataProvider.torch) for DataProvider class.
* "cudnn.torch" (https://github.com/soumith/cudnn.torch) for faster training. Can be avoided by changing "cudnn" to "nn" in models.

To install all dependencies (assuming torch is installed) use:
```bash
luarocks install https://raw.githubusercontent.com/eladhoffer/eladtools/master/eladtools-scm-1.rockspec
luarocks install https://raw.githubusercontent.com/eladhoffer/DataProvider.torch/master/dataprovider-scm-1.rockspec
```

##Training
You can reproduce the best results for direct feedback-alignment for each dataset with:
```lua
th Main.lua -dataset MNIST -network mlp.lua -LR 2e-4 -eps 0.08
```
or,
```lua
th Main.lua -dataset Cifar10 -network conv.lua -LR 2.5e-5 -whiten
```
or,
```lua
th Main.lua -dataset Cifar100 -network conv.lua -LR 2.5e-5 -whiten
```

##Additional flags
|Flag             | Default Value        |Description
|:----------------|:--------------------:|:----------------------------------------------
|modelsFolder     |  ./Models/           | Models Folder
|network          |  mlp.lua             | Model file - must return valid network.
|criterion        |  bce                 | criterion, ce(cross-entropy) or bce(binary cross-entropy)
|eps              |  0                   | adversarial regularization magnitude (fast-sign-method a.la Goodfellow)
|dropout          |  0                   | 1=apply dropout regularization
|batchnorm        |  0                   | 1=apply batch normalization
|nonlin           |  tanh                | nonlinearity (tanh,sigm,relu)
|num_layers       |  2                   | number of hidden layers (if applicable)
|num_hidden       |  800                 | number of hidden neurons (if applicable)
|bias             |  1                   | 0=do not use bias
|rfb_mag          |  0                   | random feedback magnitude, 0=auto scale
|LR               |  0.0001              | learning rate
|LRDecay          |  0                   | learning rate decay (in # samples
|weightDecay      |  0                   | L2 penalty on the weights
|momentum         |  0                   | momentum
|batchSize        |  64                  | batch size
|optimization     |  rmsprop             | optimization method
|epoch            |  300                 | number of epochs to train (-1 for unbounded)
|epoch_step       |  -1                  | learning rate step, -1 for no step, 0 for auto, >0 for multiple of epochs to decrease
|gradient         |  dfa                 | gradient for learning, bp(back-prop), fa(feedback-alignment) or dfa(direct feedback-alignment)
|maxInNorm        |  400                 | max norm on incoming weights
|maxOutNorm       |  400                 | max norm on outgoing weights
|accGradient      |  0                   | 1=accumulate back-prop and adversarial gradient (if eps>0)
|threads          |  8                   | number of threads
|type             |  cuda                | float or cuda
|devid            |  1                   | device ID (if using CUDA)
|load             |  none                |  load existing net weights
|save             |  time-identifier     | save directory
|dataset          |  MNIST               | Dataset - Cifar10, Cifar100, STL10, SVHN, MNIST
|normalization    |  scale               | scale - between 0 and 1, simple - whole sample, channel - by image channel, image - mean and std images
|format           |  rgb                 | rgb or yuv
|whiten           |  false               | whiten data
|augment          |  false               | Augment training data
|preProcDir       |  ./PreProcData/      | Data for pre-processing (means,Pinv,P)
|validate         |  false               | use validation set for testing instead of test set
|visualize        |  0                   | 1=visualizing results
