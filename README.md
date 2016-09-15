Training neural networks with back-prop, feedback-alignment and direct feedback-alignment
==========================================================================================

This repo contains code to reproduce experiments in paper
"Direct Feedback Alignment Provides Learning in Deep Neural Networks"
https://arxiv.org/abs/1609.01596

This code is copied and modified based on https://github.com/eladhoffer/ConvNet-torch

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
You can start training using:
```lua
th Main.lua -dataset Cifar10 -network conv.lua -LR 5e-5
```
or,
```lua
th Main.lua -dataset Cifar100 -network conv.lua -LR 5e-5
```

##Additional flags
===>Model options
  -modelsFolder   Models Folder [./Models/]
  -network        Model file - must return valid network. [mlp.lua]
  -criterion      criterion, ce(cross-entropy) or bce(binary cross-entropy) [bce]
  -eps            adversarial regularization magnitude (fast-sign-method a.la Goodfellow) [0]
  -dropout        apply dropout [0]
  -batchnorm      apply batch normalization [0]
  -nonlin         nonlinearity, (tanh,sigm,relu) [tanh]
  -num_layers     number of hidden layers (if applicable) [2]
  -num_hidden     number of hidden neurons (if applicable) [800]
  -bias           use bias or not [1]
  -rfb_mag        random feedback magnitude, 0=auto scale [0]
===>Training Regime
  -LR             learning rate [0.0001]
  -LRDecay        learning rate decay (in # samples) [0]
  -weightDecay    L2 penalty on the weights [0]
  -momentum       momentum [0]
  -batchSize      batch size [64]
  -optimization   optimization method [rmsprop]
  -epoch          number of epochs to train, -1 for unbounded [300]
  -epoch_step     learning rate step, -1 for no step, 0 for auto, >0 for multiple of epochs to decrease [-1]
  -gradient       gradient for learning (bp, fa or dfa) [dfa]
  -maxInNorm      max norm on incoming weights [400]
  -maxOutNorm     max norm on outgoing weights [400]
  -accGradient    accumulate back-prop and adversarial gradient (eps>0) [0]
===>Platform Optimization
  -threads        number of threads [8]
  -type           float or cuda [cuda]
  -devid          device ID (if using CUDA) [1]
  -nGPU           num of gpu devices used [1]
  -constBatchSize do not allow varying batch sizes - e.g for ccn2 kernel [false]
===>Save/Load Options
  -load           load existing net weights []
  -save           save directory [ThuSep1520:17:562016]
===>Data Options
  -dataset        Dataset - Cifar10, Cifar100, STL10, SVHN, MNIST [MNIST]
  -normalization  scale - between 0 and 1, simple - whole sample, channel - by image channel, image - mean and std images [scale]
  -format         rgb or yuv [rgb]
  -whiten         whiten data [false]
  -augment        Augment training data [false]
  -preProcDir     Data for pre-processing (means,P,invP) [./PreProcData/]
  -validate       use validation set for testing instead of test set [false]
===>Misc
  -visualize      visualizing results [0]