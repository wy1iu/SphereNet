# Deep Hyperspherical Learning

By Weiyang Liu, Yan-Ming Zhang, Xingguo Li, Zhiding Yu, Bo Dai, Tuo Zhao, Le Song

### License

SphereNet is released under the MIT License (refer to the LICENSE file for details).

### Code to be Updated
- [x] SphereNet: a neural network that learns on hyperspheres </li>
- [ ] SphereResNet: an adaptation of SphereConv to residual networks </li>
- [ ] Feature visualization on MNIST </li>

### Contents
0. [Introduction](#introduction)
0. [Citation](#citation)
0. [Requirements](#requirements)
0. [Usage](#usage)
0. [Results](#results)
0. [Notes](#notes)
0. [Third-party re-implementation](#third-party-re-implementation)
0. [Contact](#contact)


### Introduction

The repository contains an example Tensorflow implementation for SphereNets. SphereNets are introduced in the NIPS 2017 paper "[Deep Hyperspherical Learning](http://wyliu.com/papers/LiuNIPS17.pdf)" ([arXiv](https://arxiv.org/abs/1711.03189)). SphereNets are able to converge faster and more stably than its CNN counterparts, while yielding to comparable or even better classification accuracy. 

Hyperspherical learning is inspired by an interesting obvervation of the 2D Fourier transform. From the image below, we could see that magnitude information is not crucial for recognizing the identity, but phase information is very important for recognition. By droping the magnitude information, SphereNets can reduce the learning space and therefore gain more convergence speed. *Hypersphereical learning provides a new framework to improve the convolutional neural networks.*

<img src="asserts/2dfourier.png" width="52%" height="52%">

The features learned by SphereNets are also very interesting. The 2D features of SphereNets learned on MNIST are more compact and have larger margin between classes. From the image below, we can see that local behavior of convolutions could lead to dramatic difference in final features, even if they are supervised by the same standard softmax loss. *Hypersphereical learning provides a new perspective to think about convolutions and deep feature learning.*

<img src="asserts/mnist_featvis.jpg" width="51%" height="51%">

Besides, the hyperspherical learning also leads to a well-performing normalization technique, SphereNorm. SphereNorm basically can be viewed as SphereConv operator in our implementation.

### Citation

If you find our work useful in your research, please consider to cite:

    @inproceedings{liu2017deep,
        title={Deep Hyperspherical Learning},
        author={Liu, Weiyang and Zhang, Yan-Ming and Li, Xingguo and Yu, Zhiding and Dai, Bo and Zhao, Tuo and Song, Le},
        booktitle={Advances in Neural Information Processing Systems},
        pages={3953--3963},
        year={2017}
    }


### Requirements
1. `Python 2.7`
2. `TensorFlow` (Tested on version 1.01)
3. `numpy`


### Usage

#### Part 1: Setup
  - Clone the repositary and download the training set.

	```Shell
	git clone https://github.com/wy1iu/SphereNet.git
	cd SphereNet
	./dataset_setup.sh
	```

#### Part 2: Train Baseline/SphereNets

  - To train the baseline model, please open `baseline/train_baseline.py` and assign an available GPU. The default hyperparameters are exactly the same with SphereNets.

	```Shell
	python baseline/train_baseline.py
	```

  - To train the SphereNet, please open `train_spherenet.py` and assign an available GPU.

	```Shell
	python train_spherenet.py
	```


### Configuration
The default setting of SphereNet is Cosine SphereConv + Standard Softmax Loss. To change the type of SphereConv, please open the `spherenet.py` and change the `norm` variable. 

- If `norm` is set to `none`, then the network will use original convolution and become standard CNN. 
- If `norm` is set to `linear`, then the SphereNet will use linear SphereConv. 
- If `norm` is set to `cosine`, then the SphereNet will use cosine SphereConv. 
- If `norm` is set to `sigmoid`, then the SphereNet will use sigmoid SphereConv. 
- If `norm` is set to `lr_sigmoid`, then the SphereNet will use learnable sigmoid SphereConv. 

The `w_norm` variable can also be changed similarly in order to use the weight-normalized softmax loss (combined with different SphereConv). By setting `w_norm` to `none`, we will use the standard softmax loss.

There are some examples of setting these two variables provided in the `examples/` foloder.


### Results
#### Part 1: Convergence

The convergence curves for baseline CNN and several types of SphereNets are given as follows.
<img src="asserts/convergence.jpg" width="51%" height="51%">


#### Part 2: Best testing accuracy on CIFAR-10

- Baseline (standard CNN with standard softmax loss): 90.86%
- SphereNet with cosine SphereConv and standard softmax loss: 91.31%
- SphereNet with linear SphereConv and standard softmax loss: 91.65%
- SphereNet with sigmoid SphereConv and standard softmax loss: 91.81%
- SphereNet with learnable sigmoid SphereConv and standard softmax loss: 91.66%
- SphereNet with cosine SphereConv and weight-normalized softmax loss: 91.44%

#### Part 3: Training log

- Baseline: [here](baseline_training.log)    
- SphereNet with cosine SphereConv and standard softmax loss: [here](results/spherenet_cos_standard_training.log).
- SphereNet with linear SphereConv and standard softmax loss: [here](results/spherenet_linear_standard_training.log).
- SphereNet with sigmoid SphereConv and standard softmax loss: [here](results/spherenet_sigmoid_standard_training.log).
- SphereNet with learnable sigmoid SphereConv and standard softmax loss: [here](results/spherenet_lr_sigmoid_standard_training.log).
- SphereNet with cosine SphereConv and weight-normalized softmax loss: [here](results/spherenet_cos_wnsoftmax_training.log).

### Notes
- Empirically, SphereNets have more accuracy gain with larger filter number. If the filter number is very small, SphereNets may yield slightly worse accuracy but can still achieve much faster convergence.
- SphereConv may be useful for RNNs and deep Q-learning where better convergence can help.
- By adding rescaling factors to SphereConv and make them learnable in order for the SphereNorm to degrade to the original convolution, we present a new normalization technique, SphereNorm. SphereNorm does not contradict with the BatchNorm, and can be used either with or without BatchNorm

### Third-party re-implementation
- TensorFlow: [code](https://github.com/unixpickle/spherenet) by [unixpickle](https://github.com/unixpickle).


### Contact

- [Weiyang Liu](https://wyliu.com)

