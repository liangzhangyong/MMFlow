# Manifold Matching via Deep Metric learning for Improving Generative Flow-base Model

<p align="center">
<img align="middle" src="./assets/MMFlow.jpg" width="512" />
</p>

<p align="center">
<img align="middle" src="./assets/noise_sphere.gif" width="512" />
</p>

+ Unbiased estimation of the log-density of samples.

+ Memory-efficient reformulation of the gradients.

+ Implicit symmetric bi-jective map function.

+ Bi-LipSwish activation function.

+ Lightweights neural networks with 1x1conv and multi-axis self-attention.

  <p align="center">
  <img align="middle" src="./assets/flow_comparison.jpg" width="666" />
  </p>

As a result, Manifold  via Metric Flows scale to much larger networks and datasets.

<p align="center">
<img align="middle" src="./assets/result_mmflow.jpg" width="666" />
</p>


## Requirements

 - PyTorch 1.8+
 - Python 3.8+

## Preprocessing
ImageNet:
1. Follow instructions in `preprocessing/create_imagenet_benchmark_datasets`.
2. Convert .npy files to .pth using `preprocessing/convert_to_pth`.
3. Place in `data/imagenet32` and `data/imagenet64`.

CelebAHQ 64x64 5bit:

1. Download from https://github.com/aravindsrinivas/flowpp/tree/master/flows_celeba.
2. Convert .npy files to .pth using `preprocessing/convert_to_pth`.
3. Place in `data/celebahq64_5bit`.

CelebAHQ 256x256:
```
# Download Glow's preprocessed dataset.
wget https://storage.googleapis.com/glow-demo/data/celeba-tfr.tar
tar -C data/celebahq -xvf celeb-tfr.tar
python extract_celeba_from_tfrecords
```

## Density Estimation Experiments

***NOTE***: By default, O(1)-memory gradients are enabled. However, the logged bits/dim during training will not be an actual estimate of bits/dim but whatever scalar was used to generate the unbiased gradients. If you want to check the actual bits/dim for training (and have sufficient GPU memory), set `--neumann-grad=False`. Note however that the memory cost can stochastically vary during training if this flag is `False`.

MNIST:
```
python train_img.py --data mnist --imagesize 28 --actnorm True --wd 0 --save experiments/mnist
```

CIFAR10:
```
Generative Flow-based Model with Our Backbone.
python train_img.py --batchsize 64 --data cifar10 --actnorm True --save experiments/cifar10/ResFlow --model ResidualFlow
python train_img.py --batchsize 64 --data cifar10 --actnorm True --save experiments/cifar10/ResXFlow --model ResidualXFlow
python train_img.py --batchsize 64 --data cifar10 --actnorm True --save experiments/cifar10/ConvXFlow --model ConvNextFlow
python train_img.py --batchsize 64 --data cifar10 --actnorm True --save experiments/cifar10/VANFlow --model VANFlow
python train_img.py --batchsize 64 --data cifar10 --actnorm True --save experiments/cifar10/CoAtFlow --model CoAtFlow
python train_img.py --batchsize 64 --data cifar10 --actnorm True --save experiments/cifar10/MaxVitFlow --model MaxViTFlow

Manifold Matching via Deep Metric Learning for Our Generative Flow-based Model.
python train_img.py --batchsize 64 --data cifar10 --actnorm True --save experiments/cifar10/MMResFlow --model ResidualFlow
python train_img.py --batchsize 64 --data cifar10 --actnorm True --save experiments/cifar10/MMResXFlow --model ResidualXFlow
python train_img.py --batchsize 64 --data cifar10 --actnorm True --save experiments/cifar10/MMConvXFlow --model ConvNextFlow
python train_img.py --batchsize 64 --data cifar10 --actnorm True --save experiments/cifar10/MMVANFlow --model VANFlow
python train_img.py --batchsize 64 --data cifar10 --actnorm True --save experiments/cifar10/MMCoAtFlow --model CoAtFlow
python train_img.py --batchsize 64 --data cifar10 --actnorm True --save experiments/cifar10/MMMaxViTFlow --model MaxViTFlow
```

ImageNet 32x32:
```
python train_img.py --data imagenet32 --actnorm True --nblocks 32-32-32 --save experiments/imagenet32
```

ImageNet 64x64:
```
python train_img.py --data imagenet64 --imagesize 64 --actnorm True --nblocks 32-32-32 --factor-out True --squeeze-first True --save experiments/imagenet64
```

CelebAHQ 256x256:
```
python train_img.py --data celebahq --imagesize 256 --nbits 5 --actnorm True --act elu --batchsize 8 --update-freq 5 --n-exact-terms 8 --fc-end False --factor-out True --squeeze-first True --nblocks 16-16-16-16-16-16 --save experiments/celebahq256
```

## Pretrained Models

Model checkpoints can be downloaded from (To submit)

Use the argument `--resume [checkpt.pth]` to evaluate or sample from the model. 

Each checkpoint contains two sets of parameters, one from training and one containing the exponential moving average (EMA) accumulated over the course of training. Scripts will automatically use the EMA parameters for evaluation and sampling.

## BibTeX -- To submit
```
@inproceedings{liangMMFlow2022,
  title={Manifold Matching via Deep Metric learning for Improving Generative Flow-base Model},
  author={ZY Liang},
  booktitle = {},
  year={2022}
}
```
