# PyTorchx

This is a brother project with [wang-xinyu/tensorrtx](https://github.com/wang-xinyu/tensorrtx).

Popular deep learning networks are implemented with pytorch in this project. And then weights files are exported for tensorrt implementation.

## Test Environments
1. Python 3.7.3
2. cuda 10.0
3. PyTorch 1.3.0
4. torchvision 0.4.1

## prepare pytorch-summary

pytorch-summary is a very useful tool for understanding the model structure, for example it can output the dimensions of each layer.

Clone, and `cd` into the repo directory.

```
git clone https://github.com/sksq96/pytorch-summary
python setup.py build
python setup.py install
```

## Run

Most of the models are from torchvision, exception for yolov3, which has a readme inside.

A file named `xxxnet.py` can do inference and save model into .pth.
And a file named `inference.py` can do inference and save weights into .wts, which is used for tensorrt.

For example, googlenet,

```
cd googlenet
python googlenet.py  // do inference and save model into .pth firstly.
python inference.py // then do inference and save weights file
```
