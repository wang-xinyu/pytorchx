# Pytorch YOLO v3

This is forked from [ayooshkathuria/pytorch-yolo-v3](https://github.com/ayooshkathuria/pytorch-yolo-v3)

I added `inference.py` to do inference on one pic and export weights file for tensorrt.

## Test Environments
1. Python 3.7.3
2. cuda 10.0
3. PyTorch 1.3.0
4. torchvision 0.4.1

## Run

Clone, and `cd` into the repo directory. The first thing you need to do is to get the weights file
This time around, for v3, authors has supplied a weightsfile only for COCO [here](https://pjreddie.com/media/files/yolov3.weights), and place 

the weights file into your repo directory. Or, you could just type (if you're on Linux)

```
wget https://pjreddie.com/media/files/yolov3.weights 
python inference.py
```

