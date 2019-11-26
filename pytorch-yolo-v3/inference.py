import torch
from torch import nn
import torchvision
import os
import struct
from torchsummary import summary
from darknet import Darknet
from util import *
from preprocess import prep_image, inp_to_image
import cv2

def main():

    confidence = 0.5
    nms_thesh = 0.4
    num_classes = 80
    classes = load_classes('data/coco.names')

    print('cuda device count: ', torch.cuda.device_count())
    print("Loading network.....")
    net = Darknet('cfg/yolov3.cfg')
    net.load_weights('yolov3.weights')
    print("Network successfully loaded")
    net = net.to('cuda:0')
    net = net.eval()
    print('print model')
    print('model: ', net)

    #------------------------input images------------------------------------------------
    input, origin, dim = prep_image('imgs/dog.jpg', 320);
    print('input:', input)
    input = input.to('cuda:0')
    print(input.shape)
    prediction = net(input, True)
    print('pre shape: ', prediction.shape)
    print('pre : ', prediction)
    prediction = write_results(prediction, confidence, num_classes, nms = True, nms_conf = nms_thesh)
    print('pre shape1: ', prediction.shape)
    print('pre1: ', prediction)

    scaling_factor = min(320/dim[0], 320/dim[1], 1)
    print(scaling_factor)
    prediction[:,[1,3]] -= (320 - scaling_factor*dim[0]) / 2
    prediction[:,[2,4]] -= (320 - scaling_factor*dim[1]) / 2
    print('pre2: ', prediction)
    prediction[:,1:5] /= scaling_factor
    print('pre3: ', prediction)

    for i in range(prediction.shape[0]):
        prediction[i, [1,3]] = torch.clamp(prediction[i, [1,3]], 0.0, dim[0])
        prediction[i, [2,4]] = torch.clamp(prediction[i, [2,4]], 0.0, dim[1])
    print('pre4: ', prediction)

    def write(x, batches, res):
        c1 = tuple(x[1:3].int())
        c2 = tuple(x[3:5].int())
        img = res
        cls = int(x[-1])
        label = "{0}".format(classes[cls])
        cv2.rectangle(img, c1, c2, (255, 0, 0), 1)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
        c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
        cv2.rectangle(img, c1, c2, (255, 0, 0), -1)
        cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1)
        return img

    list(map(lambda x: write(x, input, origin), prediction))
    cv2.imwrite('infout.png', origin)

    #------------------------input ones------------------------------------------------
    #print('state dict: ', net.state_dict().keys())
    tmp = torch.ones(1, 3, 320, 320).to('cuda:0')
    print('input: ', tmp)
    out = net(tmp)

    print('output:', out)

    summary(net, (3, 320, 320))
    #return
    f = open("yolov3.wts", 'w')
    f.write("{}\n".format(len(net.state_dict().keys())))
    for k,v in net.state_dict().items():
        print('key: ', k)
        print('value: ', v.shape)
        vr = v.reshape(-1).cpu().numpy()
        f.write("{} {}".format(k, len(vr)))
        for vv in vr:
            f.write(" ")
            f.write(struct.pack(">f", float(vv)).hex())
        f.write("\n")

if __name__ == '__main__':
    main()

