import torch
from torch import nn
from lenet5 import Lenet5
import os
import struct

def main():
    print('cuda device count: ', torch.cuda.device_count())
    net = torch.load('lenet5.pth')
    net = net.to('cuda:0')
    net.eval()
    #print('model: ', net)
    #print('state dict: ', net.state_dict()['conv1.weight'])
    tmp = torch.ones(1, 1, 32, 32).to('cuda:0')
    #print('input: ', tmp)
    out = net(tmp)
    print('lenet out:', out)

    f = open("lenet5.wts", 'w')
    f.write("{}\n".format(len(net.state_dict().keys())))
    for k,v in net.state_dict().items():
        #print('key: ', k)
        #print('value: ', v.shape)
        vr = v.reshape(-1).cpu().numpy()
        f.write("{} {}".format(k, len(vr)))
        for vv in vr:
            f.write(" ")
            f.write(struct.pack(">f", float(vv)).hex())
        f.write("\n")

if __name__ == '__main__':
    main()

