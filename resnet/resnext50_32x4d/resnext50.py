import torch
from torch import nn
from torch.nn import functional as F
import torchvision

def main():
    print('cuda device count: ', torch.cuda.device_count())
    net = torchvision.models.resnext50_32x4d(pretrained=True)
    #net.fc = nn.Linear(512, 2)
    net = net.to('cuda:0')
    net.eval()
    print(net)
    tmp = torch.ones(2, 3, 224, 224).to('cuda:0')
    out = net(tmp)
    print('resnext50 out:', out.shape)
    torch.save(net, "resnext50.pth")

if __name__ == '__main__':
    main()

