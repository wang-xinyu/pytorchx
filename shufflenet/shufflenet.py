import torch
from torch import nn
from torch.nn import functional as F
import torchvision

def main():
    print('cuda device count: ', torch.cuda.device_count())
    net = torchvision.models.shufflenet_v2_x0_5(pretrained=True)
    #net.fc = nn.Linear(512, 2)
    net = net.eval()
    net = net.to('cuda:0')
    print(net)
    tmp = torch.ones(2, 3, 224, 224).to('cuda:0')
    out = net(tmp)
    print('shufflenet out:', out.shape)
    torch.save(net, "shufflenet.pth")

if __name__ == '__main__':
    main()

