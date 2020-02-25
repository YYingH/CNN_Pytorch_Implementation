import math
import torch
import torch.nn as nn

net_11 = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
net_13 = [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
net_16 = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
net_19 = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']

class VGGNet(nn.Module):
    def __init__(self, net_arch, num_classes, batch_norm = False):
        super().__init__()
        self.num_classes = num_classes
        
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(7*7*512, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 1000),
            nn.ReLU(True),
            nn.Linear(1000, self.num_classes)
        )
        
        layers = []
        in_channels = 3
        for arch in net_arch:
            if arch == 'M':
                layers.append(nn.MaxPool2d(kernel_size = 2, stride = 2))
            else:
                layers.append(nn.Conv2d(in_channels = in_channels, out_channels = arch,
                                            kernel_size = 3, padding = 1))
                if batch_norm:
                    layers.append(nn.BatchNorm2d(in_channels = arch))
                layers.append(nn.ReLU(inplace = True))
                in_channels = arch
            
        self.vgg = nn.ModuleList(layers)
        
        
        
    def forward(self, x):
        for layer in self.vgg:
            x = layer(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def weight_init(m):
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        m.bias.data.zero_()


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 2 every 20 epochs"""
    lr = learning_rate * (0.5 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == "__main__":
    model = VGGNet(net_16, 2)
    print(model)