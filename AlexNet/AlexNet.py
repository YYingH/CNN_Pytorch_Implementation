import torch
import torch.nn as nn
import torch.nn.functional as F

class LRN(nn.Module):
    def __init__(self, local_size = 1, alpha = 1.0, beta = 0.75, ACROSS_CHANNELS = False):
        super(LRN, self).__init__()
        self.ACROSS_CHANNELS = ACROSS_CHANNELS
        if self.ACROSS_CHANNELS:
            self.average = nn.AvgPool3d(kernel_size = (local_size, 1, 1),
                                       stride = 1,
                                       padding = (int((local_size - 1.0)/2), 0, 0))
        else:
            self.average = nn.AvgPool2d(kernel_size = local_size,
                                       stride = 1,
                                       padding=int((local_size-1.0)/2))
        self.alpha = alpha
        self.beta = beta
        
    def forward(self, x):
        if self.ACROSS_CHANNELS:
            div = x.pow(2).unsqueeze(1)
            div = self.average(div).squeeze(1)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)
        else:
            div = x.pow(2)
            div = self.average(div)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)
        x = x.div(div)
        return x

class AlexNet(nn.Module):
    def __init__(self, num_classes = 2):
        super().__init__()
        # N C H W
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 96, kernel_size = 11, stride = 4),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 3, stride = 2),
            LRN(local_size = 5, alpha = 1e-4, beta = 0.75, ACROSS_CHANNELS = True)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels = 96, out_channels = 256, kernel_size = 5, groups = 2, padding = 2),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 3, stride = 2),
            LRN(local_size = 5, alpha = 1e-4, beta = 0.75, ACROSS_CHANNELS = True)
        )
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels = 256, out_channels = 384, kernel_size = 3, padding = 1),
            nn.ReLU(inplace = True)
        )
        
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels = 384, out_channels = 384, kernel_size = 3, padding = 1),
            nn.ReLU(inplace = True)
        )
        
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels = 384, out_channels = 256, kernel_size = 3, padding = 2),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 3, stride = 2)
        )
        # change view
        self.layer6 = nn.Sequential(
            nn.Linear(in_features = 6*6*256, out_features = 4096),
            nn.ReLU(inplace = True),
            nn.Dropout()
        )
        
        self.layer7 = nn.Sequential(
            nn.Linear(in_features = 4096, out_features = 4096),
            nn.ReLU(inplace = True),
            nn.Dropout()
        )
        
        self.layer8 = nn.Linear(in_features = 4096, out_features = num_classes)

    def forward(self, x):
        x = self.layer5(self.layer4(self.layer3(self.layer2(self.layer1(x)))))
        x = x.view(-1, 6*6*256)
        x = self.layer8(self.layer7(self.layer6(x)))
        return x 

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
        if m.bias is not None:
            nn.init.zeros_(m.bias)

if __name__ == "__main__":
    model = AlexNet(num_classes=2)
    print(model)