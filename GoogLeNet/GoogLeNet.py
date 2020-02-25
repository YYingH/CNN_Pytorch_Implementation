import torch
import torch.nn as nn

class Inception(nn.Module):
    def __init__(self, in_c, c1, c2, c3, c4):
        super().__init__()
        self.p1_1 = nn.Conv2d(in_c, c1, kernel_size = 1)
        
        self.p2_1 = nn.Conv2d(in_c, c2[0], 1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size = 3, padding = 1)
        
        self.p3_1 = nn.Conv2d(in_c, c3[0], kernel_size = 1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size = 5, padding = 2)
        
        self.p4_1 = nn.MaxPool2d(kernel_size = 3, stride = 1, padding = 1)
        self.p4_2 = nn.Conv2d(in_c, c4, kernel_size = 1)
        
    def forward(self, x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_1(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        return torch.cat((p1, p2, p3, p4), dim = 1)

class GoogLeNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.b1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        )
        
        self.b2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size = 1),
            nn.Conv2d(64, 192, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, padding = 1)
        )
        
        self.b3 = nn.Sequential(
            Inception(192, 64, (96, 128), (16, 32), 32),
            Inception(256, 128, (238, 192), (32, 96), 64),
            nn.MaxPool(kernel_size = 3, stride = 2, padding = 1)
        )
        
        self.b4 = nn.Sequential(
            Inception(480, 192, (96, 208), (16, 48), 64)
            Inception(512, 169, (112, 224), (24, 64), 64)
            Inception(512, 128, (128, 256), (24, 64), 64)
            Inception(512, 112, (144, 288), (32, 64), 64)
            Inception(528, 256, (160, 320), (32, 128), 128)
            nn.MaxPool(kernel_size = 3, stride = 2, padding = 1)
        )
        
        self.b5 = nn.Sequential(
            Inception(832, 256, (160, 320), ())
        )

if __name__ == "__main__":
    model = GoogLeNet(num_classes = 2)
    print(model)