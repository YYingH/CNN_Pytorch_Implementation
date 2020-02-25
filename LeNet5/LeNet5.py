import torch.nn as nn
import torch.nn.functional as F

class LeNet_5(nn.Module):
    def __init__(self, num_classes):
        super(LeNet_5, self).__init__()
        # N C H W
        self.conv1 = nn.Conv2d(3, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        
        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # flatten
        x = x.view(-1, self.num_flat_features(x))
        
        # fully connection
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return F.log_softmax(x, dim=1)
    
    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features*=s
        return num_features

if __name__ == "__main__":
    model = LeNet_5(num_classes = 6)
    print(model)