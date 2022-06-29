
import torch
import torch.nn.functional as F

class ConvNet(torch.nn.Module):
    def __init__(self, num_class):
        super(ConvNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 3) # 
        self.pool1 = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(32, 64, 3)
        self.pool2 = torch.nn.MaxPool2d(2, 2)
        self.conv3 = torch.nn.Conv2d(64, 128, 3)
        self.pool3 = torch.nn.MaxPool2d(2, 2)

        self.fc1 = torch.nn.Linear(128 * 4 * 4, 256)
        self.fc2 = torch.nn.Linear(256, num_class)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 4 * 4)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
    
    def get_feature(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 4 * 4)
        x = self.fc1(x)
        return x

class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.lin1 = torch.nn.Linear(20, 256) # 
        self.lin2 = torch.nn.Linear(256, 4) # 

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x
