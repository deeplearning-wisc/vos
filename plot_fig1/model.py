from torch import  nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim, class_num):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 100)
        self.fc2 = nn.Linear(100, 200)
        self.fc3 = nn.Linear(200, class_num)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
