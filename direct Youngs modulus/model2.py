import torch
import torch.nn as nn
import torch.nn.functional as F

input_size = 500
output_size = 1
hidden_size =1000

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / \
            torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)

def normalized_columns_initializer(weights, std=1.0):
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1, keepdim=True))
    return out

class Network(nn.Module):
    
    def __init__(self):
        super(Network, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l3 = nn.Linear(hidden_size, 256)
        self.relu = nn.ReLU()
        self.l4 = nn.Linear(256, output_size)
        self.apply(weights_init)
        self.l1.weight.data = normalized_columns_initializer(self.l1.weight.data, 0.01)
        self.l1.bias.data.fill_(0)
        self.l3.weight.data = normalized_columns_initializer(self.l3.weight.data, 0.01)
        self.l3.bias.data.fill_(0)
        self.l4.weight.data = normalized_columns_initializer(self.l4.weight.data, 0.01)
        self.l4.bias.data.fill_(0)

    def forward(self, x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.l3(x)
        x = self.relu(x)
        x = self.l4(x)
        return x
        #F.log_softmax(x)