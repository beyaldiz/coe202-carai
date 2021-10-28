import torch
import torch.nn as nn

class SingleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SingleBlock, self).__init__()

        self.linear = nn.Linear(in_channels, out_channels)
        self.act = nn.Sigmoid()
    
    def forward(self, x):
        out = self.linear(x)
        out = self.act(out)

        return out



class AIModel(nn.Module):
    def __init__(self, in_channels, hidden_layers, hidden_channels):
        super(AIModel, self).__init__()

        self.first_layer = nn.Linear(in_channels, hidden_channels)
        self.act = nn.Sigmoid()
        self.hidden_layer = nn.Sequential()
        for i in range(hidden_layers):
            self.hidden_layer.add_module(f"hidden_{i}", SingleBlock(hidden_channels, hidden_channels))
        
        self.last_layer = nn.Linear(hidden_channels, 3)
        self.softmax = nn.Softmax(dim=0)
    
    def forward(self, x):
        out = self.first_layer(x)
        out = self.act(out)
        out = self.hidden_layer(out)
        out = self.last_layer(out)
        out = self.softmax(out)

        return out