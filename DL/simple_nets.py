import torch
import torch.nn.functional as F
import torch.nn as nn


class SimpleFCNet(nn.Module):
    def __init__(self, num_input, num_output, apply_log_softmax=True) -> None:
        super().__init__()
        self.num_input = num_input
        self.num_output = num_output
        self.apply_log_softmax = apply_log_softmax
        
        self.l1 = nn.Linear(self.num_input, 128)
        self.l2 = nn.Linear(128, 256)
        self.l3 = nn.Linear(256, 128)
        self.output = nn.Linear(128, self.num_output)
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = self.output(x)
        
        if self.apply_log_softmax:
            return self.log_softmax(x)
        else:
            return x
        
        
        
        