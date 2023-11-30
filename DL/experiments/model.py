import torch.nn as nn
from torch import optim

def simple_model(input_shape, output_shape):
    model = nn.Sequential(
        nn.Linear(input_shape, 64),  # Input layer: Fully connected (linear) with 64 units
        nn.ReLU(),  # Activation function: ReLU,
        nn.Dropout(0.3),
        nn.Linear(64, 128),
        nn.ReLU(),  # Activation function: ReLU
        nn.Dropout(0.3),
        # nn.Linear(128, 128),
        # nn.ReLU(),  # Activation function: ReLU
        # nn.Dropout(0.3),
        nn.Linear(128, output_shape)  # Output layer: Fully connected (linear) with 'output_shape' units
    )
    
    return model


class SimpleModel(nn.Module):
    def __init__(self, input_shape, output_shape, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.l1 = nn.Linear(input_shape, 64)
        self.l2 = nn.Linear(64, 256)
        self.l3 = nn.Linear(256, 128)
        self.l4 = nn.Linear(128, output_shape)
        
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        
        x = self.relu(self.l1(x))
        x = self.dropout(x)
        x = self.relu(self.l2(x))
        x = self.dropout(x)
        x = self.relu(self.l3(x))
        x = self.dropout(x)
        x = self.l4(x)
        
        return x