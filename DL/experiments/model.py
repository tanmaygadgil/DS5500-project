import torch.nn as nn
from torch import optim

def simple_model(input_shape, output_shape):
    model = nn.Sequential(
        nn.Linear(input_shape, 32),  # Input layer: Fully connected (linear) with 64 units
        nn.ReLU(),  # Activation function: ReLU
        nn.Dropout(0.3),
        nn.Linear(32, 64),
        nn.ReLU(),  # Activation function: ReLU
        nn.Dropout(0.3),
        nn.Linear(64, output_shape)  # Output layer: Fully connected (linear) with 'output_shape' units
    )
    
    return model