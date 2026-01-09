import torch
import torch.nn as nn

class TinyMLP(nn.Module):            # nn.Module is base for any model

    """
    Initializes the base class and then creates the math objects for further
    processing.
    """
    def __init__(self):              # Constructor, builds the class instance
        super().__init__()           # Constructor of the nn.Module
        self.fc1 = nn.Linear(10, 32) # Input layer processor
        self.act = nn.ReLU()         # Activation function processor
        self.fc2 = nn.Linear(32, 1)  # Output layer processor

    """
    Pass the input vector through the set of the math operations.
    """
    def forward(self, x):            # x should be a torch.Tensor
        x = self.fc1(x)              # y = xW + b, W and b is built-in
        x = self.act(x)              # y = max(0, x)
        x = self.fc2(x)              # y = xW + b, W and b is built-in
        return x
