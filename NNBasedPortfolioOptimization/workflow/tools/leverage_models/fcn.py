import torch

class FCNPortOpt_L(torch.nn.Module):
    """
    A neural network model designed for portfolio optimization using LSTM layers.
    
    Attributes / Layers:
    - lstm (torch.nn.LSTM): LSTM layer for sequence modeling.
    - fc (torch.nn.Linear): Fully connected layer for transforming LSTM outputs.
    - sigmoid (torch.nn.Softmax): Sigmoid layer for portfolio allocations (for leverage).
    """
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        """
        Initializes the LSTMPortfolioOptimization with the given dimensions.
        
        Parameters:
        - input_size (int): The number of input features.
        - hidden_size (int): The number of hidden units.
        - output_size (int): The size of the output.
        """
        super().__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size, bias=True)
        self.fc2 = torch.nn.Linear(hidden_size, output_size, bias=True)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if len(x.shape) > 1:
            x = torch.flatten(x, start_dim=1, end_dim=-1)
    
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        output = self.sigmoid(x)
        return output