import torch

class CNNPortOpt_L(torch.nn.Module):

    """
    A neural network model designed for portfolio optimization using 1D Convolutional layers.

    Attributes / Layers:
    - conv1 (torch.nn.Conv1d): 1D Convolutional layer for feature extraction.
    - fc (torch.nn.Linear): Fully connected layer for transforming convolutional outputs. Initialized dynamically based on the conv1 output.
    - sigmoid (torch.nn.Softmax): Sigmoid layer for portfolio allocations (for leverage).
    """

    def __init__(self, input_channels: int, hidden_size: int, output_size: int):
        
        """
        Initializes the CNNPortfolioOptimizer with the given dimensions.
        
        Parameters:
        - input_size (int): The number of input features.
        - hidden_size (int): The number of hidden units.
        - output_size (int): The size of the output.
        """

        super().__init__()
        self.conv1 = torch.nn.Conv1d(input_channels, hidden_size, kernel_size=5)
        self.fc = None
        self.sigmoid = torch.nn.Sigmoid()

        self._output_size = output_size
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv1(x))

        if self.fc is None:
            flattened_size = x.shape[1] * x.shape[2]
            self.fc = torch.nn.Linear(flattened_size, self._output_size)
        
        x = torch.flatten(x, 1)
        x = self.fc(x)
        output = self.sigmoid(x)
        return output
