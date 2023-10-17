import torch
import pandas as pd

class LSTMPortfolioOptimizer(torch.nn.Module):
    """
    A neural network model designed for portfolio optimization using LSTM layers.
    
    Attributes / Layers:
    - lstm (torch.nn.LSTM): LSTM layer for sequence modeling.
    - fc (torch.nn.Linear): Fully connected layer for transforming LSTM outputs.
    - softmax (torch.nn.Softmax): Softmax layer for portfolio allocations (summing up to 1).
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
        self.lstm = torch.nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, output_size, bias=True)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        lstm_out_last = lstm_out[:, -1, :]
        fc_out = self.fc(lstm_out_last)
        output = self.softmax(fc_out)
        return output
    
class AttentionLSTMPortfolioOptimizer(torch.nn.Module):

    """
    A neural network model designed for portfolio optimization using LSTM layers augmented with an attention mechanism.
    
    Attributes / Layers:
    - lstm (torch.nn.LSTM): LSTM layer for sequence modeling.
    - attention (torch.nn.MultiheadAttention): Multihead attention layer to focus on specific parts of the LSTM output.
    - fc (torch.nn.Linear): Fully connected layer for transforming attention outputs.
    - softmax (torch.nn.Softmax): Softmax layer for portfolio allocations (summing up to 1).
    
    The model first uses the LSTM layer to capture the temporal dependencies across the entire input sequence.
    The attention mechanism then refines this output, focusing on the last 10% of the sequence.
    Finally, the output from the attention mechanism is passed through a fully connected layer and a softmax function
    to produce the portfolio allocations.
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int, attention_size: float = .1):

        """
        Initializes the AttentionLSTMPortfolioOptimizer with the given dimensions.
        
        Parameters:
        - input_size (int): The number of input features.
        - hidden_size (int): The number of hidden units.
        - output_size (int): The size of the output.
        - attention_size (int): A float between 0 and 1, representing the proportion of the sequence length that the 
          attention mechanism will focus on for the final portfolio allocation
        """

        super().__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, batch_first=True)
        self.attention = torch.nn.MultiheadAttention(hidden_size, num_heads=1)
        self.fc = torch.nn.Linear(hidden_size, output_size)
        self.softmax = torch.nn.Softmax(dim=-1)
        self._attention_size=attention_size
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x) 
        last_chunk_idx = int(lstm_out.shape[1] * (1 - self._attention_size))
        lstm_out_chunk = lstm_out[:, last_chunk_idx:, :]
        attn_out, _ = self.attention(lstm_out_chunk, lstm_out_chunk, lstm_out_chunk)
        attn_out_last = attn_out[:, -1, :]
        x = self.fc(attn_out_last)
        output = self.softmax(x)
        return output
    
class CNNPortfolioOptimizer(torch.nn.Module):

    """
    A neural network model designed for portfolio optimization using 1D Convolutional layers.

    Attributes / Layers:
    - conv1 (torch.nn.Conv1d): 1D Convolutional layer for feature extraction.
    - fc (torch.nn.Linear): Fully connected layer for transforming convolutional outputs. Initialized dynamically based on the conv1 output.
    - softmax (torch.nn.Softmax): Softmax layer for portfolio allocations (summing up to 1).

    The model uses a 1D Convolutional layer to extract relevant features from the input sequence.
    The output from this layer is dynamically flattened and passed through a fully connected layer to produce portfolio allocations.
    The fully connected layer's input dimensions are determined based on the output shape of the convolutional layer.
    Finally, a softmax function is applied to ensure the portfolio allocations sum to 1.
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
        self.conv1 = torch.nn.Conv1d(input_channels, hidden_size, kernel_size=3)
        self.fc = None
        self.softmax = torch.nn.Softmax(dim=-1)

        self._output_size = output_size
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv1(x))

        if self.fc is None:
            flattened_size = x.shape[1] * x.shape[2]
            self.fc = torch.nn.Linear(flattened_size, self._output_size)
        
        x = torch.flatten(x, 1)
        x = self.fc(x)
        output = self.softmax(x)
        return output
