import torch
    
class AttentionLSTMPortOpt_NL(torch.nn.Module):

    """
    A neural network model designed for portfolio optimization using LSTM layers augmented with an attention mechanism.
    
    Attributes / Layers:
    - lstm (torch.nn.LSTM): LSTM layer for sequence modeling.
    - attention (torch.nn.MultiheadAttention): Multihead attention layer to focus on specific parts of the LSTM output.
    - fc (torch.nn.Linear): Fully connected layer for transforming attention outputs.
    - softmax (torch.nn.Softmax): Softmax layer for portfolio allocations (summing up to 1).
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