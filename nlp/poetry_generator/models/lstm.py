import torch.nn as nn

# ==========================================================
#  LSTM-based Language Model
# ==========================================================
class LSTM(nn.Module):
    """
    LSTM language model with embedding, dropout, and linear output.

    Args:
        vocab_size (int): size of the vocabulary
        em_dim (int): embedding dimension
        hidden_size (int): hidden size of LSTM
        n_layers (int): number of LSTM layers
        dropout (float): dropout probability
    """
    def __init__(self, vocab_size, em_dim, hidden_size, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, em_dim)  # token embeddings
        self.lstm = nn.LSTM(
            input_size=em_dim,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, vocab_size)       # output projection
        self.drop = nn.Dropout(dropout)

    def forward(self, x, hidden=None):
        """
        Forward pass through the LSTM model.

        Args:
            x (torch.LongTensor): input token indices [B, S]
            hidden (tuple of tensors, optional): previous hidden and cell states

        Returns:
            output (torch.FloatTensor): unnormalized logits [B, S, vocab_size]
            hidden (tuple): LSTM hidden and cell states
        """
        x = self.embedding(x)
        x = self.drop(x)
        x, hidden = self.lstm(x, hidden)
        x = self.drop(x)
        x = self.fc(x)
        return x, hidden