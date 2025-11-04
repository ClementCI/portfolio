import torch.nn as nn

# ==========================================================
#  Vanilla RNN-based Language Model
# ==========================================================
class RNN(nn.Module):
    """
    Vanilla RNN language model with embedding, dropout, and linear output.

    Args:
        vocab_size (int): size of the vocabulary
        em_dim (int): embedding dimension
        hidden_size (int): hidden size of RNN
        n_layers (int): number of RNN layers
        dropout (float): dropout probability
    """
    def __init__(self, vocab_size, em_dim, hidden_size, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, em_dim)  # token embeddings
        self.rnn = nn.RNN(
            input_size=em_dim,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, vocab_size)       # output projection
        self.drop = nn.Dropout(dropout)

    def forward(self, x, hidden=None):
        """
        Forward pass through the RNN model.

        Args:
            x (torch.LongTensor): input token indices [B, S]
            hidden (torch.Tensor, optional): previous hidden state

        Returns:
            output (torch.FloatTensor): unnormalized logits [B, S, vocab_size]
            hidden (torch.Tensor): RNN hidden state
        """
        x = self.embedding(x)
        x = self.drop(x)
        x, hidden = self.rnn(x, hidden)
        x = self.drop(x)
        x = self.fc(x)
        return x, hidden