import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ==========================================================
#  Layer Normalization
# ==========================================================
class LayerNorm(nn.Module):
    """
    Layer normalization module.

    Args:
        d_model (int): dimensionality of input features
        bias (bool): whether to include a learnable bias term
    """
    def __init__(self, d_model, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


# ==========================================================
#  Multi-head Causal Self-Attention
# ==========================================================
class CausalSelfAttention(nn.Module):
    """
    Multi-head self-attention with causal masking for autoregressive models.

    Args:
        d_model (int): embedding dimension
        n_heads (int): number of attention heads
        dropout (float): dropout probability
        bias (bool): whether linear projections use bias
        max_len (int): maximum sequence length for mask
    """
    def __init__(self, d_model, n_heads, dropout, bias=False, max_len=2000):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # Combined QKV projection
        self.attn = nn.Linear(d_model, 3 * d_model, bias=bias)

        # Output projection
        self.fc_out = nn.Linear(d_model, d_model, bias=bias)

        # Dropout layers
        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        
        # Flash Attention if available
        self.flash = hasattr(F, 'scaled_dot_product_attention')
        
        if not self.flash:
            print("Using slow attention (Flash Attention requires PyTorch >= 2.0).")
            # Causal mask (lower-triangular)
            self.register_buffer(
                "mask",
                torch.tril(torch.ones(max_len, max_len)).view(1, 1, max_len, max_len)
            )

    def forward(self, x):
        B, S, C = x.size()

        # Compute q, k, v
        q, k, v = self.attn(x).split(C, dim=2)
        q = q.view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, S, self.n_heads, self.d_head).transpose(1, 2)

        # Scaled dot-product causal attention
        if self.flash:
            # Efficient attention
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # Manual implementation
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.d_head))
            att = att.masked_fill(self.mask[:, :, :S, :S] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # weighted sum of values
            
        y = y.transpose(1, 2).contiguous().view(B, S, C)

        return self.resid_dropout(self.fc_out(y))


# ==========================================================
#  Feed-Forward Layer
# ==========================================================
class FeedForward(nn.Module):
    """
    Position-wise feed-forward network.

    Args:
        d_model (int): embedding dimension
        dropout (float): dropout probability
    """
    def __init__(self, d_model, dropout):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.ff(x)


# ==========================================================
#  Transformer Decoder Block
# ==========================================================
class DecoderBlock(nn.Module):
    """
    Single decoder block consisting of:
        - LayerNorm
        - Causal self-attention
        - Feed-forward network
        - Residual connections
    """
    def __init__(self, d_model, n_heads, dropout, bias=False):
        super().__init__()
        self.ln1 = LayerNorm(d_model, bias=bias)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout, bias=bias)
        self.ln2 = LayerNorm(d_model, bias=bias)
        self.ff = FeedForward(d_model, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


# ==========================================================
#  Decoder-only Transformer (GPT)
# ==========================================================
class GPT(nn.Module):
    """
    GPT-style decoder-only transformer.

    Args:
        vocab_size (int): size of the vocabulary
        d_model (int): embedding dimension
        n_heads (int): number of attention heads
        n_layers (int): number of decoder blocks
        dropout (float): dropout probability
        bias (bool): whether to include bias in linear layers
        max_len (int): maximum sequence length
    """
    def __init__(self, vocab_size, d_model, n_heads, n_layers, dropout, bias=False, max_len=2000):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            DecoderBlock(d_model, n_heads, dropout, bias=bias) for _ in range(n_layers)
        ])
        self.ln_f = LayerNorm(d_model, bias=bias)
        self.fc_out = nn.Linear(d_model, vocab_size, bias=False)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        """
        Forward pass through the GPT model.

        Args:
            x (torch.LongTensor): input token indices [B, S]

        Returns:
            logits (torch.FloatTensor): unnormalized scores for each token [B, S, vocab_size]
        """
        B, S = x.size()
        pos = torch.arange(S, device=x.device).unsqueeze(0)

        # Token + positional embeddings
        x = self.drop(self.tok_emb(x) + self.pos_emb(pos))

        # Pass through decoder blocks
        for block in self.blocks:
            x = block(x)

        # Final layer norm and output projection
        x = self.ln_f(x)
        logits = self.fc_out(x)

        return logits