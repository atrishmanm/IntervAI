"""
models/classifier/model.py
==========================
Small Transformer encoder classifier built from scratch in PyTorch.

Architecture:
  - Custom embedding layer (vocab_size × embed_dim)
  - Learned positional encoding
  - N Transformer encoder layers (multi-head self-attention + FFN)
  - [CLS] token pooling
  - 4-way classification head

Input format (token IDs):
  [CLS] student_answer [SEP] reference_answer [SEP] topic_text

Output:
  logits of shape (batch, 4) for classes:
    0 = correct  1 = partially_correct  2 = incorrect  3 = off_topic
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


LABEL2ID = {
    "correct":           0,
    "partially_correct": 1,
    "incorrect":         2,
    "off_topic":         3,
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
NUM_CLASSES = 4


# ────────────────────────────────────────────────────────────
# Positional Encoding (sinusoidal — no learned parameters)
# ────────────────────────────────────────────────────────────

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# ────────────────────────────────────────────────────────────
# Transformer Encoder Layer (built from scratch)
# ────────────────────────────────────────────────────────────

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.n_heads = n_heads
        self.d_head  = d_model // n_heads

        # Multi-head self-attention projections
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model)

        # Feed-forward sublayer
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
        )

        self.norm1   = nn.LayerNorm(d_model)
        self.norm2   = nn.LayerNorm(d_model)
        self.drop_attn = nn.Dropout(dropout)
        self.drop_ff   = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model) → (B, n_heads, T, d_head)
        B, T, _ = x.shape
        x = x.view(B, T, self.n_heads, self.d_head)
        return x.permute(0, 2, 1, 3)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # ── Multi-head self-attention ──
        residual = x
        x = self.norm1(x)

        Q = self._split_heads(self.W_q(x))   # (B, H, T, d_head)
        K = self._split_heads(self.W_k(x))
        V = self._split_heads(self.W_v(x))

        scale  = math.sqrt(self.d_head)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / scale  # (B, H, T, T)
        if mask is not None:
            # mask: (B, T) → broadcast to (B, 1, 1, T)
            scores = scores.masked_fill(
                mask.unsqueeze(1).unsqueeze(2) == 0, float("-inf")
            )
        attn = F.softmax(scores, dim=-1)
        attn = self.drop_attn(attn)

        out = torch.matmul(attn, V)                               # (B, H, T, d_head)
        out = out.permute(0, 2, 1, 3).contiguous()
        out = out.view(out.size(0), out.size(1), -1)              # (B, T, d_model)
        out = self.W_o(out)
        x = residual + out

        # ── Feed-forward ──
        residual = x
        x = self.norm2(x)
        x = residual + self.drop_ff(self.ff(x))

        return x


# ────────────────────────────────────────────────────────────
# Classifier Model
# ────────────────────────────────────────────────────────────

class AnswerClassifier(nn.Module):
    """
    Small Transformer encoder classifier.

    Hyperparameters (defaults sized for fast training on modest hardware):
        vocab_size  : must match trained tokenizer vocab size
        embed_dim   : 128
        n_heads     : 4
        ff_dim      : 512
        n_layers    : 3
        max_len     : 256
        dropout     : 0.1
        num_classes : 4
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int   = 128,
        n_heads: int     = 4,
        ff_dim: int      = 512,
        n_layers: int    = 3,
        max_len: int     = 256,
        dropout: float   = 0.1,
        num_classes: int = NUM_CLASSES,
        pad_id: int      = 0,
    ):
        super().__init__()
        self.pad_id    = pad_id
        self.embed_dim = embed_dim

        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.pos_encoding    = SinusoidalPositionalEncoding(embed_dim, max_len, dropout)

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, n_heads, ff_dim, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            elif "bias" in name:
                nn.init.zeros_(p)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (batch, seq_len) long tensor
        Returns:
            logits: (batch, num_classes)
        """
        mask = (input_ids != self.pad_id).long()   # (B, T)

        x = self.token_embedding(input_ids)        # (B, T, embed_dim)
        x = self.pos_encoding(x)

        for layer in self.layers:
            x = layer(x, mask)

        x = self.norm(x)

        # Pool using [CLS] token (position 0)
        cls_repr = x[:, 0, :]                      # (B, embed_dim)
        logits   = self.classifier(cls_repr)       # (B, num_classes)
        return logits

    def predict(self, input_ids: torch.Tensor) -> list[str]:
        """Convenience: returns string label names."""
        self.eval()
        with torch.no_grad():
            logits = self(input_ids)
            preds  = logits.argmax(dim=-1).tolist()
        return [ID2LABEL[p] for p in preds]

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick architecture check
    model = AnswerClassifier(vocab_size=8000)
    print(f"AnswerClassifier — {model.num_params:,} trainable parameters")
    dummy = torch.randint(0, 8000, (4, 256))
    logits = model(dummy)
    print(f"Input shape: {dummy.shape}  →  Output shape: {logits.shape}")
    print(f"Labels: {model.predict(dummy)}")
