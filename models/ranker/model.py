"""
models/ranker/model.py
======================
Bi-encoder (dual-tower) question ranking model built from scratch in PyTorch.

Architecture:
  Two towers sharing the same trained tokenizer:
    - Query tower:    encodes an interview state descriptor string
    - Document tower: encodes a question bank entry

  Each tower:
    - Shared token embedding layer
    - 2-layer Transformer encoder (same architecture as classifier, smaller)
    - Mean pooling (over non-padding tokens)
    - L2 normalisation → embedding in R^128

  Similarity: cosine similarity between query and document embeddings.

At inference:
  - Precompute document embeddings for the entire question bank (offline)
  - Query is encoded on the fly and compared via cosine similarity
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.classifier.model import SinusoidalPositionalEncoding, TransformerEncoderLayer


# ────────────────────────────────────────────────────────────
# Single tower (shared between query and document)
# ────────────────────────────────────────────────────────────

class EncoderTower(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int  = 128,
        n_heads: int    = 4,
        ff_dim: int     = 512,
        n_layers: int   = 2,
        max_len: int    = 256,
        dropout: float  = 0.1,
        pad_id: int     = 0,
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

        # Initialise
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
            embedding: (batch, embed_dim) — L2-normalised mean-pooled representation
        """
        mask = (input_ids != self.pad_id)   # (B, T)  bool

        x = self.token_embedding(input_ids)  # (B, T, D)
        x = self.pos_encoding(x)

        for layer in self.layers:
            x = layer(x, mask.long())

        x = self.norm(x)

        # Mean pooling over non-padding tokens
        mask_f = mask.unsqueeze(-1).float()   # (B, T, 1)
        summed = (x * mask_f).sum(dim=1)      # (B, D)
        counts = mask_f.sum(dim=1).clamp(min=1e-9)
        pooled = summed / counts              # (B, D)

        # L2 normalise for cosine similarity
        return F.normalize(pooled, p=2, dim=-1)


# ────────────────────────────────────────────────────────────
# Bi-encoder model
# ────────────────────────────────────────────────────────────

class QuestionRanker(nn.Module):
    """
    Bi-encoder for question retrieval.

    Both towers share the same token embedding weights so that
    query and document are projected into the same embedding space.

    Similarity at training / inference: cosine similarity.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        n_heads: int   = 4,
        ff_dim: int    = 512,
        n_layers: int  = 2,
        max_len: int   = 256,
        dropout: float = 0.1,
        pad_id: int    = 0,
    ):
        super().__init__()

        self.query_tower = EncoderTower(
            vocab_size, embed_dim, n_heads, ff_dim, n_layers, max_len, dropout, pad_id
        )
        self.doc_tower = EncoderTower(
            vocab_size, embed_dim, n_heads, ff_dim, n_layers, max_len, dropout, pad_id
        )

        # Share token embedding weights between towers
        self.doc_tower.token_embedding.weight = self.query_tower.token_embedding.weight

    def encode_query(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.query_tower(input_ids)

    def encode_doc(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.doc_tower(input_ids)

    def forward(
        self,
        query_ids: torch.Tensor,
        pos_doc_ids: torch.Tensor,
        neg_doc_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Used during training (triplet / contrastive setup).

        Returns:
            q_emb:   (B, D)
            pos_emb: (B, D)
            neg_emb: (B, D)
        """
        q_emb   = self.encode_query(query_ids)
        pos_emb = self.encode_doc(pos_doc_ids)
        neg_emb = self.encode_doc(neg_doc_ids)
        return q_emb, pos_emb, neg_emb

    @property
    def num_params(self) -> int:
        # Subtract shared embedding params counted twice
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ────────────────────────────────────────────────────────────
# InfoNCE / contrastive loss
# ────────────────────────────────────────────────────────────

class InfoNCELoss(nn.Module):
    """
    In-batch negatives contrastive loss (InfoNCE).
    For each query, the positive is the correct document and
    all other documents in the batch are treated as negatives.
    Temperature τ controls the sharpness of the distribution.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        q_emb: torch.Tensor,    # (B, D)
        doc_emb: torch.Tensor,  # (B, D)  — positive documents, one per query
    ) -> torch.Tensor:
        # Similarity matrix: (B, B)
        sim = torch.matmul(q_emb, doc_emb.T) / self.temperature
        # Labels: each query i's positive is at index i (diagonal)
        labels = torch.arange(q_emb.size(0), device=q_emb.device)
        return nn.CrossEntropyLoss()(sim, labels)


if __name__ == "__main__":
    import math
    model = QuestionRanker(vocab_size=8000)
    print(f"QuestionRanker — {model.num_params:,} parameters")

    B, T = 4, 128
    q   = torch.randint(0, 8000, (B, T))
    pos = torch.randint(0, 8000, (B, T))
    neg = torch.randint(0, 8000, (B, T))

    q_emb, pos_emb, neg_emb = model(q, pos, neg)
    print(f"Query emb: {q_emb.shape}  Positive emb: {pos_emb.shape}")

    loss_fn = InfoNCELoss()
    loss    = loss_fn(q_emb, pos_emb)
    print(f"InfoNCE loss (random init): {loss.item():.4f}  (expected ~log(B)={math.log(B):.4f})")
