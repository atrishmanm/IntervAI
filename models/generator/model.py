"""
models/generator/model.py
==========================
Decoder-only Transformer (GPT-style) for coding interview dialogue.

Architecture:
  - Token Embedding (vocab_size × embed_dim)
  - Learned Positional Embedding (max_len × embed_dim)
  - N Transformer Decoder Layers (masked self-attention + FFN)
  - Pre-LayerNorm for stable training
  - Language Model Head (embed_dim → vocab_size, tied with embedding)

Target sizes:
  Small  (~8M):  embed_dim=192, n_layers=6,  ff_dim=768,  n_heads=4
  Medium (~20M): embed_dim=256, n_layers=8,  ff_dim=1024, n_heads=4
  Large  (~45M): embed_dim=384, n_layers=10, ff_dim=1536, n_heads=6
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GeneratorConfig:
    """Configuration for the generator model."""
    vocab_size: int = 16000
    embed_dim: int = 256
    n_heads: int = 4
    n_layers: int = 8
    ff_dim: int = 1024
    max_len: int = 1024
    dropout: float = 0.1
    pad_id: int = 0
    tie_weights: bool = True  # Tie LM head with token embedding

    @property
    def head_dim(self):
        return self.embed_dim // self.n_heads

    def estimate_params(self):
        """Estimate total parameter count."""
        # Token embedding
        emb = self.vocab_size * self.embed_dim
        # Position embedding
        pos = self.max_len * self.embed_dim
        # Per transformer layer: 4 * d^2 (Q,K,V,O) + 2 * d * ff (FFN) + 2 * d (norms)
        per_layer = (
            4 * self.embed_dim ** 2
            + 2 * self.embed_dim * self.ff_dim
            + 2 * self.embed_dim
        )
        layers = self.n_layers * per_layer
        # Final norm
        final_norm = self.embed_dim
        # LM head (tied = 0 extra params)
        lm_head = 0 if self.tie_weights else self.embed_dim * self.vocab_size
        total = emb + pos + layers + final_norm + lm_head
        return total


# ─────────────────────────────────────────────────────────────
# Positional Encoding (learned)
# ─────────────────────────────────────────────────────────────

class LearnedPositionalEmbedding(nn.Module):
    """Learned positional embeddings."""
    def __init__(self, max_len: int, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.pos_emb = nn.Embedding(max_len, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T)
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        return self.dropout(x + self.pos_emb(positions))


# ─────────────────────────────────────────────────────────────
# Transformer Decoder Layer (from scratch)
# ─────────────────────────────────────────────────────────────

class TransformerDecoderLayer(nn.Module):
    """
    Single Transformer decoder layer with:
    - Masked multi-head self-attention
    - Feed-forward network
    - Pre-LayerNorm (applied before each sublayer)
    - Residual connections
    """
    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        ff_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert embed_dim % n_heads == 0, "embed_dim must be divisible by n_heads"

        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads

        # Self-attention projections
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_o = nn.Linear(embed_dim, embed_dim)

        # Feed-forward
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )

        # Layer norms (Pre-LN)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.drop_attn = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        x = x.view(B, T, self.n_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)  # (B, H, T, head_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        B, T, _ = x.shape

        # ── Masked Multi-Head Self-Attention (Pre-LN) ──
        residual = x
        x_norm = self.norm1(x)

        Q = self._split_heads(self.W_q(x_norm))
        K = self._split_heads(self.W_k(x_norm))
        V = self._split_heads(self.W_v(x_norm))

        scale = math.sqrt(self.head_dim)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / scale  # (B, H, T, T)

        # Apply causal mask (prevent attending to future tokens)
        if mask is not None:
            # mask: (B, T) with 1 for valid, 0 for pad
            # We need a causal mask + padding mask combined
            causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
            scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
            # Padding mask
            pad_mask = (mask == 0).unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(pad_mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.drop_attn(attn)

        out = torch.matmul(attn, V)  # (B, H, T, head_dim)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, T, -1)
        out = self.W_o(out)
        x = residual + out

        # ── Feed-Forward (Pre-LN) ──
        residual = x
        x_norm = self.norm2(x)
        x = residual + self.ff(x_norm)

        return x


# ─────────────────────────────────────────────────────────────
# Generator Model
# ─────────────────────────────────────────────────────────────

class InterviewGenerator(nn.Module):
    """
    Decoder-only Transformer for coding interview dialogue.

    GPT-style architecture with:
    - Learned positional embeddings
    - Pre-LayerNorm transformer layers
    - Weight-tied language model head
    - Causal attention masking

    Usage:
        config = GeneratorConfig(vocab_size=16000, embed_dim=256, n_layers=8)
        model = InterviewGenerator(config)
        logits = model(input_ids)  # (B, T, vocab_size)
        next_token_logits = model.generate(prompt_ids, max_new_tokens=100)
    """

    def __init__(self, config: GeneratorConfig):
        super().__init__()
        self.config = config
        self.pad_id = config.pad_id

        # Embeddings
        self.token_emb = nn.Embedding(config.vocab_size, config.embed_dim, padding_idx=config.pad_id)
        self.pos_emb = LearnedPositionalEmbedding(config.max_len, config.embed_dim, config.dropout)
        self.emb_dropout = nn.Dropout(config.dropout)

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(
                embed_dim=config.embed_dim,
                n_heads=config.n_heads,
                ff_dim=config.ff_dim,
                dropout=config.dropout,
            )
            for _ in range(config.n_layers)
        ])

        # Final layer norm
        self.norm = nn.LayerNorm(config.embed_dim)

        # Language model head
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)

        # Weight tying
        if config.tie_weights:
            self.lm_head.weight = self.token_emb.weight

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with scaled normal for better training stability."""
        for name, p in self.named_parameters():
            if "token_emb" in name or "pos_emb" in name:
                nn.init.normal_(p, mean=0.0, std=0.02)
            elif "W_o" in name or "ff" in name:
                # Output projections and FFN
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
                else:
                    nn.init.zeros_(p)
            elif "norm" in name:
                if "weight" in name:
                    nn.init.ones_(p)
                elif "bias" in name:
                    nn.init.zeros_(p)
            elif p.dim() > 1:
                nn.init.normal_(p, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor = None,
    ) -> dict:
        """
        Forward pass.

        Args:
            input_ids: (B, T) token IDs
            labels: (B, T) target IDs for language modeling (shifted right)

        Returns:
            dict with 'logits' (B, T, vocab_size) and optionally 'loss'
        """
        B, T = input_ids.shape
        mask = (input_ids != self.pad_id).long()

        # Embed
        x = self.token_emb(input_ids)  # (B, T, embed_dim)
        x = self.pos_emb(x)
        x = self.emb_dropout(x)

        # Transformer layers
        for layer in self.layers:
            x = layer(x, mask)

        x = self.norm(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        result = {"logits": logits}

        # Compute loss if labels provided
        if labels is not None:
            # Shift: predict next token
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=self.pad_id,
            )
            result["loss"] = loss

        return result

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        eos_token_id: int = None,
    ) -> torch.Tensor:
        """
        Generate text autoregressively.

        Args:
            prompt_ids: (B, T) prompt token IDs
            max_new_tokens: maximum tokens to generate
            temperature: sampling temperature
            top_p: nucleus sampling threshold
            top_k: top-k sampling
            eos_token_id: stop generating when this token is produced

        Returns:
            (B, T + generated) full sequence including prompt
        """
        self.eval()
        generated = prompt_ids.clone()

        for _ in range(max_new_tokens):
            # Forward pass
            result = self.forward(generated)
            logits = result["logits"][:, -1, :]  # Last token logits

            # Temperature
            logits = logits / temperature

            # Top-k filtering
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float("-inf")

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float("-inf")

            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append
            generated = torch.cat([generated, next_token], dim=1)

            # Stop if EOS
            if eos_token_id is not None:
                if (next_token == eos_token_id).all():
                    break

            # Stop if max length reached
            if generated.size(1) >= self.config.max_len:
                break

        return generated

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def num_params_millions(self) -> float:
        return self.num_params / 1_000_000


# ─────────────────────────────────────────────────────────────
# Factory functions
# ─────────────────────────────────────────────────────────────

def create_small_model(vocab_size=16000) -> InterviewGenerator:
    """~8M parameter model for GTX 1650."""
    config = GeneratorConfig(
        vocab_size=vocab_size,
        embed_dim=192,
        n_heads=4,
        n_layers=6,
        ff_dim=768,
        max_len=1024,
    )
    model = InterviewGenerator(config)
    print(f"Small model: {model.num_params_millions:.1f}M params")
    return model


def create_medium_model(vocab_size=16000) -> InterviewGenerator:
    """~20M parameter model for Colab T4."""
    config = GeneratorConfig(
        vocab_size=vocab_size,
        embed_dim=256,
        n_heads=4,
        n_layers=8,
        ff_dim=1024,
        max_len=1024,
    )
    model = InterviewGenerator(config)
    print(f"Medium model: {model.num_params_millions:.1f}M params")
    return model


def create_large_model(vocab_size=16000) -> InterviewGenerator:
    """~45M parameter model for larger GPU."""
    config = GeneratorConfig(
        vocab_size=vocab_size,
        embed_dim=384,
        n_heads=6,
        n_layers=10,
        ff_dim=1536,
        max_len=1024,
    )
    model = InterviewGenerator(config)
    print(f"Large model: {model.num_params_millions:.1f}M params")
    return model


if __name__ == "__main__":
    print("=" * 60)
    print("InterviewGenerator Architecture Check")
    print("=" * 60)

    for name, factory in [
        ("Small (8M)", create_small_model),
        ("Medium (20M)", create_medium_model),
        ("Large (45M)", create_large_model),
    ]:
        print(f"\n{name}:")
        model = factory()
        config = model.config
        print(f"  embed_dim={config.embed_dim}, n_layers={config.n_layers}, "
              f"ff_dim={config.ff_dim}, n_heads={config.n_heads}")
        print(f"  Estimated params: {config.estimate_params():,}")

        # Forward pass test
        dummy = torch.randint(0, 16000, (2, 128))
        result = model(dummy)
        print(f"  Input: {dummy.shape} -> Logits: {result['logits'].shape}")

        # Generation test
        prompt = torch.randint(0, 16000, (1, 10))
        generated = model.generate(prompt, max_new_tokens=20)
        print(f"  Prompt: {prompt.shape} -> Generated: {generated.shape}")
