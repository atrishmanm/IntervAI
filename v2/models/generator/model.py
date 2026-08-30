"""
models/generator/model.py — v2 (Research-Grade Architecture)
=============================================================
Decoder-only Transformer (GPT-style) for coding interview dialogue.

Architecture (v2):
  - Token Embedding (vocab_size x embed_dim)
  - Rotary Position Embedding (RoPE) — no learned positional params
  - N Transformer Decoder Layers:
      - Pre-RMSNorm
      - Flash Attention via F.scaled_dot_product_attention
      - SwiGLU Feed-Forward Network
  - Final RMSNorm
  - Language Model Head (embed_dim -> vocab_size, tied with embedding)
  - Optional gradient checkpointing for VRAM savings

Target sizes:
  Small  (~12M):  embed_dim=192, n_layers=8,  ff_dim=512,  n_heads=4
  Medium (~45M):  embed_dim=384, n_layers=12, ff_dim=1408, n_heads=6
  Large  (~120M): embed_dim=512, n_layers=16, ff_dim=1408, n_heads=8
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
    embed_dim: int = 768
    n_heads: int = 12
    n_layers: int = 16
    ff_dim: int = 2048
    max_len: int = 2048
    dropout: float = 0.1
    pad_id: int = 0
    tie_weights: bool = True
    gradient_checkpointing: bool = False
    label_smoothing: float = 0.0

    @property
    def head_dim(self):
        return self.embed_dim // self.n_heads

    def estimate_params(self):
        emb = self.vocab_size * self.embed_dim
        # SwiGLU has 3 weight matrices per layer
        per_layer = (
            3 * self.embed_dim ** 2          # Q, K, V (no bias)
            + self.embed_dim ** 2            # W_o
            + 3 * self.embed_dim * self.ff_dim  # SwiGLU: w1, w2, w3
            + 4 * self.embed_dim             # RMSNorm weights (2 per layer)
        )
        layers = self.n_layers * per_layer
        final_norm = 2 * self.embed_dim      # final RMSNorm
        lm_head = 0 if self.tie_weights else self.embed_dim * self.vocab_size
        total = emb + layers + final_norm + lm_head
        return total


# ─────────────────────────────────────────────────────────────
# RMSNorm (replaces LayerNorm — faster, more stable)
# ─────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich, 2019)."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


# ─────────────────────────────────────────────────────────────
# Rotary Position Embedding (RoPE)
# ─────────────────────────────────────────────────────────────

class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (Su et al., 2021).
    Provides position information via rotation of Q/K vectors,
    enabling better length generalization than learned embeddings.
    """
    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.outer(t, self.inv_freq)  # (seq_len, dim/2)
        cos_cached = freqs.cos()  # (seq_len, dim/2)
        sin_cached = freqs.sin()
        self.register_buffer("cos_cached", cos_cached, persistent=False)
        self.register_buffer("sin_cached", sin_cached, persistent=False)

    def forward(self, seq_len: int):
        if seq_len > self.cos_cached.size(0):
            self._build_cache(seq_len)
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotary positional embedding to query and key tensors.

    cos, sin: (seq_len, dim/2) from RotaryEmbedding
    q, k:     (B, n_heads, T, head_dim)

    We expand cos/sin from (1, 1, T, dim/2) to (1, 1, T, dim) via
    repeat_interleave so each pair of dimensions shares the same frequency.
    """
    cos = cos.unsqueeze(0).unsqueeze(0).repeat_interleave(2, dim=-1)  # (1,1,T,dim)
    sin = sin.unsqueeze(0).unsqueeze(0).repeat_interleave(2, dim=-1)  # (1,1,T,dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# ─────────────────────────────────────────────────────────────
# SwiGLU Feed-Forward Network
# ─────────────────────────────────────────────────────────────

class SwiGLU(nn.Module):
    """SwiGLU Feed-Forward (Shazeer, 2020).
    Uses gated linear unit with SiLU activation.
    ff_dim should be set to int(2/3 * 4 * embed_dim) for optimal params.
    """
    def __init__(self, embed_dim: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.w1 = nn.Linear(embed_dim, ff_dim, bias=False)
        self.w2 = nn.Linear(ff_dim, embed_dim, bias=False)
        self.w3 = nn.Linear(embed_dim, ff_dim, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.w2(F.silu(self.w1(x)) * self.w3(x)))


# ─────────────────────────────────────────────────────────────
# Transformer Decoder Layer (v2)
# ─────────────────────────────────────────────────────────────

class TransformerDecoderLayer(nn.Module):
    """Single Transformer decoder layer with:
    - Pre-RMSNorm
    - Flash Attention via F.scaled_dot_product_attention
    - SwiGLU FFN
    - Residual connections
    - Optional dropout
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

        # Self-attention (no bias in Q, K, V — modern standard)
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_o = nn.Linear(embed_dim, embed_dim, bias=False)

        # SwiGLU FFN
        self.ff = SwiGLU(embed_dim, ff_dim, dropout)

        # RMSNorm (Pre-norm)
        self.norm1 = RMSNorm(embed_dim)
        self.norm2 = RMSNorm(embed_dim)

        self.drop_attn = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        rotary_cos: torch.Tensor = None,
        rotary_sin: torch.Tensor = None,
        is_causal: bool = True,
    ) -> torch.Tensor:
        B, T, _ = x.shape

        # ── Masked Multi-Head Self-Attention (Pre-RMSNorm) ──
        residual = x
        x_norm = self.norm1(x)

        Q = self.W_q(x_norm).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(x_norm).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(x_norm).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        if rotary_cos is not None and rotary_sin is not None:
            Q, K = apply_rotary_pos_emb(Q, K, rotary_cos, rotary_sin)

        # Flash Attention via PyTorch SDPA (auto-selects best kernel)
        # is_causal=True automatically creates causal mask (no manual mask needed)
        attn_out = F.scaled_dot_product_attention(
            Q, K, V,
            is_causal=is_causal,
            dropout_p=self.drop_attn.p if self.training else 0.0,
        )

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, -1)
        attn_out = self.W_o(attn_out)
        x = residual + self.drop_attn(attn_out)

        # ── SwiGLU FFN (Pre-RMSNorm) ──
        residual = x
        x = residual + self.ff(self.norm2(x))

        return x


# ─────────────────────────────────────────────────────────────
# Generator Model (v2)
# ─────────────────────────────────────────────────────────────

class InterviewGenerator(nn.Module):
    """
    Decoder-only Transformer for coding interview dialogue (v2).
    
    Modern architecture with RoPE, RMSNorm, SwiGLU, and Flash Attention.
    """

    def __init__(self, config: GeneratorConfig):
        super().__init__()
        self.config = config
        self.pad_id = config.pad_id

        # Embeddings
        self.token_emb = nn.Embedding(config.vocab_size, config.embed_dim, padding_idx=config.pad_id)
        self.emb_dropout = nn.Dropout(config.dropout)

        # RoPE
        self.rope = RotaryEmbedding(config.head_dim, config.max_len)

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

        # Final RMSNorm
        self.norm = RMSNorm(config.embed_dim)

        # Language model head
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)

        # Weight tying
        if config.tie_weights:
            self.lm_head.weight = self.token_emb.weight

        # Gradient checkpointing
        self.gradient_checkpointing = config.gradient_checkpointing

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """GPT-2 style scaled initialization."""
        for name, p in self.named_parameters():
            if p.dim() > 1:
                # Embedding layers
                if "token_emb" in name:
                    nn.init.normal_(p, mean=0.0, std=0.02)
                # Output projections and FFN gate projections
                elif "W_o" in name or "w2" in name or "w3" in name:
                    nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layers))
                # Query projection (scale down for stability)
                elif "W_q" in name:
                    nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layers))
                # Key projection
                elif "W_k" in name:
                    nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layers))
                # Value projection
                elif "W_v" in name:
                    nn.init.normal_(p, mean=0.0, std=0.02)
                # SwiGLU w1
                elif "w1" in name:
                    nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layers))
                else:
                    nn.init.normal_(p, mean=0.0, std=0.02)
            # RMSNorm weights
            if "weight" in name and "norm" in name:
                nn.init.ones_(p)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor = None,
        return_logits: bool = True,
        label_smoothing: float = None,
    ) -> dict:
        B, T = input_ids.shape

        # Embed
        x = self.token_emb(input_ids)
        x = self.emb_dropout(x)

        # RoPE cache
        cos, sin = self.rope(T)

        # Transformer layers
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    layer, x, cos, sin, True,
                    use_reentrant=False,
                )
            else:
                x = layer(x, cos, sin, is_causal=True)

        x = self.norm(x)
        logits = self.lm_head(x)

        result = {}
        if return_logits:
            result["logits"] = logits

        # Compute loss if labels provided
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            ls = label_smoothing if label_smoothing is not None else self.config.label_smoothing

            if ls > 0:
                # Label smoothing cross entropy
                vocab_size = shift_logits.size(-1)
                log_probs = F.log_softmax(shift_logits.view(-1, vocab_size), dim=-1)
                nll_loss = F.nll_loss(
                    log_probs, shift_labels.view(-1),
                    ignore_index=self.pad_id, reduction="mean",
                )
                smooth_loss = -log_probs.mean(dim=-1)
                mask = (shift_labels.view(-1) != self.pad_id).float()
                smooth_loss = (smooth_loss * mask).sum() / mask.sum()
                loss = (1.0 - ls) * nll_loss + ls * smooth_loss
            else:
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
        """Generate text autoregressively with per-sequence EOS tracking."""
        self.eval()
        generated = prompt_ids.clone()
        B = generated.size(0)

        # Per-sequence finished tracking (fixes v1 EOS bug)
        finished = torch.zeros(B, dtype=torch.bool, device=generated.device)

        for _ in range(max_new_tokens):
            # Truncate to max_len if needed
            input_ids = generated[:, -self.config.max_len:]

            result = self.forward(input_ids)
            logits = result["logits"][:, -1, :]

            # Temperature
            logits = logits / temperature

            # Top-k filtering
            if top_k > 0:
                top_k_vals, _ = torch.topk(logits, top_k)
                threshold = top_k_vals[:, -1, None]
                logits = logits.masked_fill(logits < threshold, float("-inf"))

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_mask = cumulative_probs > top_p
                sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
                sorted_mask[..., 0] = 0
                # Unsort and apply mask
                mask = torch.zeros_like(logits).scatter(1, sorted_indices, sorted_mask.float())
                logits = logits.masked_fill(mask.bool(), float("-inf"))

            # Sample (only from non-finished sequences)
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Mask finished sequences: force PAD token
            next_token[finished] = self.pad_id

            generated = torch.cat([generated, next_token], dim=1)

            # Update finished tracking
            if eos_token_id is not None:
                finished = finished | (next_token.squeeze(-1) == eos_token_id)
                if finished.all():
                    break

            if generated.size(1) >= self.config.max_len:
                break

        return generated

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def num_params_millions(self) -> float:
        return self.num_params / 1_000_000

    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing (saves VRAM, slower backward pass)."""
        self.gradient_checkpointing = True

    def disable_gradient_checkpointing(self):
        """Disable gradient checkpointing (faster, uses more VRAM)."""
        self.gradient_checkpointing = False


# ─────────────────────────────────────────────────────────────
# Factory functions
# ─────────────────────────────────────────────────────────────

def create_small_model(vocab_size=16000) -> InterviewGenerator:
    """~12M parameter model for CPU/GTX 1650."""
    config = GeneratorConfig(
        vocab_size=vocab_size,
        embed_dim=256,
        n_heads=4,
        n_layers=12,
        ff_dim=688,
        max_len=2048,
    )
    model = InterviewGenerator(config)
    print(f"Small model: {model.num_params_millions:.1f}M params")
    return model


def create_medium_model(vocab_size=16000) -> InterviewGenerator:
    """~45M parameter model for T4."""
    config = GeneratorConfig(
        vocab_size=vocab_size,
        embed_dim=512,
        n_heads=8,
        n_layers=12,
        ff_dim=1408,
        max_len=2048,
    )
    model = InterviewGenerator(config)
    print(f"Medium model: {model.num_params_millions:.1f}M params")
    return model


def create_large_model(vocab_size=16000) -> InterviewGenerator:
    """~120M parameter model for 2xT4 / P100."""
    config = GeneratorConfig(
        vocab_size=vocab_size,
        embed_dim=768,
        n_heads=12,
        n_layers=16,
        ff_dim=2048,
        max_len=2048,
    )
    model = InterviewGenerator(config)
    print(f"Large model: {model.num_params_millions:.1f}M params")
    return model


if __name__ == "__main__":
    print("=" * 60)
    print("InterviewGenerator v2 Architecture Check")
    print("=" * 60)

    for name, factory in [
        ("Small (~12M)", create_small_model),
        ("Medium (~45M)", create_medium_model),
        ("Large (~120M)", create_large_model),
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

        # Loss test
        labels = torch.randint(0, 16000, (2, 128))
        result = model(dummy, labels=labels, return_logits=False)
        print(f"  Loss: {result['loss'].item():.4f}")

        # Generation test
        prompt = torch.randint(0, 16000, (1, 10))
        generated = model.generate(prompt, max_new_tokens=20)
        print(f"  Prompt: {prompt.shape} -> Generated: {generated.shape}")

        del model
        torch.cuda.empty_cache()
