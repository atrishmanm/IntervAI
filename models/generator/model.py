"""
models/generator/model.py — Research-Grade
==========================================
Decoder-only Transformer (GPT-style) for coding interview dialogue.

Architecture (v2 ~120M params):
  - RMSNorm (Zhang & Sennrich, 2019) — faster, more stable than LayerNorm
  - Rotary Position Embedding (RoPE, Su et al. 2021) — better length generalization
  - SwiGLU Feed-Forward (Shazeer, 2020) — gated FFN, superior to ReLU/GELU
  - Flash Attention via F.scaled_dot_product_attention — memory-efficient
  - Pre-norm residual connections
  - GPT-2 style weight initialization
  - Label smoothing cross-entropy loss
  - Gradient checkpointing for memory-constrained GPUs
  - Per-sequence EOS tracking in generate()

Target sizes:
  Small  (~14M):  embed_dim=256, n_layers=12, ff_dim=688,  n_heads=4
  Medium (~47M):  embed_dim=512, n_layers=12, ff_dim=1408, n_heads=8
  Large  (~126M): embed_dim=768, n_layers=16, ff_dim=2048, n_heads=12
"""

import math
from dataclasses import dataclass, field
from typing import Optional

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
    dropout: float = 0.05
    pad_id: int = 0
    tie_weights: bool = True
    label_smoothing: float = 0.03
    gradient_checkpointing: bool = False

    @property
    def head_dim(self):
        return self.embed_dim // self.n_heads

    def estimate_params(self):
        emb = self.vocab_size * self.embed_dim
        pos = self.max_len * self.embed_dim
        per_layer = (
            4 * self.embed_dim ** 2
            + 3 * self.embed_dim * self.ff_dim
            + 4 * self.embed_dim
        )
        norms = 4 * self.embed_dim
        lm_head = self.embed_dim * self.vocab_size if not self.tie_weights else 0
        total = emb + pos + self.n_layers * per_layer + norms + lm_head
        return total


# ─────────────────────────────────────────────────────────────
# RMSNorm (Zhang & Sennrich, 2019)
# ─────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
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
    """Rotary Position Embedding (Su et al., 2021)."""
    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.outer(t, self.inv_freq)
        cos_cached = freqs.cos()
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

    Expands cos/sin via repeat_interleave so each pair of dimensions
    shares the same frequency.
    """
    cos = cos.unsqueeze(0).unsqueeze(0).repeat_interleave(2, dim=-1)
    sin = sin.unsqueeze(0).unsqueeze(0).repeat_interleave(2, dim=-1)
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
# Transformer Decoder Layer (Research-Grade)
# ─────────────────────────────────────────────────────────────

class TransformerDecoderLayer(nn.Module):
    """Single Transformer decoder layer with:
    - Pre-RMSNorm
    - Flash Attention via F.scaled_dot_product_attention
    - SwiGLU FFN
    - Residual connections
    - Optional dropout
    """
    def __init__(self, embed_dim: int, n_heads: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % n_heads == 0, "embed_dim must be divisible by n_heads"
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads

        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_o = nn.Linear(embed_dim, embed_dim, bias=False)

        self.ff = SwiGLU(embed_dim, ff_dim, dropout)
        self.norm1 = RMSNorm(embed_dim)
        self.norm2 = RMSNorm(embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, rotary_cos, rotary_sin, is_causal: bool = True):
        B, T, D = x.shape

        # Pre-RMSNorm
        h = self.norm1(x)

        # Self-attention
        Q = self.W_q(h).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(h).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(h).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        Q, K = apply_rotary_pos_emb(Q, K, rotary_cos, rotary_sin)

        attn_out = F.scaled_dot_product_attention(
            Q, K, V, is_causal=is_causal, dropout_p=self.drop.p if self.training else 0.0
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, D)
        x = x + self.drop(self.W_o(attn_out))

        # SwiGLU FFN
        x = x + self.ff(self.norm2(x))

        return x


# ─────────────────────────────────────────────────────────────
# InterviewGenerator (Research-Grade)
# ─────────────────────────────────────────────────────────────

class InterviewGenerator(nn.Module):
    """Decoder-only Transformer for coding interview dialogue.

    Research-grade features:
    - RMSNorm for stable training
    - RoPE for length generalization
    - SwiGLU FFN for better parameter efficiency
    - Flash Attention via SDPA for memory efficiency
    - Label smoothing cross-entropy
    - Gradient checkpointing for small GPUs
    - Per-sequence EOS tracking in generate()
    - GPT-2 style weight initialization
    """
    def __init__(self, config: GeneratorConfig):
        super().__init__()
        self.config = config
        self.pad_id = config.pad_id

        # Token embedding (no learned position — RoPE handles positions)
        self.token_emb = nn.Embedding(config.vocab_size, config.embed_dim, padding_idx=config.pad_id)
        self.emb_dropout = nn.Dropout(config.dropout)

        # Rotary embedding
        self.rope = RotaryEmbedding(config.head_dim, max_seq_len=config.max_len)

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(
                config.embed_dim, config.n_heads, config.ff_dim, config.dropout
            )
            for _ in range(config.n_layers)
        ])

        # Final norm + LM head
        self.norm = RMSNorm(config.embed_dim)
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)

        # Weight tying
        if config.tie_weights:
            self.lm_head.weight = self.token_emb.weight

        # Gradient checkpointing flag
        self.gradient_checkpointing = config.gradient_checkpointing

        # Dynamic dropout state
        self._dropout_step = 0
        self._base_dropout = config.dropout

        # Initialize weights (GPT-2 style)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                nn.init.zeros_(module.weight[module.padding_idx])
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

        # Special initialization for v2 architecture
        for name, p in self.named_parameters():
            if p.dim() < 2:
                continue
            if "token_emb" in name:
                nn.init.normal_(p, mean=0.0, std=0.02)
            elif "W_o" in name or "w2" in name or "w3" in name:
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layers))
            elif "W_q" in name or "W_k" in name:
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layers))
            elif "W_v" in name:
                nn.init.normal_(p, mean=0.0, std=0.02)
            elif "w1" in name:
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layers))
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

        x = self.token_emb(input_ids)
        x = self.emb_dropout(x)

        cos, sin = self.rope(T)

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

        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            ls = label_smoothing if label_smoothing is not None else self.config.label_smoothing

            if ls > 0:
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
        finished = torch.zeros(B, dtype=torch.bool, device=generated.device)

        for _ in range(max_new_tokens):
            input_ids = generated[:, -self.config.max_len:]
            result = self.forward(input_ids)
            logits = result["logits"][:, -1, :]
            logits = logits / temperature

            if top_k > 0:
                top_k_vals, _ = torch.topk(logits, top_k)
                threshold = top_k_vals[:, -1, None]
                logits = logits.masked_fill(logits < threshold, float("-inf"))

            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_mask = cumulative_probs > top_p
                sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
                sorted_mask[..., 0] = 0
                mask = torch.zeros_like(logits).scatter(1, sorted_indices, sorted_mask.float())
                logits = logits.masked_fill(mask.bool(), float("-inf"))

            # temperature=0 means greedy (argmax), otherwise multinomial sampling
            if temperature <= 0.0:
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            next_token[finished] = self.pad_id

            generated = torch.cat([generated, next_token], dim=1)

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

    def update_dropout(self, step: int, total_steps: int):
        """Dynamic dropout: higher early (regularization), lower late (convergence)."""
        self._dropout_step = step
        progress = min(step / max(total_steps, 1), 1.0)
        # Linear decay from base_dropout to 0.02 over training
        new_dropout = self._base_dropout * (1.0 - progress) + 0.02 * progress
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.p = new_dropout

    def get_dropout_rate(self) -> float:
        """Get current dropout rate."""
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                return module.p
        return 0.0

    def get_weight_norms(self) -> dict:
        """Get L2 norms of all weight matrices (for monitoring)."""
        norms = {}
        for name, param in self.named_parameters():
            if param.requires_grad and param.dim() >= 2:
                norms[name] = param.data.norm(2).item()
        return norms

    def get_gradient_norms(self) -> dict:
        """Get L2 norms of all gradients (for monitoring)."""
        norms = {}
        for name, param in self.named_parameters():
            if param.requires_grad and param.grad is not None:
                norms[name] = param.grad.data.norm(2).item()
        return norms

    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False


# ─────────────────────────────────────────────────────────────
# Factory functions
# ─────────────────────────────────────────────────────────────

def create_small_model(vocab_size=16000) -> InterviewGenerator:
    """~14M parameter model for CPU/GTX 1650."""
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
    """~47M parameter model for T4."""
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
    """~126M parameter model for 2xT4 / P100."""
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
        ("Small (~14M)", create_small_model),
        ("Medium (~47M)", create_medium_model),
        ("Large (~126M)", create_large_model),
    ]:
        print(f"\n{name}:")
        model = factory()
        config = model.config
        print(f"  embed_dim={config.embed_dim}, n_layers={config.n_layers}, "
              f"ff_dim={config.ff_dim}, n_heads={config.n_heads}")
        print(f"  Estimated params: {config.estimate_params():,}")

        dummy = torch.randint(0, 16000, (2, 128))
        result = model(dummy)
        print(f"  Input: {dummy.shape} -> Logits: {result['logits'].shape}")

        labels = torch.randint(0, 16000, (2, 128))
        result = model(dummy, labels=labels, return_logits=False)
        print(f"  Loss: {result['loss'].item():.4f}")

        prompt = torch.randint(0, 16000, (1, 10))
        generated = model.generate(prompt, max_new_tokens=20)
        print(f"  Prompt: {prompt.shape} -> Generated: {generated.shape}")
