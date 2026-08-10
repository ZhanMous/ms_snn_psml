"""Disturbance-Preserving Mixer components used by FASD-Net.

The retained blocks implement the selected post-ablation architecture:
1. PatchEmbed: Conv1d patch embedding (kernel=patch_size, stride=patch_size//2)
   plus a learnable positional embedding ``[1, M, emb_dim]``.
2. AdaptiveSpectralBlock (ASB): ``torch.fft.rfft`` along the patch-token
   dimension, learnable complex *global* spectral weights, and an optional
   adaptive high-frequency mask driven by normalized FFT energy and a
   learnable threshold (TSLANet-style straight-through for the hard modes).
3. InteractiveConvolutionBlock (ICB): Conv1d k=1 and k=3 branches with
   element-wise cross interaction and a Conv1d k=1 projection.
4. DPMixerLayer: one residual spectral/interactive-convolution mixer.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════


def compute_num_tokens(seq_len: int, patch_size: int) -> int:
    """Number of patch tokens for Conv1d(k=patch, stride=patch//2, padding=0)."""
    stride = patch_size // 2
    return (seq_len - patch_size) // stride + 1


# ═══════════════════════════════════════════════════════════════════════════
# Patch Embedding
# ═══════════════════════════════════════════════════════════════════════════


class PatchEmbed(nn.Module):
    """Conv1d patch embedding: ``[B, C, T] -> [B, M, emb_dim]``.

    ``kernel_size = patch_size``, ``stride = patch_size // 2`` (overlapping,
    no padding, faithful to the spec). Adds a learnable positional embedding
    of shape ``[1, M, emb_dim]``.
    """

    def __init__(
        self, in_channels: int, emb_dim: int, patch_size: int, num_tokens: int
    ):
        super().__init__()
        self.patch_size = patch_size
        self.stride = patch_size // 2
        self.num_tokens = num_tokens
        self.proj = nn.Conv1d(
            in_channels,
            emb_dim,
            kernel_size=patch_size,
            stride=self.stride,
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, emb_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        """x: [B, C, T] -> [B, M, emb_dim]. Sequences shorter than the kernel
        are right-padded with zeros so prefix evaluation at K < patch_size
        still yields at least one token (leakage-safe: zeros, not future data)."""
        if x.shape[-1] < self.patch_size:
            pad = self.patch_size - x.shape[-1]
            x = torch.nn.functional.pad(x, (0, pad))
        h = self.proj(x)  # [B, emb_dim, M]
        h = h.transpose(1, 2)  # [B, M, emb_dim]
        M = h.shape[1]
        if M == self.pos_embed.shape[1]:
            return h + self.pos_embed
        # Safety net for unexpected sequence lengths (keep differentiable).
        if M < self.pos_embed.shape[1]:
            pe = self.pos_embed[:, :M, :]
        else:
            pe = torch.nn.functional.pad(
                self.pos_embed, (0, 0, 0, M - self.pos_embed.shape[1])
            )
        return h + pe


# ═══════════════════════════════════════════════════════════════════════════
# Adaptive Spectral Block (ASB)
# ═══════════════════════════════════════════════════════════════════════════


class AdaptiveSpectralBlock(nn.Module):
    """FFT-based spectral block with learnable complex weights and adaptive mask.

    Steps:
      1. ``rfft`` along the patch-token dimension (``norm='ortho'``).
      2. Multiply by a learnable *global* complex weight ``[F, C]``.
      3. Optionally apply an adaptive mask computed from the normalized FFT
         energy per frequency bin and a learnable threshold.
      4. ``irfft`` back to the time domain.

    ``mask_mode``:
      - ``no_mask``                  : complex weighting only.
      - ``original_high_freq_mask``  : TSLANet-style adaptive low-pass. The
        cumulative normalized energy over frequency bins is compared with a
        learnable threshold ``tau``; high-frequency bins whose cumulative
        energy exceeds ``tau`` are hard-masked via straight-through.
      - ``high_energy_mask``         : per-bin hard straight-through mask
        keeping bins with normalized energy above ``tau``.
      - ``low_energy_suppression``   : soft energy-proportional attenuation
        ``p ** alpha`` (``alpha`` learnable), suppressing low-energy bins.
      - ``learnable_soft_mask``      : soft (sigmoid) replacement of the
        original high-frequency mask, fully differentiable (no STE).
    """

    MASK_MODES = (
        "no_mask",
        "original_high_freq_mask",
        "high_energy_mask",
        "low_energy_suppression",
        "learnable_soft_mask",
    )

    def __init__(
        self,
        channels: int,
        num_tokens: int,
        mask_mode: str = "no_mask",
        temp: float = 0.1,
    ):
        super().__init__()
        assert mask_mode in self.MASK_MODES, f"unknown mask_mode {mask_mode}"
        self.mask_mode = mask_mode
        self.temp = temp
        self.num_tokens = num_tokens
        self.freqs = num_tokens // 2 + 1

        # Learnable complex global spectral weights [F, C], init ~ identity.
        w = torch.zeros(self.freqs, channels, 2)
        w[..., 0] = 1.0 + torch.randn(self.freqs, channels) * 0.02
        w[..., 1] = torch.randn(self.freqs, channels) * 0.02
        self.complex_weight = nn.Parameter(w)

        if mask_mode != "no_mask":
            # Learnable threshold; tau = sigmoid(threshold) in (0, 1).
            self.threshold = nn.Parameter(torch.tensor(0.0))

    def _energy_per_bin(self, x_f: Tensor) -> Tensor:
        """Normalized per-frequency-bin energy averaged over channels. [B, F]"""
        energy = x_f.abs() ** 2  # [B, F, C]
        e_bin = energy.mean(dim=2)  # [B, F]
        return e_bin

    def _mask(self, x_f: Tensor) -> Tensor | None:
        """Build the adaptive mask [B, F] from the raw input spectrum."""
        if self.mask_mode == "no_mask":
            return None
        e_bin = self._energy_per_bin(x_f)
        tau = torch.sigmoid(self.threshold)

        if self.mask_mode == "original_high_freq_mask":
            # Adaptive low-pass: keep low-freq band capturing `tau` of energy.
            p = e_bin / (e_bin.sum(dim=1, keepdim=True) + 1e-8)  # [B, F]
            cum = torch.cumsum(p, dim=1)  # [B, F]
            soft = torch.sigmoid((tau - cum) / self.temp)  # ~1 keep low
            hard = (cum <= tau).to(x_f.dtype)
            return hard.detach() + (soft - soft.detach())  # STE

        if self.mask_mode == "high_energy_mask":
            p = e_bin / (e_bin.amax(dim=1, keepdim=True) + 1e-8)  # [B, F] in [0,1]
            soft = torch.sigmoid((p - tau) / self.temp)
            hard = (p > tau).to(x_f.dtype)
            return hard.detach() + (soft - soft.detach())  # STE

        if self.mask_mode == "low_energy_suppression":
            p = e_bin / (e_bin.amax(dim=1, keepdim=True) + 1e-8)  # [B, F] in [0,1]
            alpha = 1.0 + torch.sigmoid(self.threshold) * 4.0  # [1, 5]
            return p.clamp(min=1e-6).pow(alpha)  # soft

        if self.mask_mode == "learnable_soft_mask":
            p = e_bin / (e_bin.sum(dim=1, keepdim=True) + 1e-8)
            cum = torch.cumsum(p, dim=1)
            return torch.sigmoid((tau - cum) / self.temp)  # soft, no STE

        return None

    def forward(self, x: Tensor) -> Tensor:
        """x: [B, M, C] -> [B, M, C]. Length-flexible: the learned complex
        weights (sized for the training token count) are sliced/padded to the
        runtime frequency bin count so prefix-length inputs work."""
        M = x.shape[1]
        x_f = torch.fft.rfft(x, dim=1, norm="ortho")  # [B, F', C]
        mask = self._mask(x_f) if self.mask_mode != "no_mask" else None
        weight = torch.view_as_complex(self.complex_weight)  # [F_train, C]
        F_rt = x_f.shape[1]
        if F_rt == weight.shape[0]:
            w = weight
        elif F_rt < weight.shape[0]:
            w = weight[:F_rt, :]  # use low-freq weights
        else:  # runtime longer than training (rare): pad with identity
            pad = torch.ones(
                F_rt - weight.shape[0],
                weight.shape[1],
                dtype=weight.dtype,
                device=weight.device,
            )
            w = torch.cat([weight, pad], dim=0)
        x_f = x_f * w
        if mask is not None:
            x_f = x_f * mask.unsqueeze(2)  # [B, F', C]
        return torch.fft.irfft(x_f, n=M, dim=1, norm="ortho")  # [B, M, C]

    @property
    def threshold_value(self) -> float | None:
        if self.mask_mode == "no_mask":
            return None
        return float(torch.sigmoid(self.threshold).detach().item())


# ═══════════════════════════════════════════════════════════════════════════
# Interactive Convolution Block (ICB)
# ═══════════════════════════════════════════════════════════════════════════


class InteractiveConvBlock(nn.Module):
    """ICB: Conv1d k=1 and k=3 branches, cross interaction by element-wise
    multiplication, then Conv1d k=1 projection.

    No internal residual/norm: the enclosing DPMixerLayer supplies the
    residual connection and the pre-norm ``LN2``.
    """

    def __init__(self, channels: int, hidden: int | None = None):
        super().__init__()
        hidden = hidden or channels
        self.branch1 = nn.Conv1d(channels, hidden, kernel_size=1)
        self.branch2 = nn.Conv1d(channels, hidden, kernel_size=3, padding=1)
        self.proj = nn.Conv1d(hidden, channels, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        """x: [B, M, C] -> [B, M, C]."""
        h = x.transpose(1, 2)  # [B, C, M]
        b1 = self.branch1(h)  # [B, hidden, M]
        b2 = self.branch2(h)  # [B, hidden, M]
        out = self.proj(b1 * b2)  # cross interaction + projection
        out = self.act(out)
        return out.transpose(1, 2)  # [B, M, C]


# ═══════════════════════════════════════════════════════════════════════════
# DropPath (stochastic depth)
# ═══════════════════════════════════════════════════════════════════════════


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.bernoulli(torch.full(shape, keep, device=x.device, dtype=x.dtype))
        return x * mask / keep


# ═══════════════════════════════════════════════════════════════════════════
# DPMixer layer
# ═══════════════════════════════════════════════════════════════════════════


class DPMixerLayer(nn.Module):
    """Single residual DPMixer layer.

    With both blocks: ``x = x + DropPath(ICB(LN2(ASB(LN1(x)))))``.
    With only ASB   : ``x = x + DropPath(ASB(LN1(x)))``.
    With only ICB   : ``x = x + DropPath(ICB(LN2(x)))``.
    """

    def __init__(
        self,
        channels: int,
        num_tokens: int,
        *,
        use_asb: bool = True,
        use_icb: bool = True,
        asb_mask: str = "no_mask",
        hidden: int | None = None,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.use_asb = use_asb
        self.use_icb = use_icb
        if use_asb:
            self.ln1 = nn.LayerNorm(channels)
            self.asb = AdaptiveSpectralBlock(channels, num_tokens, mask_mode=asb_mask)
        if use_icb:
            self.ln2 = nn.LayerNorm(channels)
            self.icb = InteractiveConvBlock(channels, hidden)
        self.drop_path = DropPath(drop_path)

    def forward(self, x: Tensor) -> Tensor:
        h = x
        if self.use_asb:
            h = self.asb(self.ln1(h))
        if self.use_icb:
            h = self.icb(self.ln2(h))
        return x + self.drop_path(h)


# ═══════════════════════════════════════════════════════════════════════════
# TSLANet Lite (full model)
# ═══════════════════════════════════════════════════════════════════════════
