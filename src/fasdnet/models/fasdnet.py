"""FASD-Net: Fast-Alarm and Slow-Diagnosis SNN-ANN fusion model.

This module hosts the canonical FASD-Net model for PSML disturbance
diagnosis: ``FASDNET_CONFIG`` = lightweight shared ``PatchEmbed`` +
compact ``ICB`` (the minimal effective DPMixer structure from the architecture
ablation) + a patch-level ``SNNAlarmHead`` spiking event branch. A high-accuracy
``UPPER_BOUND_CONFIG`` (large patch + ASB+ICB stack) is kept as a reference
upper bound, not the main model.

Architecture:
  - Shared or dual ``PatchEmbed`` producing patch tokens ``Z0``.
  - Classification path (ANN, DPMixer): stack of ``DPMixerLayer`` blocks.
  - Localization path (ANN, DPMixer): stack of ``DPMixerLayer`` blocks.
  - Optional patch-level SNN branch (``SNNAlarmHead`` or ``SpikeTCNAlarmHead``) that
    operates on ``Z0`` along the **patch-token time axis** (not post-pooling).
    The SNN produces:
      a) full-window spiking cls logits ``cls_logits_spk`` (fused with the ANN
         cls path as ``cls_logits = cls_logits_ann + beta * cls_logits_spk``),
      b) per-token alarm score trajectory ``alarm_score`` [B, M],
      c) scalar predicted alarm timestep for MAE.
  - Localization always stays on the ANN patch/ASB path.

The model is length-flexible so prefix classification can recompute patches from
``x[:, :K, :]`` and run both the ANN cls path and the SNN branch on the shorter
token sequence (no future information is used).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .dpmixer import (
    DPMixerLayer,
    PatchEmbed,
    compute_num_tokens,
)

try:  # reuse the project's surrogate-gradient spike operator
    from .snn.lif import SurrogateSpike
except Exception:  # pragma: no cover - fallback if src not on path

    class SurrogateSpike(torch.autograd.Function):
        @staticmethod
        def forward(ctx, u, width=1.0):
            ctx.save_for_backward(u)
            ctx.width = width
            return (u >= 0).to(u.dtype)

        @staticmethod
        def backward(ctx, g):
            (u,) = ctx.saved_tensors
            d = torch.clamp(1.0 - u.abs() / ctx.width, min=0.0) / ctx.width
            return g * d, None


# ═══════════════════════════════════════════════════════════════════════════
# Patch-level SNN branches
# ═══════════════════════════════════════════════════════════════════════════


class SNNAlarmHead(nn.Module):
    """PLIF-inspired learnable-decay LIF over the patch-token dimension.

    Input ``Z0`` ``[B, M, emb_dim]`` is projected to input currents and
    integrated token-by-token with a learnable membrane decay
    ``alpha = sigmoid(lambda)``. Emits binary spikes via a straight-through
    surrogate gradient.
    """

    def __init__(
        self,
        emb_dim: int,
        hidden: int,
        num_classes: int,
        threshold: float = 0.5,
        surrogate_width: float = 1.0,
        input_gain: float = 2.0,
    ):
        super().__init__()
        self.hidden = hidden
        self.threshold = threshold
        self.surrogate_width = surrogate_width
        self.input_proj = nn.Linear(emb_dim, hidden)
        self.input_gain = input_gain
        self.decay_logit = nn.Parameter(torch.tensor(0.0))  # alpha=sigmoid(0)=0.5
        # readouts
        self.cls_pool_head = nn.Linear(hidden, num_classes)  # full-window cls
        self.alarm_token_head = nn.Linear(hidden, 1)  # per-token alarm

    def _spike(self, u: Tensor) -> Tensor:
        return SurrogateSpike.apply(u - self.threshold, self.surrogate_width)

    def forward(self, z0: Tensor) -> dict:
        """z0: [B, M, emb_dim]. Returns spikes [B, M, hidden] + readouts."""
        B, M, _ = z0.shape
        cur = self.input_gain * self.input_proj(z0)  # [B, M, hidden]
        alpha = torch.sigmoid(self.decay_logit)
        mem = torch.zeros(B, self.hidden, device=z0.device, dtype=z0.dtype)
        spikes = []
        for m in range(M):
            mem = alpha * mem + cur[:, m, :]
            s = self._spike(mem)
            mem = mem - s.detach() * self.threshold
            spikes.append(s)
        S = torch.stack(spikes, dim=1)  # [B, M, hidden]
        return self._readout(S, z0)

    def _readout(self, S: Tensor, z0: Tensor) -> dict:
        snn_feat = S.mean(dim=1)  # [B, hidden]
        cls_spk = self.cls_pool_head(snn_feat)  # [B, num_classes]
        alarm_logit = self.alarm_token_head(S).squeeze(-1)  # [B, M]
        return {
            "spikes": S,
            "cls_spk": cls_spk,
            "snn_feat": snn_feat,
            "alarm_logit": alarm_logit,
        }


class SpikeTCNAlarmHead(nn.Module):
    """Spike-TCN over patch tokens: stacked causal dilated convs with spiking
    activations. Causal left-padding guarantees no future-token leakage so
    prefix evaluation is valid."""

    def __init__(
        self,
        emb_dim: int,
        hidden: int,
        num_classes: int,
        kernel_size: int = 3,
        dilations=(1, 2, 4),
        threshold: float = 0.5,
        surrogate_width: float = 1.0,
        input_gain: float = 2.0,
    ):
        super().__init__()
        self.hidden = hidden
        self.threshold = threshold
        self.surrogate_width = surrogate_width
        self.input_proj = nn.Linear(emb_dim, hidden)
        self.input_gain = input_gain
        self.dilations = dilations
        self.kernel_size = kernel_size
        self.conv_layers = nn.ModuleList(
            [
                nn.Conv1d(hidden, hidden, kernel_size=kernel_size, dilation=d)
                for d in dilations
            ]
        )
        # Init conv biases to threshold so the membrane starts at threshold and
        # neurons fire when the weighted spike input is positive (avoids a dead
        # all-zero spike network at init).
        with torch.no_grad():
            for conv in self.conv_layers:
                conv.bias.data.fill_(self.threshold)
        self.cls_pool_head = nn.Linear(hidden, num_classes)
        self.alarm_token_head = nn.Linear(hidden, 1)

    def _spike(self, u: Tensor) -> Tensor:
        return SurrogateSpike.apply(u - self.threshold, self.surrogate_width)

    def forward(self, z0: Tensor) -> dict:
        h = (self.input_gain * self.input_proj(z0)).transpose(1, 2)  # [B, hidden, M]
        for conv in self.conv_layers:
            d = conv.dilation[0]
            pad = (self.kernel_size - 1) * d  # causal: left-pad only
            inp = F.pad(h, (pad, 0))
            mem = conv(inp)
            h = self._spike(mem)  # spiking activation
        S = h.transpose(1, 2)  # [B, M, hidden]
        return self._readout(S, z0)

    def _readout(self, S: Tensor, z0: Tensor) -> dict:
        snn_feat = S.mean(dim=1)
        cls_spk = self.cls_pool_head(snn_feat)
        alarm_logit = self.alarm_token_head(S).squeeze(-1)
        return {
            "spikes": S,
            "cls_spk": cls_spk,
            "snn_feat": snn_feat,
            "alarm_logit": alarm_logit,
        }


class DenseAlarmHead(nn.Module):
    """Dense (non-spiking) per-token alarm head -- the ANN alarm baseline.

    A single causal 1D convolution over patch tokens (kernel 3) provides
    *dense* temporal context without spikes. It reads the backbone
    representation but its input is detached so the alarm loss never
    backpropagates into the shared backbone -- the cls/loc representations are
    therefore identical to the ANN-only row (clean control variable), and the
    alarm quality reflects only the dense head's temporal modeling.

    Compared to the spiking PLIF branch this is the "dense alarm baseline": it
    has a single layer of temporal context but no membrane integration /
    thresholded events, so it should be clearly worse at alarm timing than the
    SNN head while still beating a mean prior. ``cls_spk`` is zeros
    (alarm-only branch) and ``spikes`` is ``None`` (no spikes).
    """

    def __init__(self, emb_dim: int, hidden: int, num_classes: int, **_):
        super().__init__()
        self.num_classes = num_classes
        self.proj = nn.Linear(emb_dim, hidden)
        self.conv = nn.Conv1d(
            hidden, hidden, kernel_size=3, padding=2
        )  # causal via left-pad slice
        self.act = nn.GELU()
        self.alarm_token_head = nn.Linear(hidden, 1)

    def forward(self, z0: Tensor) -> dict:
        h = self.act(self.proj(z0.detach())).transpose(1, 2)  # [B, hidden, M]
        h = self.act(self.conv(h)[:, :, : z0.shape[1]])  # causal: drop right pad
        h = h.transpose(1, 2)  # [B, M, hidden]
        alarm_logit = self.alarm_token_head(h).squeeze(-1)  # [B, M]
        cls_spk = z0.new_zeros(z0.shape[0], self.num_classes)
        return {
            "spikes": None,
            "cls_spk": cls_spk,
            "snn_feat": h.mean(dim=1),
            "alarm_logit": alarm_logit,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Task-specific fast/slow network
# ═══════════════════════════════════════════════════════════════════════════


class FASDNet(nn.Module):
    """Task-specific readout with separate cls/loc ANN paths and an optional
    patch-level SNN branch fused into the classification logits.

    cls_path / loc_path configure which DPMixer blocks each path uses:
      ``"icb"``           -> use_asb=False, use_icb=True
      ``"asb_nomask"``    -> use_asb=True,  use_icb=False, asb_mask="no_mask"
      ``"asb_icb"``       -> use_asb=True,  use_icb=True,  asb_mask="no_mask"
      ``"patch_only"``    -> depth=0 (no layers)
    """

    PATH_CFG = {
        "icb": dict(use_asb=False, use_icb=True, asb_mask="no_mask"),
        "asb_nomask": dict(use_asb=True, use_icb=False, asb_mask="no_mask"),
        "asb_icb": dict(use_asb=True, use_icb=True, asb_mask="no_mask"),
        "patch_only": dict(use_asb=False, use_icb=False, asb_mask="no_mask"),
    }

    def __init__(
        self,
        *,
        in_channels: int = 91,
        seq_len: int = 960,
        # cls path
        cls_path: str = "icb",
        cls_patch_size: int = 32,
        cls_emb_dim: int = 64,
        cls_depth: int = 2,
        # loc path
        loc_path: str = "asb_nomask",
        loc_patch_size: int = 64,
        loc_emb_dim: int = 96,
        loc_depth: int = 2,
        # snn branch
        snn_branch: str = "none",  # "none" | "plif" | "spiketcn"
        snn_hidden: int = 64,
        beta: float = 1.0,  # fusion weight for cls_ann + beta*cls_spk
        learnable_beta: bool = True,
        alarm_loss: bool = True,
        # heads
        num_classes: int = 5,
        num_locations: int = 276,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.seq_len = seq_len
        self.cls_path = cls_path
        self.loc_path = loc_path
        self.snn_branch = snn_branch
        self.alarm_loss = alarm_loss
        self.num_classes = num_classes
        self.num_locations = num_locations

        self.cls_stride = cls_patch_size // 2
        self.loc_stride = loc_patch_size // 2
        self.cls_patch_size = cls_patch_size
        self.loc_patch_size = loc_patch_size

        # Shared or dual patch embedding
        self.shared_patch = (
            cls_patch_size == loc_patch_size and cls_emb_dim == loc_emb_dim
        )
        cls_M = compute_num_tokens(seq_len, cls_patch_size)
        loc_M = compute_num_tokens(seq_len, loc_patch_size)
        self.cls_M = cls_M
        self.loc_M = loc_M
        self.cls_patch_embed = PatchEmbed(
            in_channels, cls_emb_dim, cls_patch_size, cls_M
        )
        if self.shared_patch:
            self.loc_patch_embed = self.cls_patch_embed
        else:
            self.loc_patch_embed = PatchEmbed(
                in_channels, loc_emb_dim, loc_patch_size, loc_M
            )

        # cls ANN path
        self.cls_layers = self._build_path(
            cls_path, cls_emb_dim, cls_M, cls_depth, drop_path
        )
        self.cls_norm = nn.LayerNorm(cls_emb_dim)
        self.cls_head = nn.Linear(cls_emb_dim, num_classes)

        # loc ANN path
        self.loc_layers = self._build_path(
            loc_path, loc_emb_dim, loc_M, loc_depth, drop_path
        )
        self.loc_norm = nn.LayerNorm(loc_emb_dim)
        self.loc_head = nn.Linear(loc_emb_dim, num_locations)

        # SNN branch (operates on cls path's Z0)
        self.snn = None
        if snn_branch == "plif":
            self.snn = SNNAlarmHead(cls_emb_dim, snn_hidden, num_classes)
        elif snn_branch == "spiketcn":
            self.snn = SpikeTCNAlarmHead(cls_emb_dim, snn_hidden, num_classes)
        elif snn_branch == "ann_alarm":
            self.snn = DenseAlarmHead(cls_emb_dim, snn_hidden, num_classes)
        elif snn_branch != "none":
            raise ValueError(f"unknown snn_branch {snn_branch}")

        # beta fusion (only meaningful with an SNN branch)
        if self.snn is not None and learnable_beta:
            self.beta = nn.Parameter(torch.tensor(float(beta)))
        elif self.snn is not None:
            self.register_buffer("beta", torch.tensor(float(beta)))
        # else: no beta needed (no SNN fusion)

    def _build_path(
        self,
        path_name: str,
        emb_dim: int,
        num_tokens: int,
        depth: int,
        drop_path: float,
    ) -> nn.ModuleList:
        cfg = self.PATH_CFG[path_name]
        return nn.ModuleList(
            [
                DPMixerLayer(
                    emb_dim, num_tokens, drop_path=drop_path, hidden=emb_dim * 2, **cfg
                )
                for _ in range(depth)
            ]
        )

    # ── forward helpers ────────────────────────────────────────────────────

    def _run_ann_path(self, x: Tensor, *, which: str) -> tuple[Tensor, Tensor]:
        """Run patch embed + layer stack for one path. Returns (z, pooled_feat)."""
        if which == "cls":
            pe, layers, norm = self.cls_patch_embed, self.cls_layers, self.cls_norm
            x_ch = x.transpose(1, 2)
        else:
            pe, layers, norm = self.loc_patch_embed, self.loc_layers, self.loc_norm
            x_ch = x.transpose(1, 2)
        z = pe(x_ch)
        for layer in layers:
            z = layer(z)
        pooled = norm(z.mean(dim=1))
        return z, pooled

    def _alarm_pred_from_logit(self, alarm_logit: Tensor, stride: int) -> Tensor:
        """Expected alarm timestep from per-token alarm logits [B, M]."""
        M = alarm_logit.shape[1]
        tokens = torch.arange(M, device=alarm_logit.device, dtype=alarm_logit.dtype)
        p = torch.softmax(alarm_logit, dim=1)  # [B, M]
        pred_token = (p * tokens.unsqueeze(0)).sum(dim=1)  # [B]
        return pred_token * stride

    # ── public forward ─────────────────────────────────────────────────────

    def forward(self, x: Tensor, *, snn_control: str = "none") -> dict:
        """x: [B, T, C]. Returns dict with cls/loc logits, alarm pred, spikes."""
        z0_cls, cls_feat = self._run_ann_path(x, which="cls")
        cls_logits = self.cls_head(cls_feat)

        _, loc_feat = self._run_ann_path(x, which="loc")
        loc_logits = self.loc_head(loc_feat)

        out = {
            "cls_logits": cls_logits,
            "loc_logits": loc_logits,
            "alarm_pred": None,
            "alarm_logit": None,
            "spike_rate": None,
            "event_sparsity": None,
            "spike_count": None,
            "cls_logits_ann": cls_logits,
            "cls_logits_spk": None,
        }

        if self.snn is not None:
            z_in = z0_cls
            if snn_control == "reverse":
                z_in = torch.flip(z_in, dims=[1])
            elif snn_control == "shuffle":
                perm = torch.randperm(z_in.shape[1], device=z_in.device)
                z_in = z_in[:, perm, :]
            s = self.snn(z_in)
            cls_spk = s["cls_spk"]
            beta = self._beta_value()
            out["cls_logits"] = cls_logits + beta * cls_spk
            out["cls_logits_spk"] = cls_spk
            out["alarm_logit"] = s["alarm_logit"]
            out["alarm_pred"] = self._alarm_pred_from_logit(
                s["alarm_logit"], self.cls_stride
            )
            spikes = s["spikes"]
            if spikes is not None:
                spike_rate = spikes.mean().detach()
                out["spike_rate"] = spike_rate
                # Sparsity is defined over the same PLIF output tensor as the
                # spike rate: silent neuron-token slots divided by all slots.
                out["event_sparsity"] = (1.0 - spike_rate).detach()
                out["spike_count"] = spikes.sum(dim=(1, 2)).float().mean().detach()

        return out

    def forward_prefix_cls(self, x_prefix: Tensor) -> Tensor:
        """Compute fused cls logits on a prefix ``x_prefix`` ``[B, K, C]``.

        Recomputes patches from the prefix (no future information), runs the cls
        ANN path and the SNN branch on the shorter token sequence, and returns
        the fused cls logits. Localization is not computed (prefix cls only)."""
        z0, cls_feat = self._run_ann_path(x_prefix, which="cls")
        cls_logits = self.cls_head(cls_feat)
        if self.snn is not None:
            s = self.snn(z0)
            cls_logits = cls_logits + self._beta_value() * s["cls_spk"]
        return cls_logits

    def forward_prefix_snn_cls(self, x_prefix: Tensor) -> Tensor:
        """Compute SNN-only classification logits from an available prefix."""
        if self.snn is None:
            raise RuntimeError("SNN-only prefix logits require an SNN branch")
        z0, _ = self._run_ann_path(x_prefix, which="cls")
        return self.snn(z0)["cls_spk"]

    def _beta_value(self) -> Tensor | float:
        if not hasattr(self, "beta"):
            return 0.0
        return self.beta if isinstance(self.beta, nn.Parameter) else self.beta.item()

    @property
    def threshold_value(self) -> float | None:
        for layer in self.loc_layers:
            if getattr(layer, "use_asb", False):
                return layer.asb.threshold_value
        for layer in self.cls_layers:
            if getattr(layer, "use_asb", False):
                return layer.asb.threshold_value
        return None


# ═══════════════════════════════════════════════════════════════════════════
# Canonical locked configs (main model + high-accuracy upper bound)
# ═══════════════════════════════════════════════════════════════════════════
#
# Selected from the completed architecture ablation:
#   - PatchEmbed : kept, lightweight (small patch / small emb_dim).
#   - ICB        : kept, compact (single layer) -- the strongest cls driver.
#   - ASB no-mask: optional (available via loc_path="asb_nomask"), NOT the
#                  default backbone.
#   - ASB original high-freq mask: removed (harms PMU disturbance recognition).
#   - post-pooling LIF: removed from the main path (not real temporal modeling);
#                  the patch-level learnable-decay LIF branch replaces it.
#   - Large-patch ps32/ed64/d2 + ASB+ICB stack: high-accuracy but low parameter
#                  efficiency -> kept as UPPER_BOUND, not the main model.
#
# The main model is therefore "minimal effective DPMixer + SNN event
# branch": shared lightweight PatchEmbed + compact ICB + a patch-level
# PLIF-inspired branch that supplies alarm / spike evidence along the
# patch-token axis.

FASDNET_CONFIG = dict(
    in_channels=91,
    seq_len=960,
    cls_path="icb",
    cls_patch_size=16,
    cls_emb_dim=48,
    cls_depth=1,
    loc_path="icb",
    loc_patch_size=16,
    loc_emb_dim=48,
    loc_depth=1,
    snn_branch="plif",
    snn_hidden=48,
    beta=1.0,
    learnable_beta=True,
    alarm_loss=True,
    num_classes=5,
    num_locations=276,
    drop_path=0.0,
)

UPPER_BOUND_CONFIG = dict(
    in_channels=91,
    seq_len=960,
    cls_path="asb_icb",
    cls_patch_size=32,
    cls_emb_dim=64,
    cls_depth=2,
    loc_path="asb_icb",
    loc_patch_size=32,
    loc_emb_dim=64,
    loc_depth=2,
    snn_branch="plif",
    snn_hidden=64,
    beta=1.0,
    learnable_beta=True,
    alarm_loss=True,
    num_classes=5,
    num_locations=276,
    drop_path=0.0,
)


def build_fasdnet(**overrides) -> FASDNet:
    """Build the canonical FASD-Net main model."""
    cfg = dict(FASDNET_CONFIG)
    cfg.update(overrides)
    return FASDNet(**cfg)


def build_upper_bound(**overrides) -> FASDNet:
    """Build the high-accuracy upper bound (large patch + ASB+ICB stack)."""
    cfg = dict(UPPER_BOUND_CONFIG)
    cfg.update(overrides)
    return FASDNet(**cfg)


__all__ = [
    "SNNAlarmHead",
    "SpikeTCNAlarmHead",
    "DenseAlarmHead",
    "FASDNet",
    "FASDNET_CONFIG",
    "UPPER_BOUND_CONFIG",
    "build_fasdnet",
    "build_upper_bound",
]
