"""Minimal leaky integrate-and-fire neuron module."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


class SurrogateSpike(torch.autograd.Function):
    """Heaviside spike with a triangular surrogate gradient."""

    @staticmethod
    def forward(
        ctx,
        membrane_minus_threshold: Tensor,
        surrogate_width: float,
    ) -> Tensor:
        ctx.save_for_backward(membrane_minus_threshold)
        ctx.surrogate_width = surrogate_width
        return (membrane_minus_threshold >= 0).to(membrane_minus_threshold.dtype)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor, None]:
        (membrane_minus_threshold,) = ctx.saved_tensors
        width = ctx.surrogate_width
        distance = membrane_minus_threshold.abs()
        grad = torch.clamp(1.0 - distance / width, min=0.0) / width
        return grad_output * grad, None


@dataclass(frozen=True)
class LIFState:
    """Optional explicit state for one LIF time step."""

    membrane: Tensor


class LIFNeuron(nn.Module):
    """Leaky integrate-and-fire neuron over batched sequences.

    Input shape is ``[batch, time, features]``. The module integrates each time
    step, emits a spike when membrane crosses ``threshold``, then resets the
    membrane by subtracting ``threshold`` for spiking neurons. Returned membrane
    trace is post-reset.
    """

    def __init__(
        self,
        *,
        membrane_decay: float = 0.9,
        threshold: float = 1.0,
        reset_subtract: float | None = None,
        surrogate_width: float = 1.0,
    ) -> None:
        super().__init__()
        self.membrane_decay = _validate_unit_interval(
            membrane_decay,
            name="membrane_decay",
        )
        self.threshold = _validate_positive(threshold, name="threshold")
        self.reset_subtract = (
            self.threshold
            if reset_subtract is None
            else _validate_nonnegative(reset_subtract, name="reset_subtract")
        )
        self.surrogate_width = _validate_positive(
            surrogate_width,
            name="surrogate_width",
        )

    def forward(
        self,
        x: Tensor,
        *,
        initial_membrane: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Return ``(spikes, membrane_trace)`` for ``x``.

        Parameters
        ----------
        x:
            Sequence current with shape ``[batch, time, features]``.
        initial_membrane:
            Optional membrane state with shape ``[batch, features]``.
        """

        if x.ndim != 3:
            msg = "x must have shape [batch, time, features]."
            raise ValueError(msg)

        membrane = _initial_membrane_for(x, initial_membrane)
        spikes = []
        membrane_trace = []
        for time_index in range(x.shape[1]):
            membrane = self.membrane_decay * membrane + x[:, time_index, :]
            spike = SurrogateSpike.apply(
                membrane - self.threshold,
                self.surrogate_width,
            )
            membrane = membrane - spike.detach() * self.reset_subtract
            spikes.append(spike)
            membrane_trace.append(membrane)

        return torch.stack(spikes, dim=1), torch.stack(membrane_trace, dim=1)

    def step(self, input_t: Tensor, state: LIFState) -> tuple[Tensor, LIFState]:
        """Run one LIF step for ``input_t`` shaped ``[batch, features]``."""

        if input_t.ndim != 2:
            msg = "input_t must have shape [batch, features]."
            raise ValueError(msg)
        if state.membrane.shape != input_t.shape:
            msg = "state.membrane must match input_t shape."
            raise ValueError(msg)

        membrane = self.membrane_decay * state.membrane + input_t
        spike = SurrogateSpike.apply(membrane - self.threshold, self.surrogate_width)
        membrane = membrane - spike.detach() * self.reset_subtract
        return spike, LIFState(membrane=membrane)


def _initial_membrane_for(x: Tensor, initial_membrane: Tensor | None) -> Tensor:
    expected_shape = (x.shape[0], x.shape[2])
    if initial_membrane is None:
        return torch.zeros(expected_shape, dtype=x.dtype, device=x.device)
    if tuple(initial_membrane.shape) != expected_shape:
        msg = "initial_membrane must have shape [batch, features]."
        raise ValueError(msg)
    return initial_membrane.to(dtype=x.dtype, device=x.device)


def _validate_unit_interval(value: float, *, name: str) -> float:
    value = float(value)
    if not 0.0 <= value <= 1.0:
        msg = f"{name} must be in the interval [0, 1]."
        raise ValueError(msg)
    return value


def _validate_positive(value: float, *, name: str) -> float:
    value = float(value)
    if value <= 0.0:
        msg = f"{name} must be positive."
        raise ValueError(msg)
    return value


def _validate_nonnegative(value: float, *, name: str) -> float:
    value = float(value)
    if value < 0.0:
        msg = f"{name} must be nonnegative."
        raise ValueError(msg)
    return value
