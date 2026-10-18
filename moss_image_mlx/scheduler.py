"""MOSS-Image sampling schedule: flow-matching Euler on sigmas snapped to the 1000-step training grid.

Mirrors MOSSImageScheduler.set_timesteps and diffusers' FlowMatchEulerDiscreteScheduler.step as used by the
official pipeline, in float32, including torch's linspace algorithm so the snapped grid is bit-identical."""

import numpy as np

F32 = np.float32


def torch_linspace(start: float, end: float, steps: int) -> np.ndarray:
    """torch.linspace(start, end, steps) for float32 on the CPU: the first half counts up from start, the second
    half down from end, each value a fused multiply-add (one rounding). The float32 x float32 product is exact in
    float64, so evaluating in float64 and rounding once reproduces it (verified bit-equal against torch 2.14)."""
    if steps == 1:
        return np.array([start], dtype=F32)
    start, end = F32(start), F32(end)
    step = np.float64(F32((end - start) / F32(steps - 1)))
    idx = np.arange(steps, dtype=np.float64)
    halfway = steps // 2
    up = np.float64(start) + step * idx[:halfway]
    down = np.float64(end) - step * (steps - idx[halfway:] - 1)
    return np.concatenate([up, down]).astype(F32)


def _grid(count: int, shift: float, total: int) -> np.ndarray:
    lower = shift / (total + shift - 1)  # Python float, as in the reference
    values = torch_linspace(1.0, lower, count)
    return (F32(shift) * values / (F32(1) + F32(shift - 1) * values)).astype(F32)


def sigmas_for(steps: int, shift: float = 5.0, training_shift: float = 1.0, total: int = 1000) -> np.ndarray:
    """Snapped sigmas (float32) followed by a terminal 0, exactly as the reference scheduler produces them."""
    if steps < 1 or shift <= 0 or training_shift <= 0:
        raise ValueError("Steps and shifts must be positive.")
    sigmas = _grid(steps, shift, total)
    training = _grid(total, training_shift, total)[::-1].copy()
    upper = np.clip(np.searchsorted(training, sigmas, side="left"), None, total - 1)
    lower = np.clip(upper - 1, 0, None)
    nearest = np.where(np.abs(sigmas - training[lower]) <= np.abs(training[upper] - sigmas), lower, upper)
    snapped = training[nearest]
    keep = np.concatenate([[True], snapped[1:] != snapped[:-1]])  # torch.unique_consecutive
    return np.concatenate([snapped[keep], [F32(0)]]).astype(F32)


def timesteps_for(sigmas: np.ndarray, total: int = 1000) -> np.ndarray:
    return (sigmas[:-1] * F32(total)).astype(F32)
