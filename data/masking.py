# data/masking.py

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch


# Five basic mask topologies. Super-resolution, probe arrays, temporal
# interpolation and forecasting are not separate tasks here, they are just
# particular mask geometries covered by these five.
MASK_PATTERNS: Tuple[str, ...] = (
    "spatial_random",
    "spatial_grid",
    "spatial_block",
    "temporal_random",
    "temporal_block",
)

# Increment this whenever the training-time meaning of a mask changes. It is
# stored in checkpoints so auto-resume cannot silently mix incompatible mask
# distributions in one run.
MASKING_VERSION = "independentBD_fiveMask_orientedTemporalBlock_v7"

PATTERN_TO_ID: Dict[str, int] = {name: i for i, name in enumerate(MASK_PATTERNS)}

# Bx, By and Bz always share one observation layout. During training/validation,
# the exact-count path samples their common magnetic layout independently of
# Density for spatial_grid and spatial_random. Density is near-regular for
# spatial_grid and uniformly random without replacement for spatial_random. The
# legacy shared-pattern API remains available to visualization and controlled
# validation callers. Training uses sample_independent_batch_masks so B and
# Density also choose their pattern and severity independently. Spatial layouts
# remain fixed throughout a window.
SPATIAL_MASK_PATTERNS: Tuple[str, ...] = (
    "spatial_random",
    "spatial_grid",
    "spatial_block",
)
TEMPORAL_MASK_PATTERNS: Tuple[str, ...] = (
    "temporal_random",
    "temporal_block",
)

DEFAULT_PATTERN_WEIGHTS: Dict[str, float] = {
    "spatial_random": 1.0,
    "spatial_grid": 1.0,
    "spatial_block": 1.0,
    "temporal_random": 1.0,
    "temporal_block": 1.0,
}

MAX_GRID_STRIDE = 32


# ---------------------------------------------------------------------------
# Scalar random helpers.
#
# All layout decisions are drawn on CPU so that mask sampling never
# synchronizes the GPU and stays reproducible from a single CPU generator.
# ---------------------------------------------------------------------------


def _rand(generator: Optional[torch.Generator] = None) -> float:
    return float(torch.rand((), generator=generator))


def _randint(high: int, generator: Optional[torch.Generator] = None) -> int:
    if high <= 1:
        return 0
    return int(torch.randint(high, (1,), generator=generator).item())


def _log_uniform(low: float, high: float, generator: Optional[torch.Generator] = None) -> float:
    return math.exp(math.log(low) + _rand(generator) * (math.log(high) - math.log(low)))


def _round_clamp(value: float, low: int, high: int) -> int:
    return int(min(max(int(round(value)), low), high))


# ---------------------------------------------------------------------------
# Pattern weights.
# ---------------------------------------------------------------------------


def parse_pattern_weights(tokens: Sequence[str]) -> Dict[str, float]:
    """
    Parse CLI tokens of the form:

        spatial_random 1 spatial_grid 1 spatial_block 1 temporal_random 1
        temporal_block 1

    into a {pattern: weight} dict. Patterns that are not listed get weight 0.
    """
    if len(tokens) % 2 != 0:
        raise ValueError(
            "--mask-patterns expects alternating NAME WEIGHT tokens, "
            f"got an odd number of tokens: {list(tokens)}"
        )

    weights: Dict[str, float] = {}

    for name, raw_weight in zip(tokens[0::2], tokens[1::2]):
        if name not in MASK_PATTERNS:
            raise ValueError(
                f"Unknown mask pattern {name!r}. Available: {list(MASK_PATTERNS)}"
            )
        if name in weights:
            raise ValueError(f"Mask pattern {name!r} was given more than once.")

        try:
            weight = float(raw_weight)
        except ValueError as exc:
            raise ValueError(
                f"Weight for mask pattern {name!r} must be a number, got {raw_weight!r}."
            ) from exc

        if weight < 0:
            raise ValueError(f"Weight for mask pattern {name!r} must be >= 0, got {weight}.")

        weights[name] = weight

    if sum(weights.values()) <= 0:
        raise ValueError("At least one mask pattern must have a positive weight.")

    return weights


def sample_patterns(
    weights: Dict[str, float],
    n: int,
    generator: Optional[torch.Generator] = None,
) -> List[str]:
    """
    Draw n pattern names with probabilities proportional to their weights.
    """
    names = [name for name in MASK_PATTERNS if weights.get(name, 0.0) > 0]

    if not names:
        raise ValueError("No mask pattern has a positive weight.")

    probs = torch.tensor([weights[name] for name in names], dtype=torch.float32)
    idx = torch.multinomial(probs, n, replacement=True, generator=generator)
    return [names[i] for i in idx.tolist()]


# ---------------------------------------------------------------------------
# Per-pattern layout samplers.
#
# Each one returns a small CPU tensor that broadcasts to (B, C, T, X, Z):
#   - spatial patterns return (B_or_1, C, 1, X, Z)
#   - temporal patterns return (1, 1, T, 1, 1)
# 1 = visible, 0 = hidden.
# ---------------------------------------------------------------------------


def _spatial_random_plane(B, C, X, Z, p, generator):
    """
    Scatter hidden voxels at random spatial locations.
    """
    plane = (torch.rand(B, C, 1, X, Z, generator=generator) >= p).float()
    return plane, {}


def _pick_grid_stride(X, Z, target_visible, generator):
    """
    Pick the isotropic stride whose kept fraction 1 / stride**2 is closest to
    target_visible.
    """
    max_stride = max(1, min(MAX_GRID_STRIDE, max(X, Z)))

    stride = torch.arange(1, max_stride + 1, dtype=torch.float32)
    err = (1.0 / stride ** 2 - target_visible).abs()

    return int(torch.argmin(err)) + 1


def _spatial_grid_plane(B, C, X, Z, p, generator, stride=None):
    """
    Regular sparse spatial observations on a square grid with a random origin.
    """
    if stride is None:
        # Training path: the stride follows the sampled mask fraction.
        stride = _pick_grid_stride(X, Z, 1.0 - p, generator)
    else:
        # Evaluation path: a fixed, explicitly requested sampling grid.
        stride = int(stride)
        if stride < 1:
            raise ValueError(f"grid stride must be >= 1, got {stride}")

    offset_x = _randint(stride, generator)
    offset_z = _randint(stride, generator)

    plane = torch.zeros(1, C, 1, X, Z)
    plane[..., offset_x::stride, offset_z::stride] = 1.0

    info = {
        "stride": stride,
        "offset_x": offset_x,
        "offset_z": offset_z,
    }
    return plane, info


def _sparse_probe_grid_plane(B, C, X, Z, probe_count, generator):
    """Create a randomized near-regular spatial layout with exactly N probes."""
    probe_count = int(probe_count)
    if not (0 <= probe_count <= X * Z):
        raise ValueError(
            f"probe_count must lie in [0, {X * Z}], got {probe_count}."
        )

    plane = torch.zeros(1, C, 1, X, Z)
    if probe_count == 0:
        return plane, {
            "probe_count": 0,
            "grid_count_x": 0,
            "grid_count_z": 0,
            "offset_x": 0,
            "offset_z": 0,
        }

    domain_aspect = float(X) / float(Z)
    candidates = []
    for count_x in range(1, min(X, probe_count) + 1):
        count_z = int(math.ceil(probe_count / count_x))
        if count_z > Z:
            continue
        grid_sites = count_x * count_z
        aspect_error = abs(math.log((count_x / count_z) / domain_aspect))
        excess_fraction = float(grid_sites - probe_count) / float(probe_count)
        candidates.append((aspect_error + 0.1 * excess_fraction, count_x, count_z))
    if not candidates:
        raise RuntimeError(
            f"Could not place {probe_count} probes on a {X}x{Z} domain."
        )
    _score, count_x, count_z = min(candidates)

    step_x = max(1, X // count_x)
    step_z = max(1, Z // count_z)
    offset_x = _randint(step_x, generator)
    offset_z = _randint(step_z, generator)
    x_indices = (
        torch.floor(torch.arange(count_x, dtype=torch.float32) * X / count_x)
        .long()
        .add(offset_x)
        .remainder(X)
    )
    z_indices = (
        torch.floor(torch.arange(count_z, dtype=torch.float32) * Z / count_z)
        .long()
        .add(offset_z)
        .remainder(Z)
    )
    grid_x = x_indices[:, None].expand(count_x, count_z).reshape(-1)
    grid_z = z_indices[None, :].expand(count_x, count_z).reshape(-1)
    selected = torch.randperm(grid_x.numel(), generator=generator)[:probe_count]
    plane[..., grid_x[selected], grid_z[selected]] = 1.0
    return plane, {
        "probe_count": probe_count,
        "grid_count_x": count_x,
        "grid_count_z": count_z,
        "offset_x": offset_x,
        "offset_z": offset_z,
    }


def _random_probe_plane(B, C, X, Z, probe_count, generator):
    """Create a uniformly random layout with exactly N distinct probes."""
    probe_count = int(probe_count)
    spatial_sites = X * Z
    if not (0 <= probe_count <= spatial_sites):
        raise ValueError(
            f"probe_count must lie in [0, {spatial_sites}], got {probe_count}."
        )

    plane = torch.zeros(1, C, 1, X, Z)
    if probe_count > 0:
        indices = torch.randperm(spatial_sites, generator=generator)[:probe_count]
        x_indices = torch.div(indices, Z, rounding_mode="floor")
        z_indices = indices.remainder(Z)
        plane[..., x_indices, z_indices] = 1.0

    return plane, {
        "probe_count": probe_count,
        "visible_count": probe_count,
    }


def _spatial_block_plane(B, C, X, Z, p, generator):
    """
    Hide one randomly placed rectangle whose area is close to p * X * Z.
    """
    area = p * X * Z
    aspect = _log_uniform(1.0 / 3.0, 3.0, generator)

    height = math.sqrt(area * aspect)
    width = math.sqrt(area / aspect)

    # Preserve the target area when the first guess does not fit in the box.
    if height > X:
        height = float(X)
        width = area / height
    if width > Z:
        width = float(Z)
        height = min(float(X), area / width)

    hx = _round_clamp(height, 0, X)
    hz = _round_clamp(width, 0, Z)

    x0 = _randint(X - hx + 1, generator)
    z0 = _randint(Z - hz + 1, generator)

    plane = torch.ones(1, C, 1, X, Z)
    plane[..., x0 : x0 + hx, z0 : z0 + hz] = 0.0

    info = {
        "rect_x0": x0,
        "rect_z0": z0,
        "rect_height": hx,
        "rect_width": hz,
    }
    return plane, info


def _temporal_random_frames(T, p, generator):
    """
    Keep a random subset of frames. The whole spatial field is observed on a
    visible frame and fully hidden otherwise. The visible frames can be
    scattered, contiguous, or anything in between.
    """
    n_visible = _round_clamp((1.0 - p) * T, 0, T)

    visible = torch.randperm(T, generator=generator)[:n_visible]
    visible, _ = torch.sort(visible)

    tmask = torch.zeros(1, 1, T, 1, 1)
    tmask[0, 0, visible, 0, 0] = 1.0

    info = {
        "temporal_mode": "random_frames",
        "temporal_direction": "scattered",
        "num_visible_frames": int(n_visible),
        "visible_frames": [int(t) for t in visible.tolist()],
    }
    return tmask, info


def _temporal_block_from_boundaries(T: int, start: int, end: int):
    """Build one oriented temporal block from two distinct boundaries."""
    if not (0 <= start <= T and 0 <= end <= T):
        raise ValueError(
            f"Temporal boundaries must lie in [0, {T}], got {start} and {end}."
        )
    if start == end:
        raise ValueError("Temporal block boundaries must be different.")

    if end > start:
        tmask = torch.ones(1, 1, T, 1, 1)
        tmask[:, :, start:end] = 0.0
        orientation = "inside_masked"
        temporal_direction = "forward"
    else:
        tmask = torch.zeros(1, 1, T, 1, 1)
        tmask[:, :, end:start] = 1.0
        orientation = "inside_visible"
        temporal_direction = "reverse"

    visible = torch.nonzero(tmask[0, 0, :, 0, 0], as_tuple=False).flatten()
    info = {
        "temporal_mode": "oriented_block",
        "temporal_direction": temporal_direction,
        "orientation": orientation,
        "start": int(start),
        "end": int(end),
        "num_visible_frames": int(visible.numel()),
        "visible_frames": [int(t) for t in visible.tolist()],
        "actual_mask_fraction": float(1.0 - tmask.mean().item()),
    }
    return tmask, info


def _temporal_block_frames(T, generator):
    """Uniformly sample two oriented boundaries from [0, T]."""
    boundaries = torch.randperm(T + 1, generator=generator)[:2]
    return _temporal_block_from_boundaries(
        T,
        int(boundaries[0]),
        int(boundaries[1]),
    )


def _sample_modality_mask(
    *,
    modality: str,
    pattern: str,
    T: int,
    X: int,
    Z: int,
    mask_fraction: float,
    generator: Optional[torch.Generator],
    visible_count: Optional[int] = None,
):
    """Sample one B or Density layout without sampling the other modality."""
    p = float(mask_fraction)
    if pattern == "spatial_random":
        if visible_count is None:
            return _spatial_random_plane(1, 1, X, Z, p, generator)
        return _random_probe_plane(1, 1, X, Z, visible_count, generator)
    if pattern == "spatial_grid":
        if visible_count is None:
            return _spatial_grid_plane(1, 1, X, Z, p, generator)
        if modality == "magnetic":
            # Preserve the existing magnetic exact-count random layout.
            return _random_probe_plane(1, 1, X, Z, visible_count, generator)
        return _sparse_probe_grid_plane(1, 1, X, Z, visible_count, generator)
    if pattern == "spatial_block":
        return _spatial_block_plane(1, 1, X, Z, p, generator)
    if pattern == "temporal_random":
        return _temporal_random_frames(T, p, generator)
    if pattern == "temporal_block":
        return _temporal_block_frames(T, generator)
    raise ValueError(f"Unhandled mask pattern {pattern!r}")


# ---------------------------------------------------------------------------
# Public mask API.
# ---------------------------------------------------------------------------


def sample_mask(
    shape: Sequence[int],
    pattern: str,
    mask_fraction: float,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    generator: Optional[torch.Generator] = None,
    grid_stride: Optional[int] = None,
    density_probe_count: Optional[int] = None,
    magnetic_visible_count: Optional[int] = None,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Sample one mask for a (B, C, T, X, Z) block.

    Parameters
    ----------
    pattern:
        One of MASK_PATTERNS.

    mask_fraction:
        Target fraction of hidden voxels, so the visible fraction is
        approximately 1 - mask_fraction. Discrete patterns only approximate it.

    generator:
        Optional CPU torch.Generator. Every random decision is drawn from it,
        which makes masks reproducible and independent of the device.

    grid_stride:
        Optional stride that pins the spatial_grid sampling lattice instead of
        deriving it from mask_fraction. The grid offset stays random. Intended
        for evaluation; training leaves this at None.

    density_probe_count:
        Optional exact Density probe count for spatial_grid or spatial_random.
        Density uses a randomized near-regular grid for spatial_grid and
        uniformly random distinct sites for spatial_random.

    magnetic_visible_count:
        Optional exact number of visible magnetic sites for spatial_grid or
        spatial_random. Bx, By and Bz use one shared uniformly random layout.
        If omitted while density_probe_count is set, all magnetic sites remain
        visible for backward-compatible visualization calls.

    Returns
    -------
    mask:
        (B, C, T, X, Z) tensor on `device`. 1 = visible, 0 = hidden.

    info:
        Debug/logging metadata: pattern, target and actual mask fraction, plus
        pattern-specific layout parameters.
    """
    if pattern not in MASK_PATTERNS:
        raise ValueError(f"Unknown mask pattern {pattern!r}. Available: {list(MASK_PATTERNS)}")

    if len(shape) != 5:
        raise ValueError(f"Expected a (B, C, T, X, Z) shape, got {tuple(shape)}")

    B, C, T, X, Z = (int(s) for s in shape)
    if C != 4:
        raise ValueError(
            "VPIC masking expects exactly four channels ordered as "
            f"(Bx, By, Bz, Density), got C={C}."
        )
    p = float(mask_fraction)

    if magnetic_visible_count is not None and (
        pattern not in {"spatial_grid", "spatial_random"}
        or density_probe_count is None
    ):
        raise ValueError(
            "magnetic_visible_count requires spatial_grid or spatial_random "
            "together with density_probe_count."
        )

    if pattern == "spatial_random":
        if density_probe_count is None:
            magnetic_small, magnetic_info = _spatial_random_plane(
                B, 1, X, Z, p, generator
            )
            density_small, density_info = _spatial_random_plane(
                B, 1, X, Z, p, generator
            )
        else:
            if magnetic_visible_count is None:
                magnetic_small = torch.ones(1, 1, 1, X, Z)
                magnetic_info = {"visible_count": X * Z}
            else:
                magnetic_small, magnetic_info = _random_probe_plane(
                    B, 1, X, Z, magnetic_visible_count, generator
                )
            density_small, density_info = _random_probe_plane(
                B, 1, X, Z, density_probe_count, generator
            )
    elif pattern == "spatial_grid":
        if density_probe_count is None:
            magnetic_small, magnetic_info = _spatial_grid_plane(
                B, 1, X, Z, p, generator, stride=grid_stride
            )
            density_small, density_info = _spatial_grid_plane(
                B, 1, X, Z, p, generator, stride=grid_stride
            )
        else:
            if grid_stride is not None:
                raise ValueError(
                    "grid_stride and density_probe_count cannot be set together."
                )
            if magnetic_visible_count is None:
                magnetic_small = torch.ones(1, 1, 1, X, Z)
                magnetic_info = {"visible_count": X * Z}
            else:
                magnetic_small, magnetic_info = _random_probe_plane(
                    B, 1, X, Z, magnetic_visible_count, generator
                )
            density_small, density_info = _sparse_probe_grid_plane(
                B, 1, X, Z, density_probe_count, generator
            )
    elif pattern == "spatial_block":
        magnetic_small, magnetic_info = _spatial_block_plane(
            B, 1, X, Z, p, generator
        )
        density_small = magnetic_small
        density_info = dict(magnetic_info)
    elif pattern == "temporal_random":
        magnetic_small, magnetic_info = _temporal_random_frames(T, p, generator)
        density_small = magnetic_small
        density_info = dict(magnetic_info)
    elif pattern == "temporal_block":
        magnetic_small, magnetic_info = _temporal_block_frames(T, generator)
        density_small = magnetic_small
        density_info = dict(magnetic_info)
    else:
        raise ValueError(f"Unhandled mask pattern {pattern!r}")

    magnetic_small = magnetic_small.expand(
        magnetic_small.shape[0], 3, *magnetic_small.shape[2:]
    )
    small = torch.cat([magnetic_small, density_small], dim=1)

    # Keep the magnetic layout metadata at the legacy top-level keys and add
    # explicit prefixed metadata for both independently sampled modalities.
    info = dict(magnetic_info)
    info.update({f"magnetic_{key}": value for key, value in magnetic_info.items()})
    info.update({f"density_{key}": value for key, value in density_info.items()})

    info["pattern"] = pattern
    info["target_mask_fraction"] = p
    # The small tensor broadcasts over the omitted dimensions, so its mean is
    # already the exact mask fraction of the expanded tensor. Measuring it here
    # keeps the measurement on CPU.
    info["actual_mask_fraction"] = float(1.0 - small.mean().item())
    info["magnetic_actual_mask_fraction"] = float(
        1.0 - magnetic_small.mean().item()
    )
    info["density_actual_mask_fraction"] = float(
        1.0 - density_small.mean().item()
    )

    mask = small.to(device=device, dtype=dtype).expand(B, C, T, X, Z).contiguous()
    return mask, info


def sample_batch_masks(
    shape: Sequence[int],
    patterns: Sequence[str],
    mask_fractions: Sequence[float],
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    generator: Optional[torch.Generator] = None,
    density_probe_counts: Optional[Sequence[Optional[int]]] = None,
    magnetic_visible_counts: Optional[Sequence[Optional[int]]] = None,
) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
    """
    Sample one independent mask per batch element, so a single batch can mix
    patterns, mask fractions and layouts.
    """
    B, C, T, X, Z = (int(s) for s in shape)

    if len(patterns) != B or len(mask_fractions) != B:
        raise ValueError(
            f"Expected {B} patterns and {B} mask fractions, "
            f"got {len(patterns)} and {len(mask_fractions)}."
        )
    if density_probe_counts is None:
        density_probe_counts = [None] * B
    if len(density_probe_counts) != B:
        raise ValueError(
            f"Expected {B} density probe counts, got {len(density_probe_counts)}."
        )
    if magnetic_visible_counts is None:
        magnetic_visible_counts = [None] * B
    if len(magnetic_visible_counts) != B:
        raise ValueError(
            f"Expected {B} magnetic visible counts, "
            f"got {len(magnetic_visible_counts)}."
        )

    masks = []
    infos = []

    for i in range(B):
        mask, info = sample_mask(
            (1, C, T, X, Z),
            pattern=patterns[i],
            mask_fraction=float(mask_fractions[i]),
            device=device,
            dtype=dtype,
            generator=generator,
            density_probe_count=density_probe_counts[i],
            magnetic_visible_count=magnetic_visible_counts[i],
        )
        masks.append(mask)
        infos.append(info)

    return torch.cat(masks, dim=0), infos


def sample_independent_batch_masks(
    shape: Sequence[int],
    magnetic_patterns: Sequence[str],
    density_patterns: Sequence[str],
    magnetic_mask_fractions: Sequence[float],
    density_mask_fractions: Sequence[float],
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    generator: Optional[torch.Generator] = None,
    density_probe_counts: Optional[Sequence[Optional[int]]] = None,
    magnetic_visible_counts: Optional[Sequence[Optional[int]]] = None,
) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
    """Sample independent magnetic and Density patterns for every sample.

    Bx/By/Bz share the sampled magnetic layout. Density has its own pattern,
    severity and random layout, even when both modalities draw the same pattern
    name. The legacy sample_mask/sample_batch_masks shared-pattern path remains
    unchanged for visualization and controlled validation.
    """
    if len(shape) != 5:
        raise ValueError(f"Expected a (B, C, T, X, Z) shape, got {tuple(shape)}")
    B, C, T, X, Z = (int(s) for s in shape)
    if C != 4:
        raise ValueError(
            "VPIC masking expects exactly four channels ordered as "
            f"(Bx, By, Bz, Density), got C={C}."
        )

    sequences = {
        "magnetic patterns": magnetic_patterns,
        "density patterns": density_patterns,
        "magnetic mask fractions": magnetic_mask_fractions,
        "density mask fractions": density_mask_fractions,
    }
    for label, values in sequences.items():
        if len(values) != B:
            raise ValueError(f"Expected {B} {label}, got {len(values)}.")

    for pattern in (*magnetic_patterns, *density_patterns):
        if pattern not in MASK_PATTERNS:
            raise ValueError(
                f"Unknown mask pattern {pattern!r}. Available: {list(MASK_PATTERNS)}"
            )

    if density_probe_counts is None:
        density_probe_counts = [None] * B
    if magnetic_visible_counts is None:
        magnetic_visible_counts = [None] * B
    if len(density_probe_counts) != B or len(magnetic_visible_counts) != B:
        raise ValueError(
            f"Expected {B} Density and magnetic exact counts, got "
            f"{len(density_probe_counts)} and {len(magnetic_visible_counts)}."
        )

    masks = []
    infos = []
    for i in range(B):
        magnetic_pattern = magnetic_patterns[i]
        density_pattern = density_patterns[i]
        magnetic_fraction = float(magnetic_mask_fractions[i])
        density_fraction = float(density_mask_fractions[i])

        magnetic_small, magnetic_info = _sample_modality_mask(
            modality="magnetic",
            pattern=magnetic_pattern,
            T=T,
            X=X,
            Z=Z,
            mask_fraction=magnetic_fraction,
            generator=generator,
            visible_count=magnetic_visible_counts[i],
        )
        density_small, density_info = _sample_modality_mask(
            modality="density",
            pattern=density_pattern,
            T=T,
            X=X,
            Z=Z,
            mask_fraction=density_fraction,
            generator=generator,
            visible_count=density_probe_counts[i],
        )

        # Spatial masks are shaped (1, 1, 1, X, Z) and temporal masks are
        # shaped (1, 1, T, 1, 1). Independent modalities may choose one of
        # each, so broadcast both to the common spatiotemporal shape before
        # concatenating channels.
        magnetic_small = magnetic_small.expand(1, 3, T, X, Z)
        density_small = density_small.expand(1, 1, T, X, Z)
        small = torch.cat([magnetic_small, density_small], dim=1)
        mask = small.to(device=device, dtype=dtype).contiguous()

        magnetic_actual = float(1.0 - magnetic_small.mean().item())
        density_actual = float(1.0 - density_small.mean().item())
        info: Dict[str, Any] = {
            "magnetic_pattern": magnetic_pattern,
            "density_pattern": density_pattern,
            "magnetic_target_mask_fraction": magnetic_fraction,
            "density_target_mask_fraction": density_fraction,
            "magnetic_actual_mask_fraction": magnetic_actual,
            "density_actual_mask_fraction": density_actual,
            "magnetic_visible_fraction": 1.0 - magnetic_actual,
            "density_visible_fraction": 1.0 - density_actual,
            "actual_mask_fraction": (3.0 * magnetic_actual + density_actual) / 4.0,
        }
        info.update(
            {f"magnetic_{key}": value for key, value in magnetic_info.items()}
        )
        info.update(
            {f"density_{key}": value for key, value in density_info.items()}
        )
        masks.append(mask)
        infos.append(info)

    return torch.cat(masks, dim=0), infos


def make_visible_input(
    block: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    return block * mask


# ---------------------------------------------------------------------------
# Losses.
# ---------------------------------------------------------------------------


def masked_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    visible_mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    MSE only on hidden region.

    visible_mask:
        1 = visible input
        0 = hidden target region
    """
    hidden_mask = 1.0 - visible_mask
    loss = ((pred - target) ** 2) * hidden_mask
    return loss.sum() / (hidden_mask.sum() + eps)


def masked_mae_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    visible_mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    MAE only on hidden region.
    """
    hidden_mask = 1.0 - visible_mask
    loss = torch.abs(pred - target) * hidden_mask
    return loss.sum() / (hidden_mask.sum() + eps)


def full_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    visible_mask: torch.Tensor,
) -> torch.Tensor:
    """
    MSE balanced across B/Density groups and visible/hidden regions.
    """
    squared_error = (pred - target) ** 2
    group_losses = []

    for channel_slice in (slice(0, 3), slice(3, 4)):
        group_error = squared_error[:, channel_slice]
        group_visible_mask = visible_mask[:, channel_slice]
        group_hidden_mask = 1.0 - group_visible_mask

        visible_count = group_visible_mask.sum()
        hidden_count = group_hidden_mask.sum()
        component_counts = torch.stack([visible_count, hidden_count])
        component_losses = torch.stack(
            [
                (group_error * group_visible_mask).sum()
                / visible_count.clamp_min(1.0),
                (group_error * group_hidden_mask).sum()
                / hidden_count.clamp_min(1.0),
            ]
        )
        valid_components = (component_counts > 0).to(component_losses.dtype)
        group_losses.append(
            (component_losses * valid_components).sum() / valid_components.sum()
        )

    return torch.stack(group_losses).mean()


def full_mae_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    MAE over all voxels, including visible and hidden regions.
    """
    return torch.mean(torch.abs(pred - target))


def channel_completion_mask(
    block: torch.Tensor,
    visible_channels: Tuple[int, ...],
) -> torch.Tensor:
    B, C, T, X, Z = block.shape

    mask = torch.zeros_like(block)
    for c in visible_channels:
        if not (0 <= c < C):
            raise ValueError(f"Invalid channel index {c}, C={C}")
        mask[:, c, :, :, :] = 1.0

    return mask
