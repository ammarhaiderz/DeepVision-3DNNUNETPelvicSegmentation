from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import (
    distance_transform_edt,
    generate_binary_structure,
    label,
)


CONNECTIVITY_26 = generate_binary_structure(3, 3)


@dataclass(frozen=True)
class SafeZoneContext:
    largest: np.ndarray
    crop_slices: tuple[slice, slice, slice]
    distance_crop_mm: np.ndarray


def largest_connected_component(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)

    components, count = label(mask, structure=CONNECTIVITY_26)
    sizes = np.bincount(components.ravel())
    sizes[0] = 0
    largest_label = int(np.argmax(sizes))
    if count == 0 or largest_label == 0:
        return np.zeros_like(mask, dtype=bool)
    return components == largest_label


def safe_zone_from_largest(
    prediction: np.ndarray,
    spacing_zyx: np.ndarray,
    distance_mm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if distance_mm < 0:
        raise ValueError("distance_mm must be non-negative.")

    context = build_safe_zone_context(
        prediction, spacing_zyx, max_distance_mm=distance_mm
    )
    filtered, safe_zone, removed = apply_safe_zone_context(
        prediction,
        context,
        distance_mm,
        return_safe_zone=True,
    )
    return filtered, context.largest, safe_zone, removed


def build_safe_zone_context(
    prediction: np.ndarray,
    spacing_zyx: np.ndarray,
    max_distance_mm: float,
) -> SafeZoneContext:
    prediction = np.asarray(prediction, dtype=bool)
    spacing_zyx = np.asarray(spacing_zyx, dtype=np.float64)
    if spacing_zyx.shape != (3,) or np.any(spacing_zyx <= 0):
        raise ValueError(f"Invalid spacing_zyx: {spacing_zyx}")
    if max_distance_mm < 0:
        raise ValueError("max_distance_mm must be non-negative.")

    largest = largest_connected_component(prediction)
    if not largest.any():
        empty_slices = (slice(0, 0), slice(0, 0), slice(0, 0))
        return SafeZoneContext(
            largest=largest,
            crop_slices=empty_slices,
            distance_crop_mm=np.empty((0, 0, 0), dtype=np.float32),
        )

    coordinates = np.argwhere(largest)
    minimum = coordinates.min(axis=0)
    maximum = coordinates.max(axis=0) + 1
    margin_voxels = np.ceil(max_distance_mm / spacing_zyx).astype(int) + 1
    minimum = np.maximum(0, minimum - margin_voxels)
    maximum = np.minimum(np.asarray(prediction.shape), maximum + margin_voxels)
    crop_slices = tuple(
        slice(int(start), int(stop))
        for start, stop in zip(minimum, maximum)
    )
    largest_crop = largest[crop_slices]
    distance_crop = distance_transform_edt(
        ~largest_crop,
        sampling=spacing_zyx,
    )
    return SafeZoneContext(
        largest=largest,
        crop_slices=crop_slices,
        distance_crop_mm=distance_crop,
    )


def apply_safe_zone_context(
    prediction: np.ndarray,
    context: SafeZoneContext,
    distance_mm: float,
    return_safe_zone: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if distance_mm < 0:
        raise ValueError("distance_mm must be non-negative.")
    prediction = np.asarray(prediction, dtype=bool)
    filtered = np.zeros_like(prediction, dtype=bool)
    if context.distance_crop_mm.size:
        safe_zone_crop = context.distance_crop_mm <= distance_mm
        filtered[context.crop_slices] = (
            prediction[context.crop_slices] & safe_zone_crop
        )
    else:
        safe_zone_crop = np.empty((0, 0, 0), dtype=bool)
    if return_safe_zone:
        safe_zone = np.zeros_like(prediction, dtype=bool)
        safe_zone[context.crop_slices] = safe_zone_crop
    else:
        safe_zone = np.empty((0, 0, 0), dtype=bool)
    removed = prediction & ~filtered
    return filtered, safe_zone, removed
