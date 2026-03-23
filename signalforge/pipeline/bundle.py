"""
signalforge.pipeline.bundle

FeatureBundle: Stage 5 (Assemble) output.

Takes a list of FeatureTensors — one per (channel, metric) combination per
entity — and assembles them into a multi-channel tensor dataset ready for
machine learning.

Shape contract per entity:

    (n_channels, n_scales, n_time_steps)  float32

This is a multi-channel image. A CNN sees it exactly as it sees RGB — same
shape, same interface. The scale axis is height, time axis is width, channels
are depth.

Because all FeatureTensors from the same SamplingPlan have identical
(n_scales, n_time) shape, all entity tensors in the bundle are identical in
shape. No padding, no alignment — stack directly into a batch.

PyTorch is optional. The bundle stores numpy arrays. Call as_torch() or
as_dataset() when PyTorch is available. Everything else — numpy export,
CSV export, inspection — works without it.

Entity identity
---------------
An entity is identified by the canonical serialization of its keys dict.
All FeatureTensors with the same keys belong to the same entity and are
stacked as separate channels.

Channel naming
--------------
Each channel is named "{channel}/{metric}/{feature}" where feature is the
feature array name from FeatureTensor (e.g. "mean", "mean_zscore", etc.).
channel_index maps these names to their integer position in the tensor.
"""

from __future__ import annotations

import csv
import os
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .feature import FeatureTensor


# ---------------------------------------------------------------------------
# Entity key serialization
# ---------------------------------------------------------------------------


def _entity_id(keys: dict) -> str:
    """Canonical, deterministic string identifier from a keys dict."""
    parts = []
    for k in sorted(keys):
        v = keys[k]
        if isinstance(v, list):
            parts.append(f"{k}={','.join(sorted(v))}")
        else:
            parts.append(f"{k}={v}")
    return ";".join(parts) if parts else "_"


# ---------------------------------------------------------------------------
# FeatureBundle
# ---------------------------------------------------------------------------


class FeatureBundle:
    """
    Multi-channel tensor dataset assembled from FeatureTensors.

    Stores data as numpy float32 arrays. PyTorch tensors are produced
    on demand via as_torch() or as_dataset().

    Attributes
    ----------
    entities : list[str]
        Entity identifiers, one per array. Derived from FeatureTensor keys.
    arrays : list[np.ndarray]
        One (n_channels, n_scales, n_time) float32 array per entity.
    channel_index : dict[str, int]
        Maps "{channel}/{metric}/{feature}" → channel position.
    scale_index : tuple[int, ...]
        Window sizes per scale row. From SamplingPlan.
    time_index : tuple[int, ...]
        Bin positions per time column.
    coordinates : tuple[dict, ...]
        Prime exponent vector per scale row.
    sampling_plan_id : str
    labels : dict[str, any] or None
        Entity-level labels for supervised learning.
    metadata : dict
        Arbitrary key-value pairs for reproducibility.
    """

    def __init__(
        self,
        entities: List[str],
        arrays: List[np.ndarray],
        channel_index: Dict[str, int],
        scale_index: Tuple[int, ...],
        time_index: Tuple[int, ...],
        coordinates: Tuple[dict, ...],
        sampling_plan_id: str,
        labels: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if len(entities) != len(arrays):
            raise ValueError(
                f"len(entities)={len(entities)} != len(arrays)={len(arrays)}"
            )
        if arrays:
            shape = arrays[0].shape
            for i, arr in enumerate(arrays):
                if arr.shape != shape:
                    raise ValueError(
                        f"arrays[{i}].shape {arr.shape} != {shape}: "
                        f"all entity arrays must have identical shape"
                    )
        self.entities = list(entities)
        self.arrays = list(arrays)
        self.channel_index = dict(channel_index)
        self.scale_index = scale_index
        self.time_index = time_index
        self.coordinates = coordinates
        self.sampling_plan_id = sampling_plan_id
        self.labels = labels
        self.metadata = metadata or {}

    @property
    def shape(self) -> Tuple[int, int, int]:
        """(n_channels, n_scales, n_time) — shape of each entity array."""
        if not self.arrays:
            return (0, 0, 0)
        return self.arrays[0].shape

    def __len__(self) -> int:
        return len(self.entities)

    def __repr__(self) -> str:
        return (
            f"FeatureBundle("
            f"entities={len(self.entities)}, "
            f"shape={self.shape}, "
            f"channels={len(self.channel_index)})"
        )

    # --- PyTorch interface ---

    def as_torch(self):
        """
        Return list of torch.Tensor, one per entity.

        Requires PyTorch. Raises ImportError if not installed.
        """
        try:
            import torch
        except ImportError:
            raise ImportError(
                "PyTorch is required for as_torch(). "
                "Install it: https://pytorch.org/get-started/locally/"
            )
        return [torch.from_numpy(arr) for arr in self.arrays]

    def as_dataset(self, transform: Optional[Callable] = None):
        """
        Return a SurfaceDataset (torch.utils.data.Dataset).

        Parameters
        ----------
        transform : callable, optional
            Applied to each tensor at sample load time. Normalization,
            augmentation, channel selection — entirely the practitioner's choice.

        Requires PyTorch. Raises ImportError if not installed.
        """
        try:
            import torch
            import torch.utils.data
        except ImportError:
            raise ImportError(
                "PyTorch is required for as_dataset(). "
                "Install it: https://pytorch.org/get-started/locally/"
            )
        return SurfaceDataset(self, transform=transform)

    # --- Serialization ---

    def save(self, path: str) -> None:
        """
        Save to disk. Uses torch.save if PyTorch is available, else numpy.

        path should end in .pt (torch) or .npz (numpy). If PyTorch is not
        available and path ends in .pt, it is saved as .npz instead.
        """
        try:
            import torch
            torch.save(self, path)
        except ImportError:
            npz_path = path.replace(".pt", ".npz")
            np.savez(
                npz_path,
                arrays=np.stack(self.arrays) if self.arrays else np.array([]),
            )

    @classmethod
    def load(cls, path: str) -> "FeatureBundle":
        """Load a bundle saved with save()."""
        try:
            import torch
            return torch.load(path)
        except ImportError:
            raise ImportError(
                "PyTorch is required to load .pt bundles. "
                "For .npz files, use np.load() directly."
            )

    def export_numpy(self, directory: str) -> None:
        """Save one .npy file per entity to directory."""
        os.makedirs(directory, exist_ok=True)
        for entity, arr in zip(self.entities, self.arrays):
            safe = entity.replace("/", "_").replace(";", "_").replace("=", "-")
            np.save(os.path.join(directory, f"{safe}.npy"), arr)

    def export_csv(self, directory: str) -> None:
        """
        Export all arrays to CSV for inspection (not training).

        One file per entity. Rows are (channel, scale, time, value).
        """
        os.makedirs(directory, exist_ok=True)
        idx_to_channel = {v: k for k, v in self.channel_index.items()}
        for entity, arr in zip(self.entities, self.arrays):
            safe = entity.replace("/", "_").replace(";", "_").replace("=", "-")
            fpath = os.path.join(directory, f"{safe}.csv")
            with open(fpath, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["channel", "scale", "time", "value"])
                n_ch, n_sc, n_t = arr.shape
                for c in range(n_ch):
                    ch_name = idx_to_channel.get(c, str(c))
                    for s in range(n_sc):
                        scale = self.scale_index[s] if s < len(self.scale_index) else s
                        for t in range(n_t):
                            time = self.time_index[t] if t < len(self.time_index) else t
                            writer.writerow([ch_name, scale, time, arr[c, s, t]])


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------


class SurfaceDataset:
    """
    torch.utils.data.Dataset wrapping a FeatureBundle.

    Returns (tensor, label) pairs if labels are present, tensors otherwise.
    The transform hook is the practitioner's entry point for normalization,
    augmentation, and channel selection.
    """

    def __init__(self, bundle: FeatureBundle, transform: Optional[Callable] = None):
        try:
            import torch
            import torch.utils.data
            self._torch = torch
        except ImportError:
            raise ImportError("PyTorch required for SurfaceDataset.")
        self.bundle = bundle
        self.transform = transform
        self._tensors = bundle.as_torch()

    def __len__(self) -> int:
        return len(self.bundle)

    def __getitem__(self, idx: int):
        tensor = self._tensors[idx]
        if self.transform is not None:
            tensor = self.transform(tensor)
        if self.bundle.labels is not None:
            entity = self.bundle.entities[idx]
            label = self.bundle.labels.get(entity)
            return tensor, label
        return tensor

    def __add__(self, other: "SurfaceDataset") -> "SurfaceDataset":
        """Concatenate two SurfaceDatasets from compatible bundles."""
        import torch.utils.data
        return torch.utils.data.ConcatDataset([self, other])


# ---------------------------------------------------------------------------
# assemble()
# ---------------------------------------------------------------------------


def assemble(
    feature_tensors: List[FeatureTensor],
    feature_names: Optional[List[str]] = None,
    labels: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> FeatureBundle:
    """
    Assemble FeatureTensors into a FeatureBundle.

    Groups FeatureTensors by entity (canonical keys), then stacks the
    selected feature arrays across all (channel, metric) combinations into
    a single (n_channels, n_scales, n_time) array per entity.

    Parameters
    ----------
    feature_tensors : list of FeatureTensor
        All FeatureTensors to include. May span multiple entities,
        channels, and metrics.
    feature_names : list of str, optional
        Which feature arrays to include as channels. Default: all features
        from each FeatureTensor in sorted order.
    labels : dict[str, any], optional
        Entity-level labels. Keys are entity identifier strings.
    metadata : dict, optional
        Arbitrary key-value pairs attached to the bundle.

    Returns
    -------
    FeatureBundle
    """
    if not feature_tensors:
        raise ValueError("feature_tensors is empty.")

    # Verify all share the same SamplingPlan geometry.
    plan_id = feature_tensors[0].sampling_plan_id
    time_axis = feature_tensors[0].time_axis
    scale_axis = feature_tensors[0].scale_axis
    coordinates = feature_tensors[0].coordinates

    for ft in feature_tensors[1:]:
        if ft.sampling_plan_id != plan_id:
            raise ValueError(
                f"Mismatched sampling_plan_id: {ft.sampling_plan_id!r} vs {plan_id!r}. "
                f"All FeatureTensors must share the same SamplingPlan."
            )
        if ft.time_axis != time_axis or ft.scale_axis != scale_axis:
            raise ValueError(
                f"Mismatched grid shape. All FeatureTensors must have the same "
                f"time_axis and scale_axis."
            )

    # Group by entity.
    # entity_id → list of (channel_prefix, FeatureTensor)
    groups: Dict[str, List[Tuple[str, FeatureTensor]]] = defaultdict(list)
    for ft in feature_tensors:
        eid = _entity_id(ft.keys)
        prefix = f"{ft.channel}/{ft.metric}"
        groups[eid].append((prefix, ft))

    # Determine channel order: sorted by (prefix, feature_name).
    # Collect all (prefix, feature_name) pairs present across all FTs.
    all_channel_names: List[str] = []
    seen: set = set()
    for ft in feature_tensors:
        prefix = f"{ft.channel}/{ft.metric}"
        fnames = feature_names if feature_names is not None else sorted(ft.values)
        for fname in fnames:
            full = f"{prefix}/{fname}"
            if full not in seen:
                seen.add(full)
                all_channel_names.append(full)

    channel_index = {name: i for i, name in enumerate(all_channel_names)}
    n_channels = len(all_channel_names)
    n_scales = len(scale_axis)
    n_time = len(time_axis)

    # Build one array per entity.
    entities: List[str] = sorted(groups.keys())
    arrays: List[np.ndarray] = []

    for eid in entities:
        arr = np.full((n_channels, n_scales, n_time), np.nan, dtype=np.float32)
        for prefix, ft in groups[eid]:
            fnames = feature_names if feature_names is not None else sorted(ft.values)
            for fname in fnames:
                if fname not in ft.values:
                    continue
                ch_name = f"{prefix}/{fname}"
                ch_idx = channel_index[ch_name]
                arr[ch_idx] = ft.values[fname].astype(np.float32)
        arrays.append(arr)

    return FeatureBundle(
        entities=entities,
        arrays=arrays,
        channel_index=channel_index,
        scale_index=scale_axis,
        time_index=time_axis,
        coordinates=coordinates,
        sampling_plan_id=plan_id,
        labels=labels,
        metadata=metadata or {},
    )
