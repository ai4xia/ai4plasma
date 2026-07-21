# ai4plasma/data/vpic_hdf5_dataset.py

from __future__ import annotations
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def _beta_from_filename(path: Path) -> float:
    """
    Extract beta value from filename:
    VPIC_PPPL_CSV_Data_official2500_beta0.1_none.h5 -> 0.1
    """
    m = re.search(r"_beta([0-9.]+)_", path.name)
    if m is None:
        raise ValueError(f"Cannot parse beta from filename: {path.name}")
    return float(m.group(1))


def find_h5_files(
    h5_dir: Union[str, Path],
    betas: Optional[Sequence[float]] = None,
) -> List[Path]:
    """
    Find VPIC HDF5 files and sort them by numerical beta.
    """
    h5_dir = Path(os.path.expandvars(str(h5_dir))).expanduser().resolve()
    files = sorted(
        h5_dir.glob("VPIC_PPPL_CSV_Data_official2500_beta*_none.h5"),
        key=_beta_from_filename,
    )

    if betas is not None:
        wanted = {float(b) for b in betas}
        files = [p for p in files if _beta_from_filename(p) in wanted]

    if len(files) == 0:
        raise FileNotFoundError(f"No matching HDF5 files found in {h5_dir}")

    return files


class VPICWindowDataset(Dataset):
    """
    Pure HDF5 temporal-window dataset.

    It reads full-field blocks from HDF5 and returns raw blocks only.
    No masking, no corruption, no task-specific transformation is performed here.

    HDF5 field layout:
        fields: (T, C, Nx, Nz)

    Returned default layout:
        block: (C, delta_t, Nx, Nz)

    This is convenient for PyTorch Conv3d:
        input shape after batching: (B, C, delta_t, Nx, Nz)
    """

    def __init__(
        self,
        h5_dir: Union[str, Path],
        betas: Optional[Sequence[float]] = None,
        delta_t: int = 8,
        stride_t: int = 1,
        run_stride: int = 1,
        layout: str = "C T X Z",
        return_metadata: bool = True,
        load_to_memory: bool = False,
    ):
        """
        Parameters
        ----------
        h5_dir:
            Directory containing *_none.h5 files.

        betas:
            Optional list of beta values to use.
            For first experiments, use one beta only, e.g. betas=[0.1] or [0.2],
            because different beta files have different spatial resolutions.

        delta_t:
            Temporal window length.

        stride_t:
            Temporal stride for sliding windows inside each run.

        run_stride:
            Optional subsampling over runs. Usually keep 1.

        layout:
            Output tensor layout.
            - "C T X Z": returns (C, delta_t, Nx, Nz). Recommended for Conv3d.
            - "C X Z T": returns (C, Nx, Nz, delta_t). Matches your conceptual notation.

        return_metadata:
            Whether to return metadata dict.

        load_to_memory:
            Usually False. Keep HDF5 lazy reading.
        """
        super().__init__()

        assert delta_t >= 1
        assert stride_t >= 1
        assert layout in {"C T X Z", "C X Z T"}

        self._file_handles: Dict[int, h5py.File] = {}
        self.file_infos: List[Dict] = []
        self.samples: List[Tuple[int, str, int]] = []

        self.h5_dir = Path(os.path.expandvars(str(h5_dir))).expanduser().resolve()
        self.h5_files = find_h5_files(self.h5_dir, betas=betas)
        self.delta_t = int(delta_t)
        self.stride_t = int(stride_t)
        self.layout = layout
        self.return_metadata = return_metadata
        self.load_to_memory = load_to_memory
        # each sample: (file_idx, run_name, t0)

        self._build_index(run_stride=run_stride)

    def _build_index(self, run_stride: int = 1) -> None:
        """
        Build a lightweight index of all temporal windows.
        """
        for file_idx, h5_path in enumerate(self.h5_files):
            with h5py.File(h5_path, "r") as f:
                run_names = f["index/run_names"].asstr()[:]
                beta = f["index/beta"][:]
                nu = f["index/nu"][:]
                Bz0 = f["index/Bz0"][:]
                dt = f["index/dt"][:]
                tau = f["index/tau"][:]
                T = f["index/T"][:]
                Nx = f["index/Nx"][:]
                Nz = f["index/Nz"][:]

                info = {
                    "path": str(h5_path),
                    "beta_file": float(f.attrs.get("beta_file", _beta_from_filename(h5_path))),
                    "num_runs": len(run_names),
                    "channel_order": f.attrs.get("channel_order", "Bx,By,Bz,Density"),
                }
                self.file_infos.append(info)

                for i in range(0, len(run_names), run_stride):
                    run_name = str(run_names[i])
                    Ti = int(T[i])

                    if Ti < self.delta_t:
                        continue

                    for t0 in range(0, Ti - self.delta_t + 1, self.stride_t):
                        self.samples.append((file_idx, run_name, t0))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No valid temporal windows found. "
                f"delta_t={self.delta_t}, stride_t={self.stride_t}, files={self.h5_files}"
            )

    def _get_file(self, file_idx: int) -> h5py.File:
        """
        Lazily open HDF5 file per worker process.
        This is important for PyTorch num_workers > 0.
        """
        if file_idx not in self._file_handles:
            path = self.h5_files[file_idx]
            self._file_handles[file_idx] = h5py.File(path, "r")
        return self._file_handles[file_idx]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        file_idx, run_name, t0 = self.samples[idx]
        f = self._get_file(file_idx)

        g = f["runs"][run_name]
        fields = g["fields"]  # (T, C, Nx, Nz)

        block_np = fields[t0 : t0 + self.delta_t]  # (delta_t, C, Nx, Nz)
        block_np = np.asarray(block_np, dtype=np.float32)

        if self.layout == "C T X Z":
            block_np = np.transpose(block_np, (1, 0, 2, 3))  # (C, delta_t, Nx, Nz)
        elif self.layout == "C X Z T":
            block_np = np.transpose(block_np, (1, 2, 3, 0))  # (C, Nx, Nz, delta_t)

        block = torch.from_numpy(block_np)

        if not self.return_metadata:
            return block

        # Read metadata from run name. This avoids repeatedly indexing arrays.
        params = parse_run_name(run_name)

        metadata = {
            "file_idx": file_idx,
            "h5_path": str(self.h5_files[file_idx]),
            "run_name": run_name,
            "t0": int(t0),
            "delta_t": int(self.delta_t),
            "beta": float(params["beta"]),
            "nu": int(params["nu"]),
            "Bz0": float(params["Bz0"]),
            "dt": int(params["dt"]),
            "tau": int(params["tau"]),
        }

        return {
            "block": block,
            "metadata": metadata,
        }

    def close(self) -> None:
        for f in self._file_handles.values():
            try:
                f.close()
            except Exception:
                pass
        self._file_handles = {}

    def __del__(self):
        if hasattr(self, "_file_handles"):
            self.close()


def parse_run_name(run_name: str) -> Dict[str, Union[float, int]]:
    """
    Parse:
        beta0.01_nu0_Bz0.15_dt2_tau100
    """
    pattern = re.compile(
        r"^beta(?P<beta>[0-9.]+)_"
        r"nu(?P<nu>[0-9]+)_"
        r"Bz(?P<Bz0>[0-9.]+)_"
        r"dt(?P<dt>[0-9]+)_"
        r"tau(?P<tau>[0-9]+)$"
    )
    m = pattern.match(run_name)
    if m is None:
        raise ValueError(f"Cannot parse run name: {run_name}")

    d = m.groupdict()
    return {
        "beta": float(d["beta"]),
        "nu": int(d["nu"]),
        "Bz0": float(d["Bz0"]),
        "dt": int(d["dt"]),
        "tau": int(d["tau"]),
    }


def make_vpic_dataloader(
    h5_dir: Union[str, Path],
    betas: Optional[Sequence[float]] = None,
    delta_t: int = 8,
    stride_t: int = 1,
    batch_size: int = 4,
    num_workers: int = 4,
    shuffle: bool = True,
    pin_memory: bool = True,
    layout: str = "C T X Z",
) -> DataLoader:
    """
    Convenience function.

    For first experiments, use one beta at a time:
        betas=[0.1]
    because different beta files have different Nx, Nz.
    """
    dataset = VPICWindowDataset(
        h5_dir=h5_dir,
        betas=betas,
        delta_t=delta_t,
        stride_t=stride_t,
        layout=layout,
        return_metadata=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        drop_last=True,
    )

    return loader


if __name__ == "__main__":
    h5_dir = os.path.join(
        os.environ["SCRATCH"],
        "VPIC_PPPL_HDF5_by_beta_official2500_none_compat",
    )

    dataset = VPICWindowDataset(
        h5_dir=h5_dir,
        betas=[0.2],
        delta_t=8,
        stride_t=2,
        layout="C T X Z",
        return_metadata=True,
    )

    print("num files:", len(dataset.h5_files))
    print("num samples:", len(dataset))
    print("file infos:", dataset.file_infos)

    sample = dataset[0]
    print("block shape:", sample["block"].shape)
    print("metadata:", sample["metadata"])

    loader = make_vpic_dataloader(
        h5_dir=h5_dir,
        betas=[0.2],
        delta_t=8,
        stride_t=2,
        batch_size=2,
        num_workers=2,
    )

    batch = next(iter(loader))
    print("batch block shape:", batch["block"].shape)
    print("batch metadata keys:", batch["metadata"].keys())
