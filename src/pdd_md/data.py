"""Load the two public AD-3 trajectories without mixing train and test."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np

from .units import NM_TO_ANGSTROM, force_kj_mol_nm_to_ev_angstrom

BASE_URL = "https://huggingface.co/datasets/microsoft/timewarp/resolve/main/AD-3"
FILES = {
    "train": ("ad1-traj-arrays.npz", "ad1-traj-state0.pdb"),
    "test": ("ad2-traj-arrays.npz", "ad2-traj-state0.pdb"),
}


def download_ad3(root: str | Path, splits=("train", "test")) -> list[Path]:
    root = Path(root)
    paths = []
    for split in splits:
        if split not in FILES:
            raise ValueError(f"Unknown AD-3 split: {split}")
        directory = root / "AD-3" / split
        directory.mkdir(parents=True, exist_ok=True)
        for name in FILES[split]:
            path = directory / name
            if not path.exists():
                tmp = path.with_suffix(path.suffix + ".part")
                urlretrieve(f"{BASE_URL}/{split}/{name}", tmp)
                tmp.rename(path)
            paths.append(path)
    return paths


@dataclass
class AD3Trajectory:
    positions: np.ndarray  # [T,A,3], Å
    velocities: np.ndarray  # [T,A,3], Å/ps
    forces: np.ndarray  # [T,A,3], eV/Å
    steps: np.ndarray
    times_ps: np.ndarray
    dt_ps: float
    pdb_path: Path

    def __len__(self):
        return len(self.steps)

    def state(self, index: int):
        return self.positions[index].copy(), self.velocities[index].copy()


def load_ad3(root: str | Path, split: str, max_frames: int | None = None) -> AD3Trajectory:
    if split not in FILES:
        raise ValueError(f"Unknown AD-3 split: {split}")
    directory = Path(root) / "AD-3" / split
    npz = directory / FILES[split][0]
    pdb = directory / FILES[split][1]
    if not npz.exists() or not pdb.exists():
        raise FileNotFoundError(f"Missing AD-3 files in {directory}; run `pdd-md download-data`")
    with np.load(npz) as arrays:
        sl = slice(None, max_frames)
        steps = np.asarray(arrays["step"][sl], dtype=np.int64)
        times_ps = np.asarray(arrays["time"][sl], dtype=np.float64)
        positions = np.asarray(arrays["positions"][sl] * NM_TO_ANGSTROM, dtype=np.float32)
        velocities = np.asarray(arrays["velocities"][sl] * NM_TO_ANGSTROM, dtype=np.float32)
        forces = np.asarray(force_kj_mol_nm_to_ev_angstrom(arrays["forces"][sl]), dtype=np.float32)
    if positions.shape != velocities.shape or positions.shape != forces.shape:
        raise ValueError("AD-3 state array shapes disagree")
    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError("Expected [frame, atom, xyz] arrays")
    if len(steps) > 1 and not np.all(np.isin(np.diff(steps), [1, 9, 90, 900])):
        raise ValueError("Unexpected AD-3 hierarchical frame spacing")
    dt_ps = float(np.median(np.diff(times_ps) / np.diff(steps))) if len(steps) > 1 else 0.001
    if len(steps) > 1 and not np.allclose(np.diff(times_ps), np.diff(steps) * dt_ps, rtol=1e-5, atol=1e-6):
        raise ValueError("AD-3 step and time arrays disagree")
    if not all(np.isfinite(a).all() for a in (positions, velocities, forces)):
        raise ValueError("AD-3 contains non-finite states")
    return AD3Trajectory(positions, velocities, forces, steps, times_ps, dt_ps, pdb)


def pdb_atomic_numbers_and_masses(pdb_path: str | Path):
    from openmm.app import PDBFile
    from openmm import unit

    pdb = PDBFile(str(pdb_path))
    atoms = list(pdb.topology.atoms())
    numbers = np.asarray([a.element.atomic_number for a in atoms], dtype=np.int64)
    masses = np.asarray([a.element.mass.value_in_unit(unit.dalton) for a in atoms], dtype=np.float64)
    return numbers, masses
