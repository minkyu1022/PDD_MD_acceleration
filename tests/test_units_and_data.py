from pathlib import Path

import numpy as np

from pdd_md.data import load_ad3
from pdd_md.units import (
    ACCEL_PER_FORCE_PER_DALTON,
    force_ev_angstrom_to_kj_mol_nm,
    force_kj_mol_nm_to_ev_angstrom,
)


def test_force_unit_roundtrip_and_acceleration_constant():
    force = np.array([[-1.0, 0.0, 2.5]])
    assert np.allclose(
        force_kj_mol_nm_to_ev_angstrom(force_ev_angstrom_to_kj_mol_nm(force)), force
    )
    assert np.isclose(ACCEL_PER_FORCE_PER_DALTON, 9648.533212331)


def test_loader_uses_recorded_time_and_hierarchical_gaps(tmp_path: Path):
    folder = tmp_path / "AD-3" / "train"
    folder.mkdir(parents=True)
    (folder / "ad1-traj-state0.pdb").write_text("dummy")
    step = np.array([0, 1, 10, 100, 1000], dtype=np.int64)
    time = step * 0.001
    positions = np.zeros((5, 2, 3))
    forces = np.ones_like(positions) * 964.8533212331
    np.savez(
        folder / "ad1-traj-arrays.npz",
        step=step,
        time=time,
        positions=positions,
        velocities=positions,
        forces=forces,
    )
    data = load_ad3(tmp_path, "train")
    assert len(data) == 5
    assert np.isclose(data.dt_ps, 0.001)
    assert np.allclose(data.forces, 1.0)
