from pathlib import Path

import numpy as np
from openmm import unit
from openmm.app import PDBFile

from pdd_md.teacher import OpenMMTeacher, PhaseState

PDB = Path(__file__).parent / "fixtures" / "ad3-state0.pdb"


def test_teacher_force_is_negative_energy_gradient():
    pdb = PDBFile(str(PDB))
    q = np.asarray(pdb.positions.value_in_unit(unit.angstrom), dtype=np.float64)
    teacher = OpenMMTeacher(PDB)
    force, energy = teacher.force_and_potential(q)
    eps = 1e-4
    plus, minus = q.copy(), q.copy()
    plus[0, 0] += eps
    minus[0, 0] -= eps
    _, eplus = teacher.force_and_potential(plus)
    _, eminus = teacher.force_and_potential(minus)
    # Convert kJ/mol/Å to eV/Å.
    numeric = -(eplus - eminus) / (2 * eps * 96.48533212331)
    assert np.isfinite(energy)
    assert np.isclose(force[0, 0], numeric, rtol=1e-3, atol=1e-3)


def test_teacher_mean_velocity_matches_step():
    pdb = PDBFile(str(PDB))
    q = np.asarray(pdb.positions.value_in_unit(unit.angstrom), dtype=np.float64)
    teacher = OpenMMTeacher(PDB)
    initial = PhaseState(q, np.zeros_like(q))
    dt = 0.0005
    qdot, accel = teacher.mean_velocity(initial, dt)
    next_state = teacher.step(initial, dt)
    assert np.allclose(next_state.q, initial.q + dt * qdot)
    assert np.allclose(next_state.v, initial.v + dt * accel)
