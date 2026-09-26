"""Explicit units used at each model/teacher boundary.

Internal positions: Å; velocities: Å/ps; masses: dalton.
Internal forces: eV/Å; accelerations: Å/ps²; time: ps.
"""

import numpy as np

NM_TO_ANGSTROM = 10.0
KJ_MOL_NM_PER_EV_ANGSTROM = 964.8533212331
ACCEL_PER_FORCE_PER_DALTON = NM_TO_ANGSTROM * KJ_MOL_NM_PER_EV_ANGSTROM


def force_kj_mol_nm_to_ev_angstrom(force):
    return np.asarray(force) / KJ_MOL_NM_PER_EV_ANGSTROM


def force_ev_angstrom_to_kj_mol_nm(force):
    return np.asarray(force) * KJ_MOL_NM_PER_EV_ANGSTROM


def force_to_acceleration(force_ev_angstrom, masses_dalton):
    return force_ev_angstrom * (ACCEL_PER_FORCE_PER_DALTON / masses_dalton[..., None])
