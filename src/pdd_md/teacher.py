"""Timewarp AD-3 AMBER14/OBC1 potential and deterministic NVE step map."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .units import (
    ACCEL_PER_FORCE_PER_DALTON,
    NM_TO_ANGSTROM,
    force_kj_mol_nm_to_ev_angstrom,
)


@dataclass(frozen=True)
class PhaseState:
    q: np.ndarray  # Å
    v: np.ndarray  # Å/ps


class OpenMMTeacher:
    """One reusable OpenMM Context. Instances are not thread safe.

    The force field and cutoff reproduce the Timewarp amber14-implicit preset.
    Velocity Verlet is deliberately deterministic; it defines the PoC teacher
    map from a state. It is not the original stochastic Langevin integrator.
    """

    def __init__(self, pdb_path: str | Path, platform: str = "CPU"):
        import openmm as mm
        from openmm import app, unit

        pdb = app.PDBFile(str(pdb_path))
        forcefield = app.ForceField("amber14-all.xml", "implicit/obc1.xml")
        modeller = app.Modeller(pdb.topology, pdb.positions)
        modeller.addExtraParticles(forcefield)
        if len(list(modeller.topology.atoms())) != len(list(pdb.topology.atoms())):
            raise ValueError(
                "Force field added virtual sites; AD-3 state atom count would change"
            )
        system = forcefield.createSystem(
            modeller.topology,
            nonbondedMethod=app.CutoffNonPeriodic,
            nonbondedCutoff=2.0 * unit.nanometer,
            constraints=None,
        )
        self.masses = np.asarray(
            [
                system.getParticleMass(i).value_in_unit(unit.dalton)
                for i in range(system.getNumParticles())
            ],
            dtype=np.float64,
        )
        self.natoms = len(self.masses)
        if np.any(self.masses <= 0):
            raise ValueError("Nonpositive mass in AD-3 OpenMM system")
        self.system = system
        self.integrator = mm.VerletIntegrator(0.0005 * unit.picoseconds)
        self.context = mm.Context(
            system, self.integrator, mm.Platform.getPlatformByName(platform)
        )
        self.unit = unit
        self.force_calls = 0

    def force_and_potential(self, q_angstrom: np.ndarray):
        q = np.asarray(q_angstrom, dtype=np.float64)
        if q.shape != (self.natoms, 3):
            raise ValueError(f"Expected positions [{self.natoms},3], got {q.shape}")
        self.context.setPositions(q / NM_TO_ANGSTROM * self.unit.nanometer)
        state = self.context.getState(getForces=True, getEnergy=True)
        force = state.getForces(asNumpy=True).value_in_unit(
            self.unit.kilojoule_per_mole / self.unit.nanometer
        )
        potential = state.getPotentialEnergy().value_in_unit(
            self.unit.kilojoule_per_mole
        )
        self.force_calls += 1
        return np.asarray(
            force_kj_mol_nm_to_ev_angstrom(force), dtype=np.float64
        ), float(potential)

    def force(self, q_angstrom: np.ndarray):
        return self.force_and_potential(q_angstrom)[0]

    def acceleration(self, q_angstrom: np.ndarray):
        return self.force(q_angstrom) * (
            ACCEL_PER_FORCE_PER_DALTON / self.masses[:, None]
        )

    def step(self, state: PhaseState, dt_ps: float) -> PhaseState:
        """Velocity Verlet step. All teacher targets use this same map."""
        if dt_ps <= 0:
            raise ValueError("dt_ps must be positive")
        q = np.asarray(state.q, dtype=np.float64)
        v = np.asarray(state.v, dtype=np.float64)
        a0 = self.acceleration(q)
        v_half = v + 0.5 * dt_ps * a0
        q1 = q + dt_ps * v_half
        a1 = self.acceleration(q1)
        v1 = v_half + 0.5 * dt_ps * a1
        return PhaseState(q1, v1)

    def mean_velocity(self, state: PhaseState, dt_ps: float):
        next_state = self.step(state, dt_ps)
        return (next_state.q - state.q) / dt_ps, (next_state.v - state.v) / dt_ps

    def rollout(self, state: PhaseState, steps: int, dt_ps: float):
        states = [
            PhaseState(
                np.asarray(state.q, dtype=np.float64).copy(),
                np.asarray(state.v, dtype=np.float64).copy(),
            )
        ]
        for _ in range(steps):
            states.append(self.step(states[-1], dt_ps))
        return states

    def total_energy(self, state: PhaseState):
        _, potential = self.force_and_potential(state.q)
        v_nm_ps = np.asarray(state.v) / NM_TO_ANGSTROM
        kinetic = 0.5 * np.sum(self.masses[:, None] * v_nm_ps**2)
        return potential + kinetic

    def close(self):
        del self.context
        del self.integrator
