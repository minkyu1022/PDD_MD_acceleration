"""Evaluate OC20 DFT-trained proposals against a fixed pretrained MLIP relaxer.

Run in a separate fairchem-core 1.10 environment with the public GemNet-OC-2M
checkpoint. This deliberately measures total proposal, guard, and refinement cost.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import ClassVar

import numpy as np
import torch
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.constraints import FixAtoms
from ase.optimize import FIRE, LBFGS
from fairchem.core import OCPCalculator
from relaxation_poc import EquivariantBlock


class CountingCalculator(Calculator):
    implemented_properties: ClassVar[list[str]] = ["energy", "forces"]

    def __init__(self, underlying):
        super().__init__()
        self.underlying = underlying
        self.calls = 0
        self.compute_seconds = 0.0

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=None):
        super().calculate(atoms, properties, system_changes)
        start = time.perf_counter()
        self.underlying.calculate(atoms, properties, system_changes)
        self.compute_seconds += time.perf_counter() - start
        self.calls += 1
        self.results = dict(self.underlying.results)


def load_student(path: str) -> EquivariantBlock:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = EquivariantBlock(checkpoint["block"], checkpoint["width"], use_forces=checkpoint["use_forces"])
    model.load_state_dict(checkpoint["model"])
    return model.eval()


def move(atoms: Atoms, model: EquivariantBlock, force: np.ndarray, free: np.ndarray) -> float:
    with torch.no_grad():
        delta = model(
            torch.as_tensor(atoms.positions, dtype=torch.float32),
            torch.as_tensor(atoms.numbers, dtype=torch.long),
            torch.as_tensor(atoms.get_tags(), dtype=torch.long),
            torch.as_tensor(free, dtype=torch.float32),
            torch.as_tensor(atoms.cell.array, dtype=torch.float32),
            torch.as_tensor(atoms.pbc, dtype=torch.float32),
            torch.as_tensor(force, dtype=torch.float32),
        )[-1].numpy()
    max_move = float(np.max(np.linalg.norm(delta[free], axis=-1)))
    atoms.positions += delta
    return max_move


def run_method(entry, root: Path, underlying, method: str, student, fmax: float, max_steps: int, guard_ev: float, proposal_blocks: int = 1, proposal_force_source: str = "mlip", optimizer_name: str = "lbfgs", lbfgs_maxstep: float = 0.2, lbfgs_memory: int = 100, lbfgs_alpha: float = 70.0):
    with np.load(root / entry["path"], allow_pickle=False) as data:
        free = data["free"].copy()
        dft_force = data["forces"][0].copy()
        dft_endpoint = data["positions"][-1].copy()
        atoms = Atoms(numbers=data["numbers"], positions=data["positions"][0], cell=data["cell"], pbc=data["pbc"], tags=data["tags"])
        atoms.set_constraint(FixAtoms(mask=~free))
    calculator = CountingCalculator(underlying)
    atoms.calc = calculator
    start = time.perf_counter()
    initial_force = atoms.get_forces(apply_constraint=False)
    initial_energy = atoms.get_potential_energy()
    proposal_move = 0.0
    proposal_energy = None
    rejected = False
    accepted_blocks = 0
    if student is not None:
        current_force = dft_force if proposal_force_source == "dft" else initial_force
        current_energy = initial_energy
        for _ in range(proposal_blocks):
            if np.max(np.linalg.norm(current_force[free], axis=-1)) < fmax:
                break
            original = atoms.positions.copy()
            proposal_move = move(atoms, student, current_force, free)
            if proposal_move > 1.0 or not np.isfinite(atoms.positions).all():
                rejected = True
            else:
                proposal_energy = float(atoms.get_potential_energy())
                if proposal_energy > current_energy + guard_ev:
                    rejected = True
            if rejected:
                atoms.positions = original
                break
            accepted_blocks += 1
            current_energy = proposal_energy
            current_force = atoms.get_forces(apply_constraint=False)
    if optimizer_name == "lbfgs":
        optimizer = LBFGS(atoms, logfile=None, maxstep=lbfgs_maxstep, memory=lbfgs_memory, alpha=lbfgs_alpha)
    elif optimizer_name == "fire":
        optimizer = FIRE(atoms, logfile=None)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    optimizer.run(fmax=fmax, steps=max_steps)
    final_force = atoms.get_forces()
    final_fmax = float(np.max(np.linalg.norm(final_force[free], axis=-1)))
    final_energy = float(atoms.get_potential_energy())
    elapsed = time.perf_counter() - start
    delta = atoms.positions - dft_endpoint
    frac = delta @ np.linalg.inv(atoms.cell.array)
    frac -= np.round(frac) * atoms.pbc
    delta = frac @ atoms.cell.array
    tags = atoms.get_tags()
    dft_force_mae = float(np.mean(np.abs(initial_force[free] - dft_force[free])))
    return {
        "sid": entry["sid"],
        "method": method,
        "optimizer": optimizer_name,
        "proposal_force_source": proposal_force_source if student is not None else None,
        "atoms": len(atoms),
        "free_atoms": int(free.sum()),
        "dft_initial_force_mae_ev_A": dft_force_mae,
        "initial_energy_ev": float(initial_energy),
        "proposal_max_move_A": proposal_move,
        "proposal_energy_ev": proposal_energy,
        "proposal_rejected": rejected,
        "accepted_blocks": accepted_blocks,
        "converged": final_fmax < fmax,
        "optimizer_steps": optimizer.nsteps,
        "mlip_calls": calculator.calls,
        "mlip_compute_seconds": calculator.compute_seconds,
        "wall_seconds": elapsed,
        "final_fmax_ev_A": final_fmax,
        "final_energy_ev": final_energy,
        "final_dft_free_mae_A": float(np.mean(np.linalg.norm(delta[free], axis=-1))),
        "final_dft_adsorbate_mae_A": float(np.mean(np.linalg.norm(delta[tags == 2], axis=-1))),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--multi-head", required=True)
    parser.add_argument("--direct-endpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--systems", type=int, default=8)
    parser.add_argument("--fmax", type=float, default=0.05)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--guard-ev", type=float, default=0.1)
    parser.add_argument("--proposal-blocks", type=int, default=1)
    parser.add_argument("--proposal-force-source", choices=("mlip", "dft"), default="mlip")
    parser.add_argument("--optimizer", choices=("lbfgs", "fire"), default="lbfgs")
    parser.add_argument("--lbfgs-maxstep", type=float, default=0.2)
    parser.add_argument("--lbfgs-memory", type=int, default=100)
    parser.add_argument("--lbfgs-alpha", type=float, default=70.0)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    root = Path(args.data)
    manifest = json.loads((root / "manifest.json").read_text())
    selected = [e for e in manifest["systems"] if e["split"] == "validation"][: args.systems]
    if len(selected) < args.systems:
        raise ValueError("Not enough validation systems")
    label = args.optimizer.upper()
    models = {label: None, f"PDD4+{label}": load_student(args.multi_head), f"direct4+{label}": load_student(args.direct_endpoint)}
    underlying = OCPCalculator(checkpoint_path=args.checkpoint, cpu=True, seed=42)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for entry in selected:
        for name, model in models.items():
            result = run_method(entry, root, underlying, name, model, args.fmax, args.max_steps, args.guard_ev, args.proposal_blocks, args.proposal_force_source, args.optimizer, args.lbfgs_maxstep, args.lbfgs_memory, args.lbfgs_alpha)
            rows.append(result)
            print(json.dumps(result), flush=True)
            temporary = output.with_suffix(output.suffix + ".tmp")
            temporary.write_text(json.dumps({"config": vars(args), "rows": rows}, indent=2) + "\n")
            temporary.replace(output)


if __name__ == "__main__":
    main()
