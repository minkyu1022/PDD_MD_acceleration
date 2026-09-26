"""Failure-horizon diagnostics and matched eSEN compute measurements."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from .data import load_ad3
from .evaluate import _synchronize
from .model import ESENEnergyGradient
from .teacher import OpenMMTeacher, PhaseState
from .train import load_student
from .units import ACCEL_PER_FORCE_PER_DALTON


def _write(output: str, report: dict):
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return report


def diagnose_rollout(
    data_root: str,
    pdd_checkpoint: str,
    output: str,
    horizons: tuple[int, ...] = (4, 8, 20, 40, 80),
    block: int = 4,
    samples: int = 8,
    max_frames: int | None = None,
    platform: str = "CPU",
    device: str = "cpu",
):
    """Locate the first inaccurate block and check fused-head algebra."""
    if not horizons or samples <= 0 or block <= 0:
        raise ValueError("horizons, samples, and block must be positive")
    horizons = tuple(sorted(set(horizons)))
    if any(h <= 0 or h % block for h in horizons):
        raise ValueError("Each horizon must be positive and divisible by block")
    trajectory = load_ad3(data_root, "test", max_frames)
    student, saved = load_student(pdd_checkpoint, device)
    if block > saved["max_block"]:
        raise ValueError("block exceeds the checkpoint's max_block")
    teacher = OpenMMTeacher(trajectory.pdb_path, platform)
    indices = np.linspace(
        0, len(trajectory) - 1, num=min(samples, len(trajectory)), dtype=int
    )
    rows = []
    fusion = []
    onset = []
    for index in indices:
        initial = PhaseState(*trajectory.state(int(index)))
        reference = teacher.rollout(initial, horizons[-1], saved["dt_ps"])
        q = torch.as_tensor(initial.q[None], dtype=torch.float32, device=device)
        v = torch.as_tensor(initial.v[None], dtype=torch.float32, device=device)
        with torch.no_grad():
            plain_q, plain_v = student.advance(q, v, block)
            fused_q, fused_v = student.advance_fused(q, v, block)
            fusion.append(
                {
                    "initial_index": int(index),
                    "q_max_abs_angstrom": float((plain_q - fused_q).abs().max()),
                    "v_max_abs_angstrom_per_ps": float(
                        (plain_v - fused_v).abs().max()
                    ),
                }
            )
            first_over_0p1 = None
            first_nonfinite = None
            for fine_step in range(block, horizons[-1] + 1, block):
                q, v = student.advance_fused(q, v, block)
                finite = bool(torch.isfinite(q).all() and torch.isfinite(v).all())
                ref = reference[fine_step]
                q_error = (
                    float(np.sqrt(np.mean((q[0].cpu().numpy() - ref.q) ** 2)))
                    if finite
                    else None
                )
                v_error = (
                    float(np.sqrt(np.mean((v[0].cpu().numpy() - ref.v) ** 2)))
                    if finite
                    else None
                )
                if first_over_0p1 is None and (
                    q_error is None or q_error > 0.1
                ):
                    first_over_0p1 = fine_step
                if not finite:
                    first_nonfinite = fine_step
                if fine_step not in horizons:
                    if not finite:
                        break
                    continue
                rows.append(
                    {
                        "initial_index": int(index),
                        "fine_steps": fine_step,
                        "finite": finite,
                        "q_rmse_angstrom": q_error,
                        "v_rmse_angstrom_per_ps": v_error,
                    }
                )
                if not finite:
                    break
            if first_nonfinite is not None:
                for remaining in horizons:
                    if remaining > first_nonfinite:
                        rows.append(
                            {
                                "initial_index": int(index),
                                "fine_steps": remaining,
                                "finite": False,
                                "q_rmse_angstrom": None,
                                "v_rmse_angstrom_per_ps": None,
                            }
                        )
            onset.append(
                {
                    "initial_index": int(index),
                    "first_q_error_over_0p1_angstrom_fine_step": first_over_0p1,
                    "first_nonfinite_fine_step": first_nonfinite,
                }
            )
    summary = {}
    for horizon in horizons:
        selected = [r for r in rows if r["fine_steps"] == horizon]
        q_values = [r["q_rmse_angstrom"] for r in selected if r["finite"]]
        v_values = [r["v_rmse_angstrom_per_ps"] for r in selected if r["finite"]]
        summary[str(horizon)] = {
            "finite_fraction": float(np.mean([r["finite"] for r in selected])),
            "q_rmse_median_angstrom": float(np.median(q_values)) if q_values else None,
            "q_rmse_max_angstrom": float(np.max(q_values)) if q_values else None,
            "v_rmse_median_angstrom_per_ps": (
                float(np.median(v_values)) if v_values else None
            ),
            "q_error_over_0p1_angstrom_fraction": float(
                np.mean(
                    [
                        r["q_rmse_angstrom"] is None
                        or r["q_rmse_angstrom"] > 0.1
                        for r in selected
                    ]
                )
            ),
        }
    return _write(
        output,
        {
            "pdd_checkpoint": pdd_checkpoint,
            "checkpoint_iteration": saved["iteration"],
            "device": device,
            "block": block,
            "dt_ps": saved["dt_ps"],
            "initial_indices": indices.tolist(),
            "fusion_max_q_abs_angstrom": max(
                r["q_max_abs_angstrom"] for r in fusion
            ),
            "fusion_max_v_abs_angstrom_per_ps": max(
                r["v_max_abs_angstrom_per_ps"] for r in fusion
            ),
            "summary": summary,
            "fusion": fusion,
            "failure_onset": onset,
            "rows": rows,
        },
    )


def _energy_verlet(force_model, q, v, masses, dt_ps, fine_steps):
    factor = ACCEL_PER_FORCE_PER_DALTON / masses[None, :, None]
    force = force_model.force(q)
    for _ in range(fine_steps):
        acceleration = force * factor
        half_v = v + 0.5 * dt_ps * acceleration
        q = q + dt_ps * half_v
        force = force_model.force(q)
        v = half_v + 0.5 * dt_ps * force * factor
    return q, v


def benchmark_mlip(
    data_root: str,
    pdd_checkpoint: str,
    output: str,
    block: int = 4,
    fine_steps: int = 8,
    batch_sizes: tuple[int, ...] = (1, 8),
    repeats: int = 3,
    max_frames: int | None = None,
    device: str = "cpu",
):
    """Compare PDD forward compute with eSEN energy-gradient Verlet compute.

    This does not compare trajectory accuracy: the current student learned an
    OpenMM teacher, while the eSEN energy head defines another potential.
    """
    if block <= 0 or fine_steps <= 0 or fine_steps % block or repeats <= 0:
        raise ValueError("block must divide fine_steps; repeats must be positive")
    if not batch_sizes or any(size <= 0 for size in batch_sizes):
        raise ValueError("batch sizes must be positive")
    trajectory = load_ad3(data_root, "test", max_frames)
    student, saved = load_student(pdd_checkpoint, device)
    if saved["backend"] != "esen" or block > saved["max_block"]:
        raise ValueError("Benchmark requires an eSEN PDD checkpoint and valid block")
    energy_model = ESENEnergyGradient(
        saved["atomic_numbers"], saved["checkpoint"]
    ).to(device).eval()
    masses = torch.as_tensor(
        saved["masses_dalton"], dtype=torch.float32, device=device
    )
    results = []
    for size in batch_sizes:
        indices = np.linspace(0, len(trajectory) - 1, num=size, dtype=int)
        q0 = torch.as_tensor(
            trajectory.positions[indices], dtype=torch.float32, device=device
        )
        v0 = torch.as_tensor(
            trajectory.velocities[indices], dtype=torch.float32, device=device
        )
        timings = {"pdd": [], "energy_gradient_verlet": []}
        endpoint_finite = {}
        for repeat in range(repeats + 1):
            for name, elapsed in timings.items():
                _synchronize(device)
                start = time.perf_counter()
                if name == "pdd":
                    with torch.no_grad():
                        q, v = q0, v0
                        for _ in range(fine_steps // block):
                            q, v = student.advance_fused(q, v, block)
                else:
                    q, v = _energy_verlet(
                        energy_model, q0, v0, masses, saved["dt_ps"], fine_steps
                    )
                _synchronize(device)
                if repeat:
                    elapsed.append(time.perf_counter() - start)
                endpoint_finite[name] = bool(
                    torch.isfinite(q).all() and torch.isfinite(v).all()
                )
        pdd_seconds = float(np.median(timings["pdd"]))
        energy_seconds = float(np.median(timings["energy_gradient_verlet"]))
        results.append(
            {
                "batch_size": size,
                "pdd_median_wall_seconds": pdd_seconds,
                "energy_gradient_verlet_median_wall_seconds": energy_seconds,
                "compute_ratio_energy_over_pdd": energy_seconds / pdd_seconds,
                "pdd_backbone_calls": fine_steps // block,
                "energy_backbone_and_gradient_calls": fine_steps + 1,
                "pdd_endpoint_finite": endpoint_finite["pdd"],
                "energy_endpoint_finite": endpoint_finite["energy_gradient_verlet"],
            }
        )
    return _write(
        output,
        {
            "pdd_checkpoint": pdd_checkpoint,
            "energy_checkpoint": saved["checkpoint"],
            "device": device,
            "block": block,
            "fine_steps": fine_steps,
            "repeats": repeats,
            "comparison_scope": "compute_only_different_teacher_potentials",
            "results": results,
        },
    )
