"""Held-out AD-3 initial states, teacher trajectories, and runtime metrics."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from .data import load_ad3
from .model import make_adapter
from .teacher import OpenMMTeacher, PhaseState
from .train import _load, load_student, validate_teacher


def evaluate_force(
    data_root: str,
    force_checkpoint: str,
    output: str,
    samples: int = 64,
    max_frames: int | None = None,
    batch_size: int = 16,
    platform: str = "CPU",
    device: str = "cpu",
):
    """Measure warm-start force accuracy against the same teacher on AD-3 test states."""
    if samples <= 0 or batch_size <= 0:
        raise ValueError("samples and batch_size must be positive")
    trajectory = load_ad3(data_root, "test", max_frames)
    teacher = OpenMMTeacher(trajectory.pdb_path, platform)
    saved = _load(force_checkpoint)
    if saved["kind"] != "force":
        raise ValueError("Expected a force warm-start checkpoint")
    adapter = make_adapter(
        saved["backend"],
        saved["atomic_numbers"],
        saved["checkpoint"],
        saved["hidden"],
    ).to(device)
    adapter.load_state_dict(saved["adapter"])
    adapter.eval()
    indices = np.linspace(
        0, len(trajectory) - 1, num=min(samples, len(trajectory)), dtype=int
    )
    predictions = []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            batch = indices[start : start + batch_size]
            q = torch.as_tensor(
                trajectory.positions[batch], dtype=torch.float32, device=device
            )
            predictions.append(adapter(q).cpu().numpy())
    predicted = np.concatenate(predictions).astype(np.float64)
    reference = np.stack([teacher.force(trajectory.positions[i]) for i in indices])
    stored = trajectory.forces[indices].astype(np.float64)
    report = {
        "force_checkpoint": force_checkpoint,
        "device": device,
        "openmm_platform": platform,
        "torch_version": torch.__version__,
        "test_frames": len(trajectory),
        "test_indices": indices.tolist(),
        "force_rmse_teacher_ev_a": float(
            np.sqrt(np.mean((predicted - reference) ** 2))
        ),
        "force_mae_teacher_ev_a": float(np.mean(np.abs(predicted - reference))),
        "ad3_force_rmse_teacher_ev_a": float(
            np.sqrt(np.mean((stored - reference) ** 2))
        ),
    }
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    return report


def _student_rollout(
    student, initial: PhaseState, total_steps: int, block: int, device: str
):
    q = torch.tensor(initial.q[None], dtype=torch.float32, device=device)
    v = torch.tensor(initial.v[None], dtype=torch.float32, device=device)
    q_history, v_history = [q], [v]
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elif device.startswith("mps"):
        torch.mps.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(total_steps // block):
            q, v = student.advance_fused(q, v, block)
            q_history.append(q)
            v_history.append(v)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elif device.startswith("mps"):
        torch.mps.synchronize()
    elapsed = time.perf_counter() - start
    all_q = torch.cat(q_history, dim=0).cpu().numpy().astype(np.float64)
    all_v = torch.cat(v_history, dim=0).cpu().numpy().astype(np.float64)
    states = [PhaseState(qi, vi) for qi, vi in zip(all_q, all_v)]
    return states, elapsed


def _coarse_verlet_rollout(teacher, initial, total_steps, block, dt_ps):
    import openmm

    states = [initial]
    current = initial
    for _ in range(total_steps // block):
        try:
            current = teacher.step(current, block * dt_ps)
        except openmm.OpenMMException:
            current = PhaseState(
                np.full_like(initial.q, np.nan), np.full_like(initial.v, np.nan)
            )
        states.append(current)
        if not _finite_states([current]):
            states.extend([current] * (total_steps // block - len(states) + 1))
            break
    return states


def _compare(student_states, teacher_states, stride=1):
    reference = teacher_states[::stride]
    if len(student_states) != len(reference):
        raise ValueError("Rollout lengths differ")
    qerr = np.asarray(
        [np.sqrt(np.mean((s.q - t.q) ** 2)) for s, t in zip(student_states, reference)]
    )
    verr = np.asarray(
        [np.sqrt(np.mean((s.v - t.v) ** 2)) for s, t in zip(student_states, reference)]
    )
    return {
        "q_endpoint_rmse_angstrom": float(qerr[-1]),
        "v_endpoint_rmse_angstrom_per_ps": float(verr[-1]),
        "q_path_rmse_angstrom": float(np.sqrt(np.mean(qerr**2))),
        "v_path_rmse_angstrom_per_ps": float(np.sqrt(np.mean(verr**2))),
    }


def _finite_states(states):
    return all(
        np.isfinite(state.q).all() and np.isfinite(state.v).all() for state in states
    )


def _energy_drift(teacher, state, start_energy):
    if not np.isfinite(state.q).all() or not np.isfinite(state.v).all():
        return None
    import openmm

    try:
        value = abs(teacher.total_energy(state) - start_energy)
    except openmm.OpenMMException:
        return None
    return float(value) if np.isfinite(value) else None


def _metrics_or_failure(states, reference, teacher, start_energy, stride):
    drift = _energy_drift(teacher, states[-1], start_energy)
    finite = _finite_states(states) and drift is not None
    metrics = (
        _compare(states, reference, stride=stride)
        if finite
        else {
            "q_endpoint_rmse_angstrom": None,
            "v_endpoint_rmse_angstrom_per_ps": None,
            "q_path_rmse_angstrom": None,
            "v_path_rmse_angstrom_per_ps": None,
        }
    )
    metrics.update({"finite": finite, "absolute_energy_drift_kj_mol": drift})
    return metrics


def evaluate(
    data_root: str,
    pdd_checkpoint: str,
    output: str,
    block_sizes=(1, 2, 4, 8),
    direct_checkpoint: str | None = None,
    fine_steps: int = 80,
    samples: int = 8,
    max_frames: int | None = None,
    platform: str = "CPU",
    device: str = "cpu",
):
    if fine_steps <= 0 or samples <= 0:
        raise ValueError("fine_steps and samples must be positive")
    trajectory = load_ad3(data_root, "test", max_frames)
    teacher = OpenMMTeacher(trajectory.pdb_path, platform)
    student, saved = load_student(pdd_checkpoint, device)
    if saved["backend"] == "esen":
        student.eval()
    dt_ps = saved["dt_ps"]
    # Saved AD-3 frames seed trajectories; their stochastic frame spacing need
    # not equal the deterministic NVE teacher integration step.
    for block in block_sizes:
        if block < 1 or block > saved["max_block"] or fine_steps % block:
            raise ValueError(
                "Each block must divide fine_steps and be <= trained max_block"
            )
    direct, direct_saved = (
        load_student(direct_checkpoint, device) if direct_checkpoint else (None, None)
    )
    if direct and not np.isclose(
        direct_saved["dt_ps"], dt_ps * direct_saved["coarse_factor"]
    ):
        raise ValueError("Direct checkpoint coarse dt does not match PDD dt")
    indices = np.linspace(
        0, len(trajectory) - 1, num=min(samples, len(trajectory)), dtype=int
    )
    report = {
        "pdd_checkpoint": pdd_checkpoint,
        "direct_checkpoint": direct_checkpoint,
        "device": device,
        "openmm_platform": platform,
        "torch_version": torch.__version__,
        "test_frames": len(trajectory),
        "test_initial_indices": indices.tolist(),
        "fine_steps": fine_steps,
        "dt_ps": dt_ps,
        "ad3_saved_step_dt_ps": trajectory.dt_ps,
        "path_sampling": "block_endpoints",
        "teacher_consistency": validate_teacher(trajectory, teacher),
        "fine_teacher_wall_seconds": [],
        "fine_teacher_force_calls": [],
        "results": [],
    }
    for index in indices:
        initial = PhaseState(*trajectory.state(int(index)))
        start_energy = teacher.total_energy(initial)
        force_calls_before = teacher.force_calls
        start = time.perf_counter()
        reference = teacher.rollout(initial, fine_steps, dt_ps)
        reference_seconds = time.perf_counter() - start
        report["fine_teacher_wall_seconds"].append(reference_seconds)
        report["fine_teacher_force_calls"].append(
            teacher.force_calls - force_calls_before
        )
        for block in block_sizes:
            states, seconds = _student_rollout(
                student, initial, fine_steps, block, device
            )
            metrics = _metrics_or_failure(
                states, reference, teacher, start_energy, stride=block
            )
            metrics.update(
                {
                    "method": f"pdd_L{block}",
                    "initial_index": int(index),
                    "backbone_evaluations": fine_steps // block,
                    "wall_seconds": seconds,
                    "speedup_vs_fine_teacher": (
                        reference_seconds / seconds if metrics["finite"] else None
                    ),
                }
            )
            report["results"].append(metrics)
            force_calls_before = teacher.force_calls
            start = time.perf_counter()
            coarse = _coarse_verlet_rollout(teacher, initial, fine_steps, block, dt_ps)
            coarse_seconds = time.perf_counter() - start
            metrics = _metrics_or_failure(
                coarse, reference, teacher, start_energy, stride=block
            )
            metrics.update(
                {
                    "method": f"teacher_coarse_verlet_L{block}",
                    "initial_index": int(index),
                    "force_evaluations": teacher.force_calls - force_calls_before,
                    "wall_seconds": coarse_seconds,
                    "speedup_vs_fine_teacher": (
                        reference_seconds / coarse_seconds
                        if metrics["finite"]
                        else None
                    ),
                }
            )
            report["results"].append(metrics)
        if direct:
            block = direct_saved["coarse_factor"]
            if fine_steps % block:
                raise ValueError("fine_steps must be divisible by direct coarse factor")
            states, seconds = _student_rollout(
                direct, initial, fine_steps // block, 1, device
            )
            metrics = _metrics_or_failure(
                states, reference, teacher, start_energy, stride=block
            )
            metrics.update(
                {
                    "method": f"direct_L{block}",
                    "initial_index": int(index),
                    "backbone_evaluations": fine_steps // block,
                    "wall_seconds": seconds,
                    "speedup_vs_fine_teacher": (
                        reference_seconds / seconds if metrics["finite"] else None
                    ),
                }
            )
            report["results"].append(metrics)
    summary = {}
    for method in sorted({row["method"] for row in report["results"]}):
        rows = [row for row in report["results"] if row["method"] == method]
        summary[method] = {
            "finite_fraction": float(np.mean([r["finite"] for r in rows]))
        }
        for key in rows[0]:
            if key in {"method", "initial_index", "finite"}:
                continue
            values = [row[key] for row in rows if row[key] is not None]
            summary[method][key] = float(np.mean(values)) if values else None
    report["summary"] = summary
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    return report
