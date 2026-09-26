"""Held-out AD-3 initial states, teacher trajectories, and runtime metrics."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from .data import load_ad3
from .teacher import OpenMMTeacher, PhaseState
from .train import load_student, validate_teacher


def _student_rollout(
    student, initial: PhaseState, total_steps: int, block: int, device: str
):
    q = torch.tensor(initial.q[None], dtype=torch.float32, device=device)
    v = torch.tensor(initial.v[None], dtype=torch.float32, device=device)
    q_history, v_history = [], []
    if device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(total_steps // block):
            qdots, accels = student(q, v, block)
            qs, vs = student.states_in_block(q, v, qdots, accels)
            q_history.append(qs[:, 1:].detach())
            v_history.append(vs[:, 1:].detach())
            q, v = qs[:, -1], vs[:, -1]
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    all_q = torch.cat(q_history, dim=1)[0].cpu().numpy().astype(np.float64)
    all_v = torch.cat(v_history, dim=1)[0].cpu().numpy().astype(np.float64)
    states = [initial] + [PhaseState(qi, vi) for qi, vi in zip(all_q, all_v)]
    return states, elapsed


def _coarse_verlet_rollout(teacher, initial, total_steps, block, dt_ps):
    states = [initial]
    current = initial
    for _ in range(total_steps // block):
        current = teacher.step(current, block * dt_ps)
        states.append(current)
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
        "test_frames": len(trajectory),
        "test_initial_indices": indices.tolist(),
        "fine_steps": fine_steps,
        "dt_ps": dt_ps,
        "ad3_saved_step_dt_ps": trajectory.dt_ps,
        "teacher_consistency": validate_teacher(trajectory, teacher),
        "results": [],
    }
    for index in indices:
        initial = PhaseState(*trajectory.state(int(index)))
        start_energy = teacher.total_energy(initial)
        reference = teacher.rollout(initial, fine_steps, dt_ps)
        for block in block_sizes:
            states, seconds = _student_rollout(
                student, initial, fine_steps, block, device
            )
            metrics = _compare(states, reference)
            metrics.update(
                {
                    "method": f"pdd_L{block}",
                    "initial_index": int(index),
                    "backbone_evaluations": fine_steps // block,
                    "wall_seconds": seconds,
                    "absolute_energy_drift_kj_mol": float(
                        abs(teacher.total_energy(states[-1]) - start_energy)
                    ),
                }
            )
            report["results"].append(metrics)
            coarse = _coarse_verlet_rollout(teacher, initial, fine_steps, block, dt_ps)
            metrics = _compare(coarse, reference, stride=block)
            metrics.update(
                {
                    "method": f"teacher_coarse_verlet_L{block}",
                    "initial_index": int(index),
                    "force_evaluations": 2 * fine_steps // block,
                    "absolute_energy_drift_kj_mol": float(
                        abs(teacher.total_energy(coarse[-1]) - start_energy)
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
            metrics = _compare(states, reference, stride=block)
            metrics.update(
                {
                    "method": f"direct_L{block}",
                    "initial_index": int(index),
                    "backbone_evaluations": fine_steps // block,
                    "wall_seconds": seconds,
                    "absolute_energy_drift_kj_mol": float(
                        abs(teacher.total_energy(states[-1]) - start_energy)
                    ),
                }
            )
            report["results"].append(metrics)
    summary = {}
    for method in sorted({row["method"] for row in report["results"]}):
        rows = [row for row in report["results"] if row["method"] == method]
        summary[method] = {
            key: float(np.mean([row[key] for row in rows]))
            for key in rows[0]
            if key not in {"method", "initial_index"}
        }
    report["summary"] = summary
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    return report
