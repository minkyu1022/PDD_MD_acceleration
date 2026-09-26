"""Force warm start and on-policy parallel decoding distillation."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch

from .data import load_ad3, pdb_atomic_numbers_and_masses
from .model import ParallelStudent, make_adapter
from .teacher import OpenMMTeacher, PhaseState


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _tensor(a, device):
    return torch.as_tensor(np.asarray(a), dtype=torch.float32, device=device)


def _batch_states(trajectory, indices, device):
    return _tensor(trajectory.positions[indices], device), _tensor(
        trajectory.velocities[indices], device
    )


def _sample_indices(rng, nframes: int, batch_size: int):
    return rng.integers(0, nframes, size=batch_size)


def _save(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    torch.save(payload, temporary)
    temporary.replace(path)


def _load(path):
    # Checkpoints here are produced locally and include metadata dictionaries.
    return torch.load(path, map_location="cpu", weights_only=False)


def _restore_optimizer(optimizer, saved, device):
    optimizer.load_state_dict(saved["optimizer"])
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def _resume_training(resume, kind, model, optimizer, rng, device, expected=None):
    if resume is None:
        return 1, []
    saved = _load(resume)
    if saved["kind"] != kind:
        raise ValueError(f"Cannot resume {kind} from {saved['kind']} checkpoint")
    for key, value in (expected or {}).items():
        if saved[key] != value:
            raise ValueError(f"Resume mismatch for {key}: {saved[key]} != {value}")
    key = "adapter" if kind == "force" else "student"
    model.load_state_dict(saved[key])
    _restore_optimizer(optimizer, saved, device)
    rng.bit_generator.state = saved["rng_state"]
    return int(saved["iteration"]) + 1, list(saved["history"])


def _topology(trajectory):
    atomic_numbers, masses = pdb_atomic_numbers_and_masses(trajectory.pdb_path)
    if len(atomic_numbers) != trajectory.positions.shape[1]:
        raise ValueError("PDB and NPZ atom count mismatch")
    return atomic_numbers.tolist(), masses


def validate_teacher(trajectory, teacher, count=8, seed=0):
    rng = np.random.default_rng(seed)
    picks = _sample_indices(rng, len(trajectory), min(count, len(trajectory)))
    errors, norms = [], []
    for index in picks:
        target = trajectory.forces[index]
        predicted = teacher.force(trajectory.positions[index])
        errors.append(np.mean(np.abs(predicted - target)))
        norms.append(np.sqrt(np.mean(target**2)))
    return {
        "force_mae_ev_a": float(np.mean(errors)),
        "reference_force_rms_ev_a": float(np.mean(norms)),
        "sampled_frames": len(picks),
    }


def train_force(
    data_root: str,
    output: str,
    backend: str = "esen",
    checkpoint: str = "esen-sm-direct-all-omol",
    steps: int = 1000,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    max_frames: int | None = None,
    label_source: str = "teacher",
    platform: str = "CPU",
    device: str = "cpu",
    seed: int = 7,
    hidden: int = 32,
    log_every: int = 50,
    save_every: int = 500,
    resume: str | None = None,
):
    if label_source not in {"teacher", "ad3"}:
        raise ValueError("label_source must be teacher or ad3")
    set_seed(seed)
    trajectory = load_ad3(data_root, "train", max_frames)
    atomic_numbers, masses = _topology(trajectory)
    teacher = OpenMMTeacher(trajectory.pdb_path, platform)
    consistency = validate_teacher(trajectory, teacher)
    print("teacher_consistency", json.dumps(consistency), flush=True)
    adapter = make_adapter(backend, atomic_numbers, checkpoint, hidden).to(device)
    adapter.train()
    optimizer = torch.optim.AdamW(
        adapter.parameters(), lr=learning_rate, weight_decay=1e-6
    )
    rng = np.random.default_rng(seed)
    start_step, history = _resume_training(
        resume,
        "force",
        adapter,
        optimizer,
        rng,
        device,
        {
            "backend": backend,
            "atomic_numbers": atomic_numbers,
            "label_source": label_source,
        },
    )

    def snapshot(iteration):
        return {
            "kind": "force",
            "backend": backend,
            "checkpoint": checkpoint,
            "hidden": hidden,
            "atomic_numbers": atomic_numbers,
            "masses_dalton": masses.tolist(),
            "adapter": adapter.state_dict(),
            "optimizer": optimizer.state_dict(),
            "rng_state": rng.bit_generator.state,
            "iteration": iteration,
            "label_source": label_source,
            "teacher_consistency": consistency,
            "history": history,
            "seed": seed,
        }

    for iteration in range(start_step, steps + 1):
        indices = _sample_indices(rng, len(trajectory), batch_size)
        q, _ = _batch_states(trajectory, indices, device)
        if label_source == "teacher":
            force = np.stack([teacher.force(trajectory.positions[i]) for i in indices])
        else:
            force = trajectory.forces[indices]
        target = _tensor(force, device)
        prediction = adapter(q)
        loss = ((prediction - target) ** 2).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 10.0)
        optimizer.step()
        if iteration == 1 or iteration % log_every == 0 or iteration == steps:
            record = {
                "iteration": iteration,
                "force_rmse_ev_a": float(loss.sqrt().detach().cpu()),
            }
            history.append(record)
            print(json.dumps(record), flush=True)
        if save_every and iteration % save_every == 0:
            _save(output, snapshot(iteration))
    payload = snapshot(max(steps, start_step - 1))
    _save(output, payload)
    return payload


def _student_from_force(force_checkpoint, max_block, dt_ps, device):
    warm = _load(force_checkpoint)
    if warm["kind"] != "force":
        raise ValueError("Expected a force warm-start checkpoint")
    adapter = make_adapter(
        warm["backend"], warm["atomic_numbers"], warm["checkpoint"], warm["hidden"]
    )
    adapter.load_state_dict(warm["adapter"])
    student = ParallelStudent(adapter, warm["masses_dalton"], max_block, dt_ps).to(
        device
    )
    return student, warm


def _teacher_mean_batch(teacher, q, v, dt_ps):
    q_np, v_np = q.detach().cpu().numpy(), v.detach().cpu().numpy()
    means = [
        teacher.mean_velocity(PhaseState(q_np[i], v_np[i]), dt_ps)
        for i in range(len(q_np))
    ]
    qdot = np.stack([pair[0] for pair in means])
    accel = np.stack([pair[1] for pair in means])
    return _tensor(qdot, q.device), _tensor(accel, q.device)


def _teacher_coarse_mean_batch(teacher, q, v, fine_dt_ps, fine_steps):
    q_np, v_np = q.detach().cpu().numpy(), v.detach().cpu().numpy()
    qdots, accels = [], []
    coarse_dt = fine_dt_ps * fine_steps
    for start_q, start_v in zip(q_np, v_np):
        state = PhaseState(start_q, start_v)
        final = teacher.rollout(state, fine_steps, fine_dt_ps)[-1]
        qdots.append((final.q - state.q) / coarse_dt)
        accels.append((final.v - state.v) / coarse_dt)
    return _tensor(np.stack(qdots), q.device), _tensor(np.stack(accels), q.device)


def _scaled_loss(qdot, accel, target_qdot, target_accel, velocity_scale, accel_scale):
    return (((qdot - target_qdot) / velocity_scale) ** 2).mean() + (
        ((accel - target_accel) / accel_scale) ** 2
    ).mean()


def train_pdd(
    data_root: str,
    force_checkpoint: str,
    output: str,
    steps: int = 2000,
    batch_size: int = 4,
    max_block: int = 8,
    block_sizes: tuple[int, ...] = (1, 2, 4, 8),
    dt_ps: float | None = None,
    learning_rate: float = 2e-5,
    max_frames: int | None = None,
    prefix_blocks: int = 0,
    velocity_scale: float = 10.0,
    accel_scale: float = 10000.0,
    platform: str = "CPU",
    device: str = "cpu",
    seed: int = 7,
    log_every: int = 50,
    save_every: int = 500,
    resume: str | None = None,
):
    set_seed(seed)
    trajectory = load_ad3(data_root, "train", max_frames)
    # AD-3 provides initial states; the deterministic teacher step is chosen
    # independently of its stored (stochastic) frame spacing.
    dt_ps = 0.0005 if dt_ps is None else dt_ps
    if any(not 1 <= x <= max_block for x in block_sizes):
        raise ValueError("block sizes must be between one and max_block")
    teacher = OpenMMTeacher(trajectory.pdb_path, platform)
    student, warm = _student_from_force(force_checkpoint, max_block, dt_ps, device)
    if list(warm["atomic_numbers"]) != _topology(trajectory)[0]:
        raise ValueError("Warm-start checkpoint and AD-3 topology differ")
    student.train()
    optimizer = torch.optim.AdamW(
        student.parameters(), lr=learning_rate, weight_decay=1e-6
    )
    rng = np.random.default_rng(seed)
    start_step, history = _resume_training(
        resume,
        "pdd",
        student,
        optimizer,
        rng,
        device,
        {"max_block": max_block, "dt_ps": dt_ps, "block_sizes": list(block_sizes)},
    )

    def snapshot(iteration):
        return {
            "kind": "pdd",
            "backend": warm["backend"],
            "checkpoint": warm["checkpoint"],
            "hidden": warm["hidden"],
            "atomic_numbers": warm["atomic_numbers"],
            "masses_dalton": warm["masses_dalton"],
            "student": student.state_dict(),
            "optimizer": optimizer.state_dict(),
            "rng_state": rng.bit_generator.state,
            "iteration": iteration,
            "max_block": max_block,
            "dt_ps": dt_ps,
            "block_sizes": list(block_sizes),
            "prefix_blocks": prefix_blocks,
            "velocity_scale": velocity_scale,
            "accel_scale": accel_scale,
            "force_checkpoint": str(force_checkpoint),
            "history": history,
            "seed": seed,
        }

    for iteration in range(start_step, steps + 1):
        indices = _sample_indices(rng, len(trajectory), batch_size)
        q, v = _batch_states(trajectory, indices, device)
        block = int(rng.choice(block_sizes))
        if prefix_blocks:
            prefix = int(rng.integers(0, prefix_blocks + 1))
            with torch.no_grad():
                for _ in range(prefix):
                    q, v = student.advance(q, v, block)
            q, v = q.detach(), v.detach()
        qdots, accels = student(q, v, block)
        qs, vs = student.states_in_block(q, v, qdots, accels)
        offsets = rng.integers(0, block, size=batch_size)
        rows = torch.arange(batch_size, device=device)
        selected = torch.as_tensor(offsets, dtype=torch.long, device=device)
        target_qdot, target_accel = _teacher_mean_batch(
            teacher, qs[rows, selected], vs[rows, selected], dt_ps
        )
        loss = _scaled_loss(
            qdots[rows, selected],
            accels[rows, selected],
            target_qdot,
            target_accel,
            velocity_scale,
            accel_scale,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 10.0)
        optimizer.step()
        if iteration == 1 or iteration % log_every == 0 or iteration == steps:
            record = {
                "iteration": iteration,
                "block": block,
                "loss": float(loss.detach().cpu()),
                "teacher_force_calls": teacher.force_calls,
            }
            history.append(record)
            print(json.dumps(record), flush=True)
        if save_every and iteration % save_every == 0:
            _save(output, snapshot(iteration))
    payload = snapshot(max(steps, start_step - 1))
    _save(output, payload)
    return payload


def train_direct(
    data_root: str,
    force_checkpoint: str,
    output: str,
    coarse_factor: int = 8,
    dt_ps: float = 0.0005,
    steps: int = 2000,
    batch_size: int = 4,
    learning_rate: float = 2e-5,
    max_frames: int | None = None,
    velocity_scale: float = 10.0,
    accel_scale: float = 10000.0,
    platform: str = "CPU",
    device: str = "cpu",
    seed: int = 7,
    log_every: int = 50,
    save_every: int = 500,
    resume: str | None = None,
):
    """Equal-backbone one-forward coarse transition comparison model."""
    set_seed(seed)
    trajectory = load_ad3(data_root, "train", max_frames)
    fine_dt = dt_ps
    teacher = OpenMMTeacher(trajectory.pdb_path, platform)
    student, warm = _student_from_force(
        force_checkpoint, 1, fine_dt * coarse_factor, device
    )
    student.train()
    optimizer = torch.optim.AdamW(
        student.parameters(), lr=learning_rate, weight_decay=1e-6
    )
    rng = np.random.default_rng(seed)
    start_step, history = _resume_training(
        resume,
        "direct",
        student,
        optimizer,
        rng,
        device,
        {"coarse_factor": coarse_factor, "dt_ps": fine_dt * coarse_factor},
    )

    def snapshot(iteration):
        return {
            "kind": "direct",
            "backend": warm["backend"],
            "checkpoint": warm["checkpoint"],
            "hidden": warm["hidden"],
            "atomic_numbers": warm["atomic_numbers"],
            "masses_dalton": warm["masses_dalton"],
            "student": student.state_dict(),
            "optimizer": optimizer.state_dict(),
            "rng_state": rng.bit_generator.state,
            "iteration": iteration,
            "max_block": 1,
            "dt_ps": fine_dt * coarse_factor,
            "coarse_factor": coarse_factor,
            "history": history,
            "seed": seed,
        }

    for iteration in range(start_step, steps + 1):
        indices = _sample_indices(rng, len(trajectory), batch_size)
        q, v = _batch_states(trajectory, indices, device)
        target_qdot, target_accel = _teacher_coarse_mean_batch(
            teacher, q, v, fine_dt, coarse_factor
        )
        qdots, accels = student(q, v, 1)
        loss = _scaled_loss(
            qdots[:, 0],
            accels[:, 0],
            target_qdot,
            target_accel,
            velocity_scale,
            accel_scale,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 10.0)
        optimizer.step()
        if iteration == 1 or iteration % log_every == 0 or iteration == steps:
            record = {
                "iteration": iteration,
                "loss": float(loss.detach().cpu()),
                "teacher_force_calls": teacher.force_calls,
            }
            history.append(record)
            print(json.dumps(record), flush=True)
        if save_every and iteration % save_every == 0:
            _save(output, snapshot(iteration))
    payload = snapshot(max(steps, start_step - 1))
    _save(output, payload)
    return payload


def load_student(checkpoint_path: str, device="cpu"):
    saved = _load(checkpoint_path)
    if saved["kind"] not in {"pdd", "direct"}:
        raise ValueError("Expected PDD or direct transition checkpoint")
    adapter = make_adapter(
        saved["backend"], saved["atomic_numbers"], saved["checkpoint"], saved["hidden"]
    )
    student = ParallelStudent(
        adapter, saved["masses_dalton"], saved["max_block"], saved["dt_ps"]
    )
    student.load_state_dict(saved["student"])
    return student.to(device).eval(), saved
