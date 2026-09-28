"""Small, reproducible OC20 DFT-trajectory block prediction experiment.

This is an offline PDD-style architecture check, not an MLIP relaxation speedup.
The official per-adsorbate tar is streamed; its members are never extracted by path.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import lzma
import random
import tarfile
import time
from pathlib import Path

import numpy as np
import torch
from ase.io import read
from torch import nn


def minimum_image(delta: torch.Tensor, cell: torch.Tensor, pbc: torch.Tensor) -> torch.Tensor:
    fractional = delta @ torch.linalg.inv(cell)
    fractional = fractional - torch.round(fractional).detach() * pbc
    return fractional @ cell


def prepare(args: argparse.Namespace) -> None:
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    source = Path(args.tar)
    md5 = hashlib.md5()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            md5.update(chunk)
    expected = "3697f04faf04251a23da8b88a78209f7" if source.name == "H_1.tar" else None
    if expected and md5.hexdigest() != expected:
        raise ValueError("OC20 *H archive MD5 mismatch")

    systems = []
    with tarfile.open(source, "r:gz") as archive:
        for member in archive:
            if not member.isfile() or not member.name.endswith(".extxyz.xz"):
                continue
            sid = Path(member.name).name.removesuffix(".extxyz.xz")
            raw = lzma.decompress(archive.extractfile(member).read())
            frames = read(io.StringIO(raw.decode("utf-8")), index=":", format="extxyz")
            if len(frames) < args.block + 2:
                continue
            atoms = frames[0]
            positions = np.stack([frame.positions for frame in frames]).astype(np.float32)
            forces = np.stack([frame.get_forces(apply_constraint=False) for frame in frames]).astype(np.float32)
            energies = np.array([frame.get_potential_energy() for frame in frames], dtype=np.float32)
            fixed = np.zeros(len(atoms), dtype=bool)
            for constraint in atoms.constraints:
                fixed[np.asarray(constraint.get_indices(), dtype=int)] = True
            path = root / f"{sid}.npz"
            np.savez_compressed(
                path,
                positions=positions,
                forces=forces,
                energies=energies,
                numbers=atoms.numbers.astype(np.int16),
                tags=atoms.get_tags().astype(np.int8),
                free=(~fixed),
                cell=np.array(atoms.cell.array, dtype=np.float32),
                pbc=np.asarray(atoms.pbc, dtype=bool),
            )
            split = "validation" if int(hashlib.sha256(sid.encode()).hexdigest()[:8], 16) % 5 == 0 else "train"
            systems.append({"sid": sid, "path": path.name, "frames": len(frames), "atoms": len(atoms), "split": split})
            if len(systems) >= args.max_systems:
                break
    manifest = {
        "source": str(source),
        "archive_md5": md5.hexdigest(),
        "adsorbate": "*H" if source.name == "H_1.tar" else "unknown",
        "block": args.block,
        "systems": systems,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"systems": len(systems), "train": sum(s["split"] == "train" for s in systems), "validation": sum(s["split"] == "validation" for s in systems), "archive_md5": md5.hexdigest()}))


class EquivariantBlock(nn.Module):
    def __init__(self, block: int, width: int = 32, cutoff: float = 6.0, use_forces: bool = True):
        super().__init__()
        self.block = block
        self.cutoff = cutoff
        self.use_forces = use_forces
        self.element = nn.Embedding(101, 16)
        self.tag = nn.Embedding(4, 4)
        self.edge = nn.Sequential(nn.Linear(50, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU())
        self.node = nn.Sequential(nn.Linear(width + 21, width), nn.SiLU())
        self.head = nn.Sequential(nn.Linear(width * 2 + 8, width), nn.SiLU(), nn.Linear(width, block))
        self.force_head = nn.Linear(width, block)
        self.register_buffer("centers", torch.linspace(0.5, cutoff, 8))

    def forward(self, pos: torch.Tensor, numbers: torch.Tensor, tags: torch.Tensor, free: torch.Tensor, cell: torch.Tensor, pbc: torch.Tensor, force: torch.Tensor) -> torch.Tensor:
        n = len(pos)
        rel = minimum_image(pos[None, :, :] - pos[:, None, :], cell, pbc)
        distance = torch.linalg.vector_norm(rel, dim=-1).clamp_min(1e-6)
        active = ((distance < self.cutoff) & ~torch.eye(n, dtype=torch.bool, device=pos.device)).float()
        rbf = torch.exp(-((distance[..., None] - self.centers) / 0.8) ** 2)
        scaled_force = force / (1 + torch.linalg.vector_norm(force, dim=-1, keepdim=True)) if self.use_forces else torch.zeros_like(force)
        features = torch.cat((self.element(numbers), self.tag(tags.clamp(0, 3)), torch.linalg.vector_norm(scaled_force, dim=-1, keepdim=True)), -1)
        pair = torch.cat((features[:, None, :].expand(-1, n, -1), features[None, :, :].expand(n, -1, -1), rbf), -1)
        edges = self.edge(pair) * active[..., None]
        pooled = edges.sum(1) / active.sum(1).clamp_min(1)[..., None]
        nodes = self.node(torch.cat((features, pooled), -1))
        head_input = torch.cat((nodes[:, None, :].expand(-1, n, -1), nodes[None, :, :].expand(n, -1, -1), rbf), -1)
        coefficients = self.head(head_input) * active[..., None]
        incremental = (coefficients[..., None] * (rel / distance[..., None])[..., None, :]).sum(1)
        incremental = incremental / active.sum(1).clamp_min(1)[..., None, None]
        incremental = incremental + self.force_head(nodes)[..., None] * scaled_force[:, None, :]
        incremental = incremental.permute(1, 0, 2) * free[None, :, None] * 0.1
        return torch.cumsum(incremental, dim=0)


def load_examples(root: Path, block: int, windows_per_system: int, start_mode: str):
    manifest = json.loads((root / "manifest.json").read_text())
    by_split = {"train": [], "validation": []}
    for entry in manifest["systems"]:
        with np.load(root / entry["path"], allow_pickle=False) as data:
            pos = torch.from_numpy(data["positions"].copy())
            forces = torch.from_numpy(data["forces"].copy())
            cell = torch.from_numpy(data["cell"].copy())
            pbc = torch.from_numpy(data["pbc"].astype(np.float32))
            free = torch.from_numpy(data["free"].astype(np.float32))
            valid = len(pos) - block
            if valid < 1:
                continue
            if start_mode == "early":
                starts = np.arange(min(windows_per_system, valid))
            else:
                starts = np.unique(np.linspace(0, valid - 1, min(windows_per_system, valid), dtype=int))
            static = (torch.from_numpy(data["numbers"].astype(np.int64)), torch.from_numpy(data["tags"].astype(np.int64)), free, cell, pbc)
            for start in starts:
                target = minimum_image(pos[start + 1 : start + block + 1] - pos[start], cell, pbc)
                by_split[entry["split"]].append((pos[start], *static, forces[start], target, entry["sid"]))
    return by_split


@torch.no_grad()
def evaluate(model, examples, direct=False):
    model.eval()
    errors = []
    zero_errors = []
    prefix_errors = []
    ads_errors = []
    ads_zero_errors = []
    by_system = {}
    for pos, numbers, tags, free, cell, pbc, force, target, sid in examples:
        pred = model(pos, numbers, tags, free, cell, pbc, force)
        mask = free.bool()
        local_errors = torch.linalg.vector_norm(pred[-1, mask] - target[-1, mask], dim=-1).tolist()
        errors.extend(local_errors)
        by_system.setdefault(sid, []).extend(local_errors)
        zero_errors.extend(torch.linalg.vector_norm(target[-1, mask], dim=-1).tolist())
        ads_mask = mask & (tags == 2)
        ads_errors.extend(torch.linalg.vector_norm(pred[-1, ads_mask] - target[-1, ads_mask], dim=-1).tolist())
        ads_zero_errors.extend(torch.linalg.vector_norm(target[-1, ads_mask], dim=-1).tolist())
        if not direct:
            prefix_errors.extend(torch.linalg.vector_norm(pred[:-1, mask] - target[:-1, mask], dim=-1).flatten().tolist())
    system_errors = [float(np.mean(v)) for v in by_system.values()]
    return {"endpoint_mae_A": float(np.mean(errors)), "endpoint_p90_A": float(np.quantile(errors, 0.9)), "system_endpoint_mae_A": float(np.mean(system_errors)), "system_endpoint_p90_A": float(np.quantile(system_errors, 0.9)), "zero_motion_mae_A": float(np.mean(zero_errors)), "adsorbate_mae_A": float(np.mean(ads_errors)), "adsorbate_zero_mae_A": float(np.mean(ads_zero_errors)), "prefix_mae_A": float(np.mean(prefix_errors)) if prefix_errors else None}


def force_scale_baseline(train_examples, validation_examples, block: int):
    numerator = torch.zeros(block)
    denominator = torch.zeros(block)
    for pos, numbers, tags, free, cell, pbc, force, target, sid in train_examples:
        mask = free[:, None]
        numerator += (force[None] * target * mask).sum((1, 2))
        denominator += ((force * mask) ** 2).sum() * torch.ones(block)
    scale = numerator / denominator.clamp_min(1e-12)
    endpoint_errors, ads_errors = [], []
    for pos, numbers, tags, free, cell, pbc, force, target, sid in validation_examples:
        prediction = scale[-1] * force
        mask = free.bool()
        endpoint_errors.extend(torch.linalg.vector_norm((prediction - target[-1])[mask], dim=-1).tolist())
        ads_errors.extend(torch.linalg.vector_norm((prediction - target[-1])[mask & (tags == 2)], dim=-1).tolist())
    return {"endpoint_mae_A": float(np.mean(endpoint_errors)), "adsorbate_mae_A": float(np.mean(ads_errors)), "fitted_displacement_per_force": scale.tolist()}


def train(args: argparse.Namespace):
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    examples = load_examples(Path(args.data), args.block, args.windows_per_system, args.start_mode)
    if not examples["train"] or not examples["validation"]:
        raise ValueError("Need trajectory-disjoint train and validation examples")
    results = {"config": vars(args), "counts": {k: len(v) for k, v in examples.items()}, "models": {}}
    results["force_scale_baseline"] = force_scale_baseline(examples["train"], examples["validation"], args.block)
    print(json.dumps({"model": "force_scale_baseline", **results["force_scale_baseline"]}), flush=True)
    for name, heads in (("multi_head", args.block), ("direct_endpoint", 1)):
        model = EquivariantBlock(heads, args.width, use_forces=args.use_forces)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        t0 = time.perf_counter()
        losses = []
        for update in range(args.updates):
            model.train()
            pos, numbers, tags, free, cell, pbc, force, target, _ = random.choice(examples["train"])
            pred = model(pos, numbers, tags, free, cell, pbc, force)
            truth = target if heads > 1 else target[-1:]
            weights = free * (1.0 + (tags == 2).float() * args.adsorbate_weight)
            loss = (((pred - truth) / 0.1) ** 2 * weights[None, :, None]).sum() / (weights.sum() * 3 * heads)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            losses.append(float(loss.detach()))
            if (update + 1) % args.log_every == 0:
                print(json.dumps({"model": name, "update": update + 1, "train_loss_recent": float(np.mean(losses[-args.log_every:]))}), flush=True)
        elapsed = time.perf_counter() - t0
        metrics = evaluate(model, examples["validation"], direct=heads == 1)
        metrics["train_seconds"] = elapsed
        metrics["final_train_loss"] = float(np.mean(losses[-min(50, len(losses)):]))
        checkpoint = Path(args.output).with_name(Path(args.output).stem + f"_{name}.pt")
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "block": heads, "width": args.width, "use_forces": args.use_forces}, checkpoint)
        metrics["checkpoint"] = str(checkpoint)
        results["models"][name] = metrics
        print(json.dumps({"model": name, **metrics}), flush=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--tar", required=True)
    prep.add_argument("--output", required=True)
    prep.add_argument("--max-systems", type=int, default=120)
    prep.add_argument("--block", type=int, default=4)
    fit = sub.add_parser("train")
    fit.add_argument("--data", required=True)
    fit.add_argument("--output", required=True)
    fit.add_argument("--block", type=int, default=4)
    fit.add_argument("--updates", type=int, default=300)
    fit.add_argument("--windows-per-system", type=int, default=6)
    fit.add_argument("--start-mode", choices=("early", "uniform"), default="early")
    fit.add_argument("--adsorbate-weight", type=float, default=3.0)
    fit.add_argument("--use-forces", action=argparse.BooleanOptionalAction, default=True)
    fit.add_argument("--width", type=int, default=32)
    fit.add_argument("--learning-rate", type=float, default=0.001)
    fit.add_argument("--threads", type=int, default=4)
    fit.add_argument("--seed", type=int, default=42)
    fit.add_argument("--log-every", type=int, default=50)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
