"""A shared equivariant backbone with parallel per-interval vector heads."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from .units import ACCEL_PER_FORCE_PER_DALTON


@dataclass
class Encoded:
    data: object
    embedding: object
    scalars: torch.Tensor  # [B,A,C]


class ESENAdapter(nn.Module):
    """Direct-force eSEN checkpoint; one backbone evaluation per block.

    This uses FAIR-Chem's model loader and its own on-the-fly graph construction.
    The gated checkpoint is downloaded by FAIR-Chem after the user has accepted
    the model license and authenticated with Hugging Face.
    """

    def __init__(
        self, atomic_numbers: list[int], checkpoint: str = "esen-sm-direct-all-omol"
    ):
        super().__init__()
        try:
            from fairchem.core.calculate import pretrained_mlip
            from fairchem.core.units.mlip_unit.utils import load_inference_model
        except ImportError as exc:
            raise RuntimeError(
                "Install the eSEN extra: pip install -e '.[esen]'"
            ) from exc

        if Path(checkpoint).is_file():
            checkpoint_path = checkpoint
        else:
            checkpoint_path = pretrained_mlip.pretrained_checkpoint_path_from_name(
                checkpoint
            )
        loaded, ckpt = load_inference_model(checkpoint_path, use_ema=True)
        model = loaded.module if hasattr(loaded, "module") else loaded
        model.setup_tasks(ckpt.tasks_config)
        if not hasattr(model, "backbone") or not hasattr(model, "output_heads"):
            raise TypeError("The eSEN checkpoint is not a Hydra backbone + heads model")
        if "forces" not in model.output_heads:
            raise KeyError(
                f"No direct forces head in eSEN checkpoint; heads={list(model.output_heads)}"
            )
        self.backbone = model.backbone
        # AD-3 is nonperiodic even if the training checkpoint used periodic
        # graph generation for other datasets.
        self.backbone.always_use_pbc = False
        self.force_head = model.output_heads["forces"]
        # FAIR-Chem trains in normalized target units and denormalizes in its
        # prediction unit, outside HydraModel.forward. Bypassing that step
        # would silently give incorrect eV/Å forces and MD accelerations.
        force_tasks = [
            task
            for task in model.tasks.values()
            if task.property == "forces" and "omol" in task.datasets
        ]
        if len(force_tasks) != 1:
            raise ValueError(f"Expected one OMol force task, found {len(force_tasks)}")
        self.force_normalizer = deepcopy(force_tasks[0].normalizer)
        if not hasattr(self.force_head, "linear"):
            raise TypeError(
                "Expected FAIR-Chem Linear_Force_Head; checkpoint head layout changed"
            )
        self.register_buffer(
            "atomic_numbers", torch.tensor(atomic_numbers, dtype=torch.long)
        )
        self.scalar_dim = int(self.backbone.sphere_channels)

    def _atomic_data(self, q: torch.Tensor):
        from fairchem.core.datasets.atomic_data import AtomicData

        batch_size, atoms, _ = q.shape
        if atoms != len(self.atomic_numbers):
            raise ValueError(
                "Atom count differs from the eSEN checkpoint input topology"
            )
        device = q.device
        if self.backbone.otf_graph:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
            nedges = torch.zeros(batch_size, dtype=torch.long, device=device)
        else:
            # Some checkpoints expect external edges. Build the exact small
            # nonperiodic neighbor graph from the current student positions.
            distance = torch.cdist(q, q)
            mask = distance < float(self.backbone.cutoff)
            diagonal = torch.eye(atoms, dtype=torch.bool, device=device)[None]
            mask = mask & ~diagonal
            max_neighbors = int(self.backbone.max_neighbors)
            if max_neighbors < atoms - 1:
                nearest = (
                    distance.masked_fill(~mask, float("inf"))
                    .topk(max_neighbors, dim=-1, largest=False)
                    .indices
                )
                keep = torch.zeros_like(mask)
                keep.scatter_(-1, nearest, True)
                mask &= keep
            batch_id, center, neighbor = mask.nonzero(as_tuple=True)
            edge_index = torch.stack(
                [batch_id * atoms + center, batch_id * atoms + neighbor]
            )
            nedges = torch.bincount(batch_id, minlength=batch_size)
        cell_offsets = torch.zeros(
            (edge_index.shape[1], 3), dtype=q.dtype, device=device
        )
        return AtomicData(
            pos=q.reshape(-1, 3),
            atomic_numbers=self.atomic_numbers.repeat(batch_size),
            cell=torch.zeros((batch_size, 3, 3), dtype=q.dtype, device=device),
            pbc=torch.zeros((batch_size, 3), dtype=torch.bool, device=device),
            natoms=torch.full((batch_size,), atoms, dtype=torch.long, device=device),
            edge_index=edge_index,
            cell_offsets=cell_offsets,
            nedges=nedges,
            charge=torch.zeros(batch_size, dtype=torch.long, device=device),
            spin=torch.ones(batch_size, dtype=torch.long, device=device),
            fixed=torch.zeros(batch_size * atoms, dtype=torch.long, device=device),
            tags=torch.zeros(batch_size * atoms, dtype=torch.long, device=device),
            batch=torch.arange(batch_size, device=device).repeat_interleave(atoms),
            sid=[""] * batch_size,
            dataset=["omol"] * batch_size,
        )

    def encode(self, q: torch.Tensor) -> Encoded:
        data = self._atomic_data(q)
        embedding = self.backbone(data)
        node = embedding["node_embedding"]
        if node.ndim != 3:
            raise ValueError(
                f"Unexpected eSEN node embedding shape: {tuple(node.shape)}"
            )
        scalars = node[:, 0, :].reshape(q.shape[0], q.shape[1], -1)
        return Encoded(data, embedding, scalars)

    def apply_force_head(self, head: nn.Module, encoded: Encoded):
        out = head(encoded.data, encoded.embedding)
        denormalized = self.force_normalizer.denorm(out["forces"])
        return denormalized.reshape(
            encoded.scalars.shape[0], encoded.scalars.shape[1], 3
        )

    def forward(self, q: torch.Tensor):
        return self.apply_force_head(self.force_head, self.encode(q))


class TinyForceHead(nn.Module):
    """Small rotation-equivariant vector head for local tests and ablations."""

    def __init__(self, hidden: int):
        super().__init__()
        self.weight = nn.Linear(hidden, 1, bias=False)

    def forward(self, data, emb):
        coeff = self.weight(emb["node_scalar"]).squeeze(-1)
        r = emb["relative"]
        dist = torch.linalg.vector_norm(r, dim=-1).clamp_min(1e-6)
        radial = torch.exp(-(dist**2) / 9.0)
        eye = torch.eye(r.shape[1], dtype=torch.bool, device=r.device)[None]
        radial = radial.masked_fill(eye, 0)
        force = (
            (coeff[:, :, None] + coeff[:, None, :])[:, :, :, None]
            * radial[..., None]
            * r
        ).sum(dim=2)
        return {"forces": force.reshape(-1, 3)}


class TinyAdapter(nn.Module):
    """Lightweight trainable backend used to exercise the whole pipeline offline."""

    def __init__(self, atomic_numbers: list[int], hidden: int = 32):
        super().__init__()
        self.register_buffer(
            "atomic_numbers", torch.tensor(atomic_numbers, dtype=torch.long)
        )
        self.scalar_dim = hidden
        self.embedding = nn.Embedding(100, hidden)
        self.radial = nn.Sequential(
            nn.Linear(2 * hidden + 2, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )
        self.force_head = TinyForceHead(hidden)

    def encode(self, q: torch.Tensor) -> Encoded:
        batch, atoms, _ = q.shape
        z = self.embedding(self.atomic_numbers).unsqueeze(0).expand(batch, -1, -1)
        r = q[:, None, :, :] - q[:, :, None, :]
        dist = torch.linalg.vector_norm(r, dim=-1)
        zi = z[:, :, None, :].expand(-1, -1, atoms, -1)
        zj = z[:, None, :, :].expand(-1, atoms, -1, -1)
        pair = torch.cat([zi, zj, dist[..., None], torch.exp(-dist[..., None])], dim=-1)
        mask = (~torch.eye(atoms, dtype=torch.bool, device=q.device))[None, :, :, None]
        scalar = z + (self.radial(pair) * mask).sum(dim=2) / max(1, atoms - 1)
        return Encoded(None, {"node_scalar": scalar, "relative": r}, scalar)

    def apply_force_head(self, head: nn.Module, encoded: Encoded):
        return head(encoded.data, encoded.embedding)["forces"].reshape(
            *encoded.scalars.shape[:2], 3
        )

    def forward(self, q: torch.Tensor):
        return self.apply_force_head(self.force_head, self.encode(q))


def make_adapter(
    backend: str, atomic_numbers: list[int], checkpoint: str, hidden: int = 32
):
    if backend == "esen":
        return ESENAdapter(atomic_numbers, checkpoint)
    if backend == "tiny":
        return TinyAdapter(atomic_numbers, hidden)
    raise ValueError(f"Unknown backbone: {backend}")


class ParallelStudent(nn.Module):
    """PDD phase-space mean velocities with cloned direct-force heads.

    The force component starts from m copies of the pretrained force head.
    Velocity-dependent corrections and position-velocity heads are zero
    initialized. Every head uses a single shared backbone representation.
    """

    def __init__(self, adapter: nn.Module, masses_dalton, max_block: int, dt_ps: float):
        super().__init__()
        if max_block < 1 or dt_ps <= 0:
            raise ValueError("max_block and dt_ps must be positive")
        self.adapter = adapter
        self.max_block = max_block
        self.dt_ps = dt_ps
        self.register_buffer(
            "masses", torch.as_tensor(masses_dalton, dtype=torch.float32)
        )
        self.force_heads = nn.ModuleList(
            [deepcopy(adapter.force_head) for _ in range(max_block)]
        )
        # Four equivariant vector bases per atom: local velocity, weighted
        # neighbor velocity, relative velocity, and geometric neighbor vector.
        self.qdot_heads = nn.ModuleList(
            [nn.Linear(adapter.scalar_dim + 1, 4) for _ in range(max_block)]
        )
        self.accel_heads = nn.ModuleList(
            [nn.Linear(adapter.scalar_dim + 1, 4) for _ in range(max_block)]
        )
        for layer in [*self.qdot_heads, *self.accel_heads]:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def _vector_bases(self, q: torch.Tensor, v: torch.Tensor):
        r = q[:, None, :, :] - q[:, :, None, :]
        dist2 = (r**2).sum(dim=-1)
        weights = torch.exp(-dist2 / 9.0)
        eye = torch.eye(q.shape[1], device=q.device, dtype=torch.bool)[None]
        weights = weights.masked_fill(eye, 0)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        neighbor_v = torch.einsum("bij,bjc->bic", weights, v)
        neighbor_r = torch.einsum("bij,bijc->bic", weights, r)
        return torch.stack([v, neighbor_v, neighbor_v - v, neighbor_r], dim=2)

    def forward(self, q: torch.Tensor, v: torch.Tensor, block: int | None = None):
        if q.shape != v.shape or q.ndim != 3 or q.shape[-1] != 3:
            raise ValueError("q and v must have shape [batch, atoms, 3]")
        block = block or self.max_block
        if not 1 <= block <= self.max_block:
            raise ValueError("block exceeds max_block")
        encoded = self.adapter.encode(q)
        bases = self._vector_bases(q, v)
        scalar_input = torch.cat(
            [encoded.scalars, (v**2).sum(dim=-1, keepdim=True) / 100.0], dim=-1
        )
        qdots, accels = [], []
        for k in range(block):
            force = self.adapter.apply_force_head(self.force_heads[k], encoded)
            base_accel = force * (
                ACCEL_PER_FORCE_PER_DALTON / self.masses[None, :, None]
            )
            qcoeff = self.qdot_heads[k](scalar_input)
            acoeff = self.accel_heads[k](scalar_input)
            # Coefficient corrections are dimensioned via the dt and a
            # moderate force scale; they are zero at initialization.
            qdot = (
                v
                + self.dt_ps * (k + 0.5) * base_accel
                + torch.einsum("baf,bafc->bac", qcoeff, bases)
            )
            accel = base_accel + 100.0 * torch.einsum("baf,bafc->bac", acoeff, bases)
            qdots.append(qdot)
            accels.append(accel)
        return torch.stack(qdots, dim=1), torch.stack(accels, dim=1)

    def states_in_block(self, q, v, qdots, accels):
        """All student states, including the start, from one model call."""
        q_steps = torch.cat(
            [q[:, None], q[:, None] + self.dt_ps * torch.cumsum(qdots, dim=1)], dim=1
        )
        v_steps = torch.cat(
            [v[:, None], v[:, None] + self.dt_ps * torch.cumsum(accels, dim=1)], dim=1
        )
        return q_steps, v_steps

    def advance(self, q, v, block: int):
        qdots, accels = self(q, v, block)
        return q + self.dt_ps * qdots.sum(dim=1), v + self.dt_ps * accels.sum(dim=1)
