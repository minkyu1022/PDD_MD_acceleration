"""Exercise the FAIR-Chem tensor interface with a tiny initialized eSCN.

This tests software wiring; it does not test the gated pretrained weights.
"""

import os
from pathlib import Path

import pytest
import torch
from torch import nn

from pdd_md.model import ESENAdapter, ParallelStudent


@pytest.mark.slow
@pytest.mark.parametrize("otf_graph", [True, False])
def test_esen_tensor_contract(tmp_path: Path, otf_graph: bool):
    os.environ.setdefault("FAIRCHEM_CACHE_DIR", str(tmp_path / "fairchem"))
    os.environ.setdefault("WARP_CACHE_PATH", str(tmp_path / "warp"))
    pytest.importorskip("fairchem")
    from fairchem.core.models.uma.escn_md import Linear_Force_Head, eSCNMDBackbone
    from fairchem.core.modules.normalization.normalizer import Normalizer

    adapter = ESENAdapter.__new__(ESENAdapter)
    nn.Module.__init__(adapter)
    adapter.backbone = eSCNMDBackbone(
        sphere_channels=4,
        lmax=2,
        mmax=2,
        otf_graph=otf_graph,
        edge_channels=5,
        num_distance_basis=7,
        use_dataset_embedding=False,
        always_use_pbc=False,
    )
    adapter.force_head = Linear_Force_Head(adapter.backbone)
    adapter.force_normalizer = Normalizer(rmsd=2.0)
    adapter.register_buffer("atomic_numbers", torch.tensor([1, 6, 8]))
    adapter.scalar_dim = 4
    q = torch.tensor(
        [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]], dtype=torch.float32
    )
    v = torch.randn_like(q)
    encoded = adapter.encode(q)
    raw = adapter.force_head(encoded.data, encoded.embedding)["forces"].reshape_as(q)
    assert torch.allclose(
        adapter.apply_force_head(adapter.force_head, encoded), 2 * raw
    )
    assert adapter(q).shape == q.shape
    student = ParallelStudent(adapter, [1.0, 12.0, 16.0], 2, 0.0005)
    qdot, accel = student(q, v, 2)
    assert qdot.shape == accel.shape == (1, 2, 3, 3)
    (qdot.square().mean() + accel.square().mean()).backward()
    assert student.force_heads[0].linear.weight.grad is not None
