"""Physical symmetry checks for the OC20 relaxation PoC model."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from relaxation_poc import EquivariantBlock


def test_block_respects_constraints_periodicity_and_rotation():
    torch.manual_seed(7)
    model = EquivariantBlock(4).eval()
    pos = torch.tensor([[0.1, 0.2, 0.0], [1.1, 0.2, 0.0], [0.1, 1.2, 0.0]])
    numbers = torch.tensor([1, 29, 29])
    tags = torch.tensor([2, 1, 0])
    free = torch.tensor([1.0, 1.0, 0.0])
    cell = torch.diag(torch.tensor([10.0, 10.0, 20.0]))
    pbc = torch.ones(3)
    force = torch.tensor([[0.2, 0.1, 0.0], [0.1, 0.3, 0.0], [0.0, 0.0, 0.0]])
    base = model(pos, numbers, tags, free, cell, pbc, force)
    assert torch.count_nonzero(base[:, 2]) == 0

    wrapped = pos.clone()
    wrapped[1] += cell[0]
    torch.testing.assert_close(model(wrapped, numbers, tags, free, cell, pbc, force), base, atol=1e-6, rtol=1e-6)

    rotation = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    rotated = model(pos @ rotation.T, numbers, tags, free, cell @ rotation.T, pbc, force @ rotation.T)
    torch.testing.assert_close(rotated, base @ rotation.T, atol=1e-6, rtol=1e-6)
