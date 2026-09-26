"""Keep trajectory comparisons on the same physical time grid."""

import numpy as np

from pdd_md.evaluate import (
    _coarse_verlet_rollout,
    _compare,
    _metrics_or_failure,
    _student_rollout,
)
from pdd_md.teacher import PhaseState


class ConstantVelocityStudent:
    def advance_fused(self, q, v, block):
        return q + block * v, v


def test_block_rollout_compares_matching_endpoints():
    initial = PhaseState(np.zeros((1, 3)), np.ones((1, 3)))
    states, seconds = _student_rollout(
        ConstantVelocityStudent(), initial, total_steps=4, block=2, device="cpu"
    )
    reference = [PhaseState(np.full((1, 3), k), initial.v) for k in range(5)]
    assert len(states) == 3
    assert seconds > 0
    assert _compare(states, reference, stride=2)["q_path_rmse_angstrom"] == 0


def test_nan_rollout_is_reported_as_failure():
    initial = PhaseState(np.zeros((1, 3)), np.ones((1, 3)))
    failed = PhaseState(np.full((1, 3), np.nan), initial.v)
    result = _metrics_or_failure([initial, failed], [initial, initial], None, 0, 1)
    assert result["finite"] is False
    assert result["q_endpoint_rmse_angstrom"] is None
    assert result["absolute_energy_drift_kj_mol"] is None


def test_failed_coarse_integrator_keeps_comparison_grid():
    from openmm import OpenMMException

    class FailedTeacher:
        def rollout(self, state, steps, dt):
            raise OpenMMException("unstable integration")

    initial = PhaseState(np.zeros((1, 3)), np.zeros((1, 3)))
    states = _coarse_verlet_rollout(FailedTeacher(), initial, 8, 2, 0.0005)
    assert len(states) == 5
    assert not np.isfinite(states[-1].q).all()
