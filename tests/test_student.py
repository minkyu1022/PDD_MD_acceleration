import torch

from pdd_md.model import ParallelStudent, TinyAdapter


def test_cloned_heads_and_one_backbone_call():
    torch.manual_seed(1)
    adapter = TinyAdapter([1, 6, 8], hidden=12)
    student = ParallelStudent(adapter, [1.0, 12.0, 16.0], max_block=4, dt_ps=0.0005)
    for head in student.force_heads:
        assert torch.allclose(head.weight.weight, adapter.force_head.weight.weight)
    q = torch.randn(2, 3, 3)
    v = torch.randn(2, 3, 3)
    calls = []
    original_encode = student.adapter.encode

    def counted_encode(q):
        calls.append(1)
        return original_encode(q)

    student.adapter.encode = counted_encode
    qdots, accels = student(q, v, block=4)
    assert calls == [1]
    assert qdots.shape == accels.shape == (2, 4, 3, 3)
    qs, vs = student.states_in_block(q, v, qdots, accels)
    assert qs.shape == vs.shape == (2, 5, 3, 3)
    assert torch.allclose(qs[:, -1], q + student.dt_ps * qdots.sum(dim=1))


def test_student_is_translation_and_rotation_equivariant():
    torch.manual_seed(2)
    student = ParallelStudent(TinyAdapter([1, 6, 8], hidden=12), [1, 12, 16], 3, 0.0005)
    q = torch.randn(2, 3, 3)
    v = torch.randn(2, 3, 3)
    rotation = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    shift = torch.tensor([2., -1., 0.5])
    qdot, accel = student(q, v)
    qdot_rot, accel_rot = student(q @ rotation.T + shift, v @ rotation.T)
    assert torch.allclose(qdot_rot, qdot @ rotation.T, atol=2e-4, rtol=2e-4)
    assert torch.allclose(accel_rot, accel @ rotation.T, atol=2e-2, rtol=2e-4)
