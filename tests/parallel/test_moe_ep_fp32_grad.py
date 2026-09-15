"""Regression for rounding across the expert/replica autograd boundary."""

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _worker(rank, rendezvous, train_gate, train_down):
    from tests.parallel.test_moe_ep_load_balance_dispatch import ep_load_balance as ep
    from tests.parallel.test_moe_ep_load_balance_dispatch import moe_layer as moe

    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        plan = ep.build_ep_balance_plan(torch.tensor([[0]]), 2, dist.group.WORLD, 1)
        assert len(plan.replicas) == 1
        assert plan.replicas[0].source_rank == 0 and plan.replicas[0].target_rank == 1
        gate = torch.zeros(1, 2, 3, dtype=torch.bfloat16, requires_grad=train_gate)
        down = torch.zeros(1, 3, 1, dtype=torch.bfloat16, requires_grad=train_down)
        tokens = torch.ones(1, 3, dtype=torch.bfloat16, requires_grad=True)

        def forward(ctx, tokens, cumsum, combined_gate, combined_down, limit):
            ctx.save_for_backward(combined_gate, combined_down)
            return tokens.clone()

        def backward(ctx, grad, *, weight_gradient_dtype=None, weight_needs_grad=None):
            assert weight_gradient_dtype == torch.float32
            outputs = []
            for weight, needed in zip(ctx.saved_tensors, weight_needs_grad, strict=True):
                if not needed:
                    outputs.append(None)
                    continue
                value = torch.zeros_like(weight, dtype=weight_gradient_dtype)
                value[0 if rank == 0 else 1].fill_(257.0 if rank == 0 else -256.0)
                outputs.append(value)
            return grad, None, *outputs, None

        original_forward, original_backward = moe.EPMergedFc1GroupGemm.forward, moe.EPMergedFc1GroupGemm.backward
        moe.EPMergedFc1GroupGemm.forward = staticmethod(forward)
        moe.EPMergedFc1GroupGemm.backward = staticmethod(backward)
        try:
            output = moe._BalancedEPMergedFc1GroupGemm.apply(
                tokens,
                torch.tensor([1, 2]),
                gate,
                down,
                torch.zeros_like(gate),
                torch.zeros_like(down),
                plan,
                dist.group.WORLD,
                None,
            )
            output.sum().backward()
        finally:
            moe.EPMergedFc1GroupGemm.forward = staticmethod(original_forward)
            moe.EPMergedFc1GroupGemm.backward = staticmethod(original_backward)
        for weight, needed in ((gate, train_gate), (down, train_down)):
            if needed:
                assert torch.equal(weight.grad, torch.full_like(weight, 1 if rank == 0 else 0))
                assert weight.grad.dtype == torch.bfloat16
            else:
                assert weight.grad is None
        assert torch.equal(tokens.grad, torch.ones_like(tokens))

        # A separate BF16 Cat edge quantizes 257 to 256 before replica
        # reduction, producing zero rather than the true one.
        old_weight = torch.zeros(1, 2, 3, dtype=torch.bfloat16, requires_grad=True)
        combined = ep.cat_local_and_replica_weights(old_weight, torch.zeros_like(old_weight), plan, dist.group.WORLD)
        partial = torch.zeros_like(combined, dtype=torch.float32)
        partial[0 if rank == 0 else 1].fill_(257 if rank == 0 else -256)
        (combined.float() * partial).sum().backward()
        assert torch.count_nonzero(old_weight.grad) == 0
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("train_gate,train_down", [(True, True), (True, False), (False, True)])
def test_fp32_replica_gradients_cross_autograd_boundary_before_final_rounding(tmp_path, train_gate, train_down):
    mp.spawn(_worker, args=(f"file://{tmp_path / 'fp32_grad'}", train_gate, train_down), nprocs=2, join=True)
