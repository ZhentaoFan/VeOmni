"""Ordered FP32 expert-weight reduction without changing the disabled kernel."""

import hashlib
import json
import math

import torch
import torch.distributed as dist


def weight_grad_token_granularity() -> int:
    """Use boundaries compatible with every configured weight-gradient tile."""
    from ...ops.kernels.moe._kernels.kernel.group_gemm import group_gemm_same_mn_kernel

    return math.lcm(*(int(config["BLOCK_K"]) for config in group_gemm_same_mn_kernel.configs.values()))


def weight_grad_config_metadata() -> tuple[int, int]:
    """Identify the exact tile configuration shared by an ordered EP group."""
    from ...ops.kernels.moe._kernels.kernel.group_gemm import group_gemm_same_mn_kernel

    serialized = json.dumps(group_gemm_same_mn_kernel.configs, sort_keys=True).encode()
    signature = int.from_bytes(hashlib.sha256(serialized).digest()[:8], "little") & ((1 << 63) - 1)
    return weight_grad_token_granularity(), signature


def _launch_weight_grad(a, b, c, cumsum, original_total, accumulate):
    import triton

    from ...ops.kernels.moe._kernels.kernel.group_gemm import group_gemm_same_mn_kernel

    groups, m, n = c.shape
    group_gemm_same_mn_kernel[
        lambda config: (triton.cdiv(m, config["BLOCK_M"]) * triton.cdiv(n, config["BLOCK_N"]), groups)
    ](
        a_ptr=a,
        b_ptr=b,
        c_ptr=c,
        cumsum_K=cumsum,
        total_K=original_total,
        G=groups,
        M=m,
        N=n,
        TRANSPOSE_A=True,
        TRANSPOSE_B=False,
        ACCUMULATE_TO_C=accumulate,
    )


def make_ordered_weight_grad(plan, ep_group, device, *, launcher=None):
    """Build a per-backward closure that forwards original-order prefix states.

    Replica quotas consume consecutive prefixes of the global sender-ordered
    expert rows. Each replica extends the preceding FP32 accumulator, then the
    original owner extends it with its suffix. Independent rounded partials
    are never added in a different tree. The public disabled path is unchanged.

    ``launcher`` is a CPU-test seam; production calls the existing Triton kernel.
    """
    launch = _launch_weight_grad if launcher is None else launcher
    rank = dist.get_rank(ep_group)
    local = plan.num_local_experts
    chains = {}
    for replica in plan.replicas:
        chains.setdefault(replica.source_expert, []).append(replica)
    chains = dict(sorted(chains.items()))
    lengths = plan.tokens_per_local_physical_expert.sum(dim=0).tolist()
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    sizes = torch.tensor(lengths, dtype=torch.long, device=device)

    def weight_grad(a, b, c, cumsum_K, max_K, transpose_a=False, transpose_b=False):
        if not transpose_a or transpose_b:
            raise ValueError("Ordered expert wgrad requires A.T @ B.")
        if c.dtype != torch.float32:
            raise TypeError("Ordered expert wgrad requires FP32 prefix buffers.")
        if not (a.is_contiguous() and b.is_contiguous() and c.is_contiguous()):
            raise ValueError("Ordered expert wgrad requires contiguous operands.")
        if c.shape[0] != len(lengths) or a.shape[0] != b.shape[0] or a.shape[0] != offsets[-1]:
            raise ValueError("Ordered expert wgrad shapes do not match the physical plan.")

        def segment(slot, owner, accumulate):
            launch(
                a[offsets[slot] : offsets[slot + 1]],
                b[offsets[slot] : offsets[slot + 1]],
                c[slot : slot + 1],
                sizes[slot : slot + 1],
                plan.rank_loads_before[owner],
                accumulate,
            )

        # Unaffected originals complete here. Affected owner suffixes are
        # overwritten by the incoming prefix and replayed after communication.
        launch(a, b, c[:local], cumsum_K[:local], plan.rank_loads_before[rank], False)
        for stage in range(max((len(chain) for chain in chains.values()), default=0)):
            for chain in chains.values():
                if stage < len(chain) and chain[stage].target_rank == rank:
                    replica = chain[stage]
                    segment(local + replica.target_slot, replica.source_rank, stage > 0)
            works = []
            buffers = []
            # All ranks walk the same ordered edges, including empty senders.
            for chain in chains.values():
                if stage >= len(chain):
                    continue
                replica = chain[stage]
                last = stage + 1 == len(chain)
                receiver = replica.source_rank if last else chain[stage + 1].target_rank
                recv_slot = replica.source_local_expert if last else local + chain[stage + 1].target_slot
                send_slot = local + replica.target_slot
                if receiver == replica.target_rank:
                    if rank == receiver:
                        c[recv_slot].copy_(c[send_slot])
                elif rank == replica.target_rank:
                    tensor = c[send_slot].contiguous()
                    buffers.append(tensor)
                    works.append(dist.isend(tensor, dst=dist.get_global_rank(ep_group, receiver), group=ep_group))
                elif rank == receiver:
                    tensor = c[recv_slot]
                    buffers.append(tensor)
                    works.append(
                        dist.irecv(tensor, src=dist.get_global_rank(ep_group, replica.target_rank), group=ep_group)
                    )
            for work in works:
                work.wait()
        for chain in chains.values():
            owner = chain[0]
            if owner.source_rank == rank and lengths[owner.source_local_expert] > 0:
                segment(owner.source_local_expert, rank, True)

    return weight_grad
