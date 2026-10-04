# Qwen3.5-MoE EP evidence index

## Current status

This branch rebases the feature onto upstream `86c5ec1`. Validation of the
rebased source is limited to local CPU/Gloo/static checks.
Accelerator precision, performance and memory have not been revalidated on this
source. Earlier tests do not establish Transformers 5.16.1 or rebased-source
hardware acceptance. Balancing stays opt-in; a more even token histogram is not
evidence of net benefit.

Current [semantics/support boundaries](../design/qwen3_5_moe_ep_load_balance.md)
and [executable recipes](../examples/qwen3_5.md#qwen35-moe-ep-load-balance-experiments)
are separate from the six immutable historical ledgers below. Their original
versions, hashes, failures and raw-artifact locations remain authoritative for
those runs only.

## Historical experiments

| Ledger | Evidence and limitation |
| --- | --- |
| [August 22 Ascend toy](qwen3_5_moe_ep_load_balance.md#ascend-a3-matched-result) | Torch/torch_npu 2.9.0, Transformers 5.9.0; 10-step forced hotspot, exact scalar curves, +0.997% steady tokens/s, +113,223,680 bytes allocated peak. CUDA/full-scale were untested **then**. |
| [September 14 repair](qwen3_5_moe_ep_repair_20260914.md#matched-cuda-training-experiment) | Four H200s; planner/layout/absolute-telemetry repairs, 32-step precision pass. Final matched repaired ON is 1.47% slower; no Ascend evidence. |
| [September 16 optimization](qwen3_5_moe_ep_optimization_20260916.md#completed-step-wall-intervals) | Eight H200s; scalar-only heatmap savings belong to both later ON/OFF arms. [Failed precision gates](qwen3_5_moe_ep_optimization_20260916.md#failed-gates) remain failures, not accepted speedups. |
| [September 16 ON/OFF](qwen3_5_moe_ep_onoff_20260916.md#final-production-results) | Six exact 128-step scalar comparisons. E256 hotspot mean time improves 1.647%, but native E256 regresses **3.143%** (1.589648 -> 1.639615 s); four-layer fixture, not full 35B. |
| [September 19 full 35B](qwen3_5_full35b_ep_20260919.md#repeated-frozen-comparison) | Eight H200s, all 40 text layers, native routing; repeated 64-step scalar curves exact. ON costs **~1.25%** more time (2.367141 -> 2.396628 s); allocator peaks increase. 10% benefit target unmet. |
| [September 20 execution](qwen3_5_full35b_ep_execution_20260920.md#complete-model-timing) | Same full model; four exact 64-step scalar curves. Pair signs differ (-0.0903%, +0.1825%); mean difference **~1.1 ms**, effectively tied. Old-ON improvement is not EP-enablement gain; **10% target remains unmet**. |

Each ledger records the actual timing window and measurement boundary. Operator
microbenchmarks, profiler sums, health/keeper checks and forced-hotspot smoke
tests must not be substituted for matched end-to-end performance.
