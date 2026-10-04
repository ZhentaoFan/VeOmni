# Full35B EP execution: remove replay, defer waits, reduce planning work

> Historical ledger: statements, commands, environments and measurements below
> refer to the recorded experiment revision, not current rebased-source validation.
> See the [evidence index](qwen3_5_moe_ep_evidence.md) for current status and later results.

Status: implementation, repeated comparisons, final profile analysis and keeper
restoration are complete.
The repeated matrix is effectively tied; it does not demonstrate significant
enabled-versus-disabled acceleration.

## Fixed scope

This follows the [September 19 full-model study](qwen3_5_full35b_ep_20260919.md).
Only three requested code directions are changed: avoid overwritten owner dW,
submit independent dX before ordered communication waits, and reduce planner CPU
work. There is no model/layer reduction, new routing policy, token dropping,
auxiliary-loss change, common Conv3d optimization or dependency replacement.

Hardware is test27 only, eight H200 GPUs. The complete pretrained Qwen3.5-35B-A3B
trainer model has 35,107,181,936 parameters and 40 text layers (30 linear and 10
full attention). GBS8/MBS1, EP8/FSDP2/SP1, 2K native routing, the same fixed
synthetic multimodal fixture, BF16 parameters/activations, FP32 FSDP reduction,
no activation checkpointing, seed 0 and strict ordered gradients are retained.
Router auxiliary loss is OFF in every arm (`output_router_logits=false`).
No Ascend, other node, platform job, remote push, or generated modeling edit is
part of this task.

Repository/branch: `perf/issue-1038-full35b` under
`/volume/tzhen/tmps/veomni-35b-ep-momo-0919/VeOmni`.

Experiment helpers/logs are under
`/volume/tzhen/tmps/veomni-35b-ep-execution-momo-0920` (called `WORK` below).
Immutable training harnesses and raw training results stay under the previous
task root `/volume/tzhen/tmps/veomni-35b-ep-momo-0919` (called `P` below).
Old artifacts are retained; new labels start with `execution20-`.

## Implemented changes

### 1. Compute affected owner suffixes only once — `a485fb3`

The earlier ordered backward first computed all local expert gradients. Incoming
FP32 prefixes then overwrote affected owner rows, after which their suffixes
were computed again. The first suffix result was unused.

The initial GEMM now selects only unaffected local groups. Affected owners are
written by prefix receipt and, when nonempty, one suffix accumulation. An
internal group-index indirection retains each original group's cumsum/matrix
offsets, original-owner tile-selection key and reduction order. Unaffected
empty groups still write zeros; an empty selection launches nothing. The
ordinary grouped-GEMM path defaults to no indirection.

This removes redundant arithmetic; it does not necessarily reduce the number of
kernel launches, since the remaining local groups still share one GEMM launch.

### 2. Defer down-projection waits behind independent dX — `cbb8a75`

An internal pending handle submits the first ordered down-wgrad stage and its
P2P operations, then returns without inserting its completion wait. Gate
derivatives and the input-gradient GEMM are submitted before that wait. NCCL's
existing communication stream can overlap this independent compute; no new
compute stream is introduced.

The handle retains operands and P2P buffers through the suspended generator.
Its wait advances the remaining prefix/suffix stages, and down communication
still completes before gate/up communication begins. This preserves peer message
order and each expert's accumulator order. The caller uses `finally` to wait
when independent work raises. CPU tests cover that error path; this is not a
claim of accelerator fault-injection coverage.

The synchronous callable still exists. Default helpers returning a Tensor are
not mistaken for pending handles. Frozen gate/down weights and empty stages are
covered, and repeated waits do not repeat gradient work.

### 3. Lazily merge owner-local planner priorities — `3b33f23`

Each owner's expert list is sorted by owner token count and expert ID. The
planner lazily merges those lists with the unchanged complete global key
`(-rank_load, -owner_tokens, owner, expert)`. After accepting a move, only the
chosen owner's list is re-sorted, including cases where its allocation increases.
No accepted move is evaluated against a partly updated merge.

An impossible first-replica candidate is also filtered before entering the merge
when its owner cannot beat the lightest available target by two integer tokens.
Existing replicas retain their multi-rank redistribution search. This changes
the search cost, not its outputs or tie breaks.

Numba was examined but not used: both controller and GPU-node imports fail
because the installed Numba accepts NumPy up to 2.3 while the environment has
NumPy 2.5. The training environment was not downgraded. The shipped optimization
uses Python integer bookkeeping and standard-library lazy merging.

## Correctness and isolated observations

Every implementation commit passed independent review and `make quality`.

- Baseline full `pytest tests/` was attempted and stopped during collection on
  missing optional `av`. Full-repository regression is therefore incomplete;
  this failure predates the changes and was not fixed by changing dependencies.
- Prior-source focused CPU/Gloo suite: 118 passed / 21 skipped.
- Final combined CPU/Gloo related suite: **213 passed / 141 skipped**.
- Group-selection GPU-host suite: **100 passed**, comprising 64 CUDA operator
  cases and 36 CPU/Gloo cases. Cases include both low-precision dtypes,
  accumulation, empty/tail groups, permuted group IDs, and untouched skipped rows.
- Pending-handle tests cover frozen weights, independent-dX exceptions,
  retained operands, empty stages and idempotent waits. Real four-rank Gloo
  tests compare deferred and synchronous chains, including multiple replicas.
- Two execution probes each cover 40 layers on eight ranks using actual saved
  full-model inputs and pretrained expert weights. All 320 case/rank records
  match output, dX, router gradient and both expert-weight gradients bytewise.
  The reference output also matches the recorded full-model expert output.
- The final planner matches every field of 320 complete native plans against
  the preceding implementation. CPU oracle tests also cover ties, empty
  owners, exhaustive small counts, non-power-of-two EP sizes and more than 64 ranks.

The execution probes precompute identical plans and use six alternating trials
per layer; they are not full training throughput. The CPU planner benchmark uses
30 alternating trials over 40 native histograms. No outliers are deleted.

| Isolated change | Before | After | Measurement |
|---|---:|---:|---|
| Avoid overwritten dW | 242.654 ms | 241.164 ms | Sum of 40 per-case max-rank execution means |
| Defer down-wgrad waits | 240.970 ms | 234.689 ms | Same execution measure, with some noisy trials |
| Lazy planner merge | 19.697 ms | 12.519 ms | CPU decision-only total over 40 histograms |

These observations cannot be added together or substituted for step time.
See `no-replay-execution-bench/`, `overlap-execution-bench/`, and
`production-planner-bench.json` under `WORK`.

## Complete-model timing

Original OFF is `4eb43ae`, previous ON is `4323191`, and new ON is `3b33f23`.
Every timing run is unprofiled, starts from the same original checkpoint and
uses a clean frozen worktree with the identical V2 harness. The primary window
is steps **9-64**, all 56 completed-step intervals, with no fastest-step selection
or outlier removal. Wall includes fetch, training, monitoring/logging and common
measurement overhead. Compute is the mean of per-step max-rank synchronized
forward/backward/optimizer times, not a maximum over steps.

The repeated matrix follows OFF/ON/ON/OFF order; the staged overlap smoke and
CPU development work occurred between the first OFF and final ON. The last three
matrix arms ran consecutively, with no concurrent GPU probe or keeper.

| Label | Version | Wall mean, steps 9-64 (s) | Mean max-rank compute (s) |
|---|---|---:|---:|
| `execution20-off-a64` | Original OFF | 2.372569 | 2.353267 |
| `execution20-new-on-b64` | New ON | 2.374711 | 2.356114 |
| `execution20-new-on-c64` | New ON | 2.373398 | 2.353709 |
| `execution20-off-d64` | Original OFF | 2.377737 | 2.358803 |

Pairwise time reductions are **-0.0903%** and **+0.1825%**. The mean is OFF
**2.375153 s**, ON **2.374054 s**, a difference of about 1.1 ms. The direction
changes across pairs; this should be treated as a tie, not a stable net speedup.
The earlier working 10% target remains unmet. `P/execution20-final-abba.json`
records precision success separately from the failed net-benefit/10% gates;
requiring significant benefit intentionally exits with code 2.

All four arms' 64 loss/gradient-norm values match exactly, under the unchanged
gate `abs(delta) <= 1e-4 + 1e-3*abs(reference)`. Used FLA configurations match on
all 32 rank/run records. This is not bytewise validation of every model parameter
at every optimizer step, or a guarantee across fresh cold FLA autotuning.

Previous-ON references bracket the paired study:
`execution20-old-on64` gives **2.401640 s**, and `execution20-old-on-e64` gives
**2.391593 s** over the same 9-64 window. Their mean is **2.396617 s**, versus
**2.374054 s** for new ON: an observed **0.941% time reduction** relative to the
previous enabled implementation. All recorded scalar curves remain exact.
These references are reported separately, not substituted into the original-OFF
matrix. This modest old-ON improvement is not a significant EP-enablement gain.

The two staged 32-step smokes also match baseline loss/gradient-norm prefixes:
selected-group dW gives 2.405462 s over steps 9-32, and deferred waits give
2.378598 s. These are single functional smokes, not matched per-change end-to-end
speedup proofs. In particular, the removal-only smoke is not hidden because its
timing is slower than nearby references.

## Profile and resource restoration

`execution20-new-profile16` completes with all 16 scalar values exactly matching
the unprofiled baseline prefix and identical used FLA configurations on all eight
ranks. Eight Chrome/PyTorch traces cover steps 12-13. They are not Nsight Systems
reports and are excluded from throughput acceptance.

`WORK/analyze_dx_overlap.py` uses CPU
scope/correlation IDs to select the second (input) dX GEMM and down-prefix
SendRecv launches, then intersects their actual GPU intervals. This avoids
equating overlapping CPU submission scopes with GPU overlap. Across the eight
previous-ON rank traces, none of the 640 backward scopes has this overlap. The
new trace has **24/640** such scopes, distributed by rank as `[0,1,1,0,4,5,6,7]`.
Average intersection time is **0.0983 ms per rank per step** for this particular
input-dX GEMM. It excludes earlier independent gate derivatives/concatenation,
counts NCCL kernel lifetime rather than productive transfer time, and is not a
causal step-speedup estimate. The reordering creates real overlap, but these
two sampled steps do not demonstrate broad communication hiding.

The previous-ON profile has mean accumulated weight-gradient GEMM GPU time of
19.867 ms per rank/step; new ON has 17.553 ms. Launch counts remain 200-229 per
rank/step, consistent with reducing the selected groups inside existing launches
rather than eliminating whole launches. These are profiled GPU kernel sums,
not the unprofiled step-time delta. See `old-prefix-overlap-rank0.json`,
`old-prefix-overlap-rest.json`, `new-prefix-overlap.json` under `WORK`, and
`P/results/execution20-new-profile16/profile-summary.json`.

The original test27-only eight-GPU synthetic keeper is restored as generation 2,
`fake-test27-8g-momo-0916-a00002`, supervisor PID 2156317. Ownership/progress checks
in `WORK/keeper-restored-a.json` and `keeper-restored-b.json` verify eight owned
GPU processes, high utilization and increasing launches. The task-owned STOP
was archived as `WORK/keeper-STOP-paused-20260920.txt`; the old 24-GPU STOP remains
unchanged. This restores synthetic driver-burn, not indefinite 35B training.
No test25/test26/test28 resources or shared workers were stopped or restarted.

Remaining interpretation: these three changes reduce enabled-path execution
cost, but the measured ON/OFF matrix still does not establish significant net
benefit. Common Conv3d work is unchanged, and strict prefix ordering/FP32
communication costs remain; neither may be silently changed to manufacture an
EP enablement gain.
