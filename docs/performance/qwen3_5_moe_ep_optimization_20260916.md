# Qwen3.5-MoE EP optimization evidence, 2026-09-16

> Historical ledger: statements, commands, environments and measurements below
> refer to the recorded experiment revision, not current rebased-source validation.
> See the [evidence index](qwen3_5_moe_ep_evidence.md) for current status and later results.

## Scope and status

This is a local CUDA/H200 optimization report for the issue-1038 development branch, not Ascend acceptance or a claim that the upstream issue is complete.
All comparisons use eight GPUs on one server. At most two independent cases use sixteen GPUs concurrently. No cross-server absolute timings are subtracted. test28 is not used.

Source baseline: `8f4bee334f0c240168de88e151821569dc09e615` (feature enabled).
Candidates: `84990e1` compact dispatch; `106d241` adds bulk host metadata; `fa105b3` adds exact integer-optimal planner stopping; `11ab77a` adds scalar-only monitoring without unconsumed images.
The candidate branches are local. This optimization round has not pushed them or changed a PR.

The final local branch is `perf/issue-1038-lazy-heatmaps`. The clearest measured full-step benefit is removal of unconsumed heatmap rendering with WandB already disabled: approximately 16-17% lower completion time in the two matched 128-step cases. Kernel/planner improvements are reported separately. This is not an enabled-WandB or universal feature-on/off speedup claim.

## Design and invariants

1. Compact CUDA routing stably sorts flattened physical expert occurrences, replacing dense expert/token masks and routing-weight matrices. It preserves expert-major accumulation order, BF16 multiplication before FP32 scatter, sender/expert block transposes, empty-rank collectives and replica-gradient return. Per-token physical IDs must be unique, as guaranteed by native Qwen top-k plus the alias map. Logical repeats split across distinct physical aliases remain covered. The disabled and NPU fused dispatch paths are unchanged.
2. Host quota allocation uses Python integers after counts reach CPU, with one remaining-count tensor materialization, bulk local-column assignment and destination split sums. Greedy decisions, sender priority, aliases, validation, and all float math are unchanged. No cross-forward plan or weight cache is added.
3. Greedy search stops at rank-load range <=1. For conserved integer token totals, the descending-sorted load vector is already globally minimal. This is an exact objective bound, not a tunable heuristic; validation and dispatch still execute.

4. The training monitor previously rendered three heatmaps even with WandB disabled, then discarded them. It now requests scalar-only formatting when no image sink is enabled. All scalar values, collection cadence, collectives, reset and failure handling remain unchanged. Direct monitor callers default to image rendering, and enabled-WandB behavior is retained. Gains from this change apply to scalar-only/no-WandB consumers; they are not a kernel improvement or an enabled-WandB speedup. Both sides of each measurement already had WandB disabled, so this is a code change, not a configuration change.

## Measurement protocol

Fixed seed-0 Qwen3.5-MoE fixture: hidden 2048, expert intermediate 512, four full-attention MoE layers, 16 experts/top-2, vocabulary 248320, 27-layer vision tower. One saved checkpoint and 256-sample seq-2048 dataset are reused. EP8/FSDP2/SP1, GBS8/MBS1, BF16 parameter compute, FP32 reduction, no activation checkpointing, one replica slot/rank, monitoring every step. Transformers 5.9.0; NVIDIA H200; driver 580.105.08.
Hot replay selects experts [0,2]; balanced replay selects [2*r,2*r+1] for EP rank r; native replay preserves native top-k IDs. Routing weights still come from the model. Neither model dimensions nor GBS change between compared arms.
Each unprofiled launch uses two epochs with max_steps=64 per epoch: 128 actual optimizer steps. Primary window is steps 9-64 (56 intervals); secondary window is 9-128 (120 intervals). The latter includes the epoch boundary. Do not describe these as 64-step runs.
Compute interval: synchronize around forward/backward/optimizer and reduce elapsed time by MAX across eight ranks. This excludes data fetch and step-end monitoring/logging, so it is not complete training throughput.
Wall interval (only `wall-*` runs): differences between rank-0 completed-step CPU timestamps, after callbacks/logging. This includes the next data fetch, compute, callbacks and logging. The wrapper adds no further GPU synchronization, but inherits the compute harness's two synchronizations and elapsed-time MAX reduction: this is instrumented training completion throughput, not uninstrumented native throughput or a max-rank wall-time series. The host-metadata and optimal-stop wall repeats run candidate before baseline; the scalar-only monitor comparisons run baseline before candidate.
Recorded production-file hashes are checked against pinned Git blobs (MoE modules, and in later launches the monitor/callback files). Launch commands match except output path, rendezvous port and, for disabled controls, feature enablement. LR and input-length curves must match. Later launches also record training-harness hashes and require paired equality; early launches lack those hashes, so a complete historical harness-provenance claim is not made. Small single-run differences are not confidence intervals or universal speedups.
Performance eligibility requires finite 128-point loss/grad-norm curves within abs_error <= 1e-4 + 1e-3*abs(baseline), plus exact MoE telemetry for enabled-vs-enabled comparisons. The comparison script exits nonzero on failed gates; ineligible comparisons are excluded from performance tables and listed separately.

## Matched compute intervals

Positive reduction means faster. All means below use max-rank per-step samples, not the maximum over the whole run.

| Baseline -> candidate | Steps 9-64 baseline / candidate (s) | Reduction | Steps 9-128 reduction | Max loss / grad-norm delta |
|---|---:|---:|---:|---:|
| `baseline-hot-64` -> `compact-hot-64` | 1.593762 / 1.579434 | +0.899% | +0.944% | 0 / 0 |
| `baseline-balanced-64` -> `compact-balanced-64` | 1.571644 / 1.585307 | -0.869% | -1.041% | 0 / 0 |
| `baseline-native-64` -> `compact-native-64` | 1.578922 / 1.603648 | -1.566% | -0.586% | 0 / 0 |
| `node25-baseline-balanced` -> `node25-metadata-balanced` | 1.537617 / 1.587383 | -3.237% | -2.626% | 0 / 0 |
| `node26-baseline-hot` -> `node26-metadata-hot` | 1.601002 / 1.575726 | +1.579% | +1.823% | 0 / 0 |
| `node26-baseline-native` -> `node26-metadata-native` | 1.596939 / 1.582888 | +0.880% | +0.626% | 0 / 0 |
| `node25-disabled-balanced` -> `node25-metadata-balanced` | 1.575080 / 1.587383 | -0.781% | -0.362% | 0 / 0 |
| `node25-wall-baseline-hot` -> `node25-wall-metadata-hot` | 1.577730 / 1.573947 | +0.240% | -0.040% | 0 / 0 |
| `node26-wall-baseline-native` -> `node26-wall-metadata-native` | 1.616790 / 1.589526 | +1.686% | +0.762% | 0 / 0 |
| `node25-wall-metadata-balanced` -> `node25-wall-optimal-balanced` | 1.565572 / 1.540560 | +1.598% | +1.563% | 0 / 0 |
| `node26-wall-optimal-hot` -> `node26-wall-lazy-hot` | 1.586076 / 1.583282 | +0.176% | -0.528% | 0 / 0 |
| `node25-wall-optimal-balanced` -> `node25-wall-lazy-balanced` | 1.540560 / 1.571561 | -2.012% | -1.470% | 0 / 0 |

## Completed-step wall intervals

| Baseline -> candidate | Steps 9-64 baseline / candidate (s) | Reduction | Steps 9-128 reduction |
|---|---:|---:|---:|
| `node25-wall-baseline-hot` -> `node25-wall-metadata-hot` | 1.925139 / 1.911867 | +0.689% | +0.356% |
| `node26-wall-baseline-native` -> `node26-wall-metadata-native` | 1.969410 / 1.930418 | +1.980% | +0.909% |
| `node25-wall-metadata-balanced` -> `node25-wall-optimal-balanced` | 1.911231 / 1.886417 | +1.298% | +1.235% |
| `node26-wall-optimal-hot` -> `node26-wall-lazy-hot` | 1.929997 / 1.594916 | +17.362% | +16.758% |
| `node25-wall-optimal-balanced` -> `node25-wall-lazy-balanced` | 1.886417 / 1.583077 | +16.080% | +16.453% |

## Isolated MoE and planner measurements

The compact microbench uses real eight-rank NCCL token exchange and replica forward/backward, but no FSDP model/optimizer. Three paired warmups and seven alternating measured pairs; each timing is max across all ranks. It is not whole-training acceleration.

| Experts / top-k | Routing | Dense / compact (ms) | Time reduction |
|---|---|---:|---:|
| 16 / 2 | hot | 6.758 / 6.444 | +4.65% |
| 16 / 2 | balanced | 4.563 / 3.925 | +13.97% |
| 16 / 2 | random | 6.067 / 5.929 | +2.27% |
| 256 / 8 | hot | 14.877 / 14.860 | +0.12% |
| 256 / 8 | balanced | 12.199 / 11.954 | +2.00% |
| 256 / 8 | random | 12.550 / 12.536 | +0.11% |

Integer-optimal stopping: real eight-GPU public planner, including histogram collection, planning, validation and alias rewrite. Three paired warmups, fifteen alternating measured pairs, max-rank elapsed. All fields match in all six cases on all eight ranks.

| Experts / top-k | Routing | Metadata / optimal mean (ms) | Time reduction |
|---|---|---:|---:|
| 16 / 2 | balanced | 0.846 / 0.505 | +40.33% |
| 16 / 2 | hot | 1.369 / 1.372 | -0.19% |
| 16 / 2 | random | 1.937 / 1.975 | -1.98% |
| 256 / 8 | balanced | 6.740 / 0.731 | +89.16% |
| 256 / 8 | hot | 3.044 / 3.072 | -0.90% |
| 256 / 8 | random | 4.496 / 5.635 | -25.33% |

Repetition with reversed first-arm order and 41 pairs, same inputs and pinned code. The first run's E256 random case contained three long-tail candidate samples (13.6/9.6/7.3ms vs ~4.5ms typical), so both means and medians are retained. The change in random-case mean sign is not evidence of a stable random-routing gain.

| Experts / top-k | Routing | Parent / optimal mean (ms) | Parent / optimal median (ms) | Mean reduction |
|---|---|---:|---:|---:|
| 16 / 2 | balanced | 0.814 / 0.514 | 0.813 / 0.490 | +36.90% |
| 16 / 2 | hot | 1.366 / 1.363 | 1.343 / 1.352 | +0.22% |
| 16 / 2 | random | 1.659 / 1.638 | 1.623 / 1.627 | +1.23% |
| 256 / 8 | balanced | 5.717 / 0.695 | 5.623 / 0.680 | +87.84% |
| 256 / 8 | hot | 3.165 / 3.229 | 2.892 / 2.773 | -2.02% |
| 256 / 8 | random | 5.257 / 4.747 | 4.478 / 4.469 | +9.70% |

## Profiles

Profiles are PyTorch Chrome traces, not Nsight .nsys-rep files. Each profile launch runs 16 actual steps; the native schedule [5,6) captures one active step with four MoE calls/rank. CPU `user_annotation` must not be mixed with `gpu_user_annotation`. Inclusive parent/child or concurrent GPU durations cannot be added to obtain step time.

| Inclusive CPU region, mean across eight ranks (ms/captured step) | Baseline | Compact | Bulk metadata | Final scalar-only |
|---|---:|---:|---:|---:|
| `build_ep_balance_plan` | 22.576 | 22.328 | 17.087 | 16.441 |
| `_allocate_replica_rows` | 5.630 | 5.595 | 0.306 | 0.305 |
| `token_pre_all2all` | 3.533 | 2.749 | 2.713 | 2.733 |
| `tokens_post_all2all` | 7.570 | 4.314 | 4.206 | 4.222 |
| `generate_weights_idx` | 2.552 | 0.000 | 0.000 | 0.000 |

## Correctness and limits

- Compact + host metadata related suite: 228 passed, 4 skipped on the controller (CPU environment). This is not a full CUDA/NPU regression pass. Final independent review: safe; 26 additional independently run tests and quality checks passed.
- Integer-optimal planner suite: 33 passed; independent boundary/hotspot/random subset: 10 passed, safe review. CPU prototype retained exact planner results for 320 cases. Host bulk metadata retained 120 exact complete plans and 60 scalar-reference allocation cases.
- Final combined branch related suite: 233 passed, 4 skipped on the controller. These skipped accelerator prerequisites are not GPU/NPU acceptance evidence.
- Scalar-only monitor change: 42 monitor tests passed, including Gloo reductions/failure retry, exact scalar equivalence and image-consumer selection; independent full monitor review also passed 42 tests and quality checks. The controller's offline-WandB test skipped on missing import prerequisites; the GPU-node offline smoke result is retained separately.
- Final scalar-only branch related suite: 235 passed, 4 skipped on the controller. The GPU-node offline WandB test initially skipped because sentry_sdk was missing. Installing only sentry-sdk==2.0.0 into a smoke-only target directory (matching WandB 0.22.0's declared minimum), with no training-environment change, allowed the retry to pass: 1 passed. WANDB_MODE=offline and WANDB_ERROR_REPORTING=false were forced; no live run was uploaded.
- Compact real-GPU microbench: 48 case/rank records, maximum relative-L2 error 0 for output, input, router-weight, gate/up and down gradients under deterministic algorithms. This does not establish bitwise identity for every training parameter/checkpoint or another backend.
- The first non-deterministic microbench attempt failed the 1e-3 input-gradient threshold. Preserved AA/AB/BB investigation showed the same BF16 accumulation variability in unchanged baseline repeats. Enabling the deterministic mode already used by full training produced zero differences without relaxing the threshold. Failure artifacts are retained, not discarded.
- Hot-route telemetry: before/after physical imbalance 1.5 -> 0; four-layer total 24 active replica events and 98,304 moved routed occurrences per step. Balanced routes create zero replicas. These counters are EP-global, not multiplied again by EP size.
- The toy model has a large vocabulary projection relative to routed expert compute. Local kernel/planner reductions do not establish production-model throughput, multi-node scaling, or NPU performance. No changes to HSDP, GBS or model dimensions are credited as code gains.
- No universal feature-on non-degradation claim is made: inspect the same-config disabled comparisons and Failed gates section. The upstream Ascend acceptance and PR merge requirements remain outside the evidence established here.

## Reproduction and artifacts

Shared artifact root: `/volume/tzhen/tmps/veomni-ep1038-perf-momo-0916`.
`comparison.json` and `compare_runs.py` preserve all paired metrics and gates; `results/<label>/launch.json`, `metrics.json`, `exit.json`, `run.log` retain source pins, raw series and process exit status. `run_case.py` specifies every workload argument; `train_fixture.py` defines max-rank compute timing; `train_fixture_wall.py` defines completion throughput.
`microbench/`, `microbench-aa/`, `planner-gpu/`, `bulk-metadata-cpu.json`, and `optimal-stop-prototype.json` retain the separate measurements. Profile traces are in `results/*-profile8/profile/`. `WORKLOG.md` records queue ownership and keeper restoration requirements.

## Failed gates

- `node26-disabled-hot` -> `node26-metadata-hot`: loss gate True, grad-norm gate False, telemetry gate not applicable; performance not accepted.
- `node26-disabled-native` -> `node26-metadata-native`: loss gate False, grad-norm gate False, telemetry gate not applicable; performance not accepted.
- `node25-wall-disabled-hot` -> `node25-wall-metadata-hot`: loss gate True, grad-norm gate False, telemetry gate not applicable; performance not accepted.
- `node26-wall-disabled-native` -> `node26-wall-metadata-native`: loss gate False, grad-norm gate False, telemetry gate not applicable; performance not accepted.
- Disabled-control diagnosis (hot): original enabled baseline and optimized enabled run have exact loss curves=True and exact grad-norm curves=True; the original enabled baseline already fails the strict disabled grad-norm gate at 17 steps, starting at step 106. Full indices are in comparison.json. This distinguishes the existing feature-on/off difference from the optimization A/B; it does not waive the failed acceptance gate.
- Disabled-control diagnosis (native): original enabled baseline and optimized enabled run have exact loss curves=True and exact grad-norm curves=True; the original enabled baseline already fails the strict disabled grad-norm gate at 63 steps, starting at step 4. Full indices are in comparison.json. This distinguishes the existing feature-on/off difference from the optimization A/B; it does not waive the failed acceptance gate.

## Resource restoration

All declared training pairs and short profiles completed with exit code 0.
The original 24-GPU keeper was restored on test25/test26/test27 as
`warmup-continuous24-momo-0913-a00002` (generation 2). All 24 GPU processes were
checked as descendants of the three current supervisors. Optimizer steps 1-4
completed with no NaN or skipped iterations, and all three stages entered the
existing early-send schedule (32/32/40 early jobs, zero pre-send jobs).

Runtime module hashes and pins remain Megatron
`c0b4c2d935571a589da74dbe202a6902f7a7b167` and kernels
`8003de5c44350aa80c598e0a50c99f1e8442051d`: 48 layers, seq 524288, GBS8/MBS1,
TP8/PP3/VPP4/EP8/CP1/DP1, no recompute. The supervisor retains its existing
performance-run policy of reloading the initial checkpoint for a new attempt,
not resuming optimizer/RNG state from step 2540. It remains running with automatic
retry enabled. test28 and the old 16/32-GPU stop flags were not touched.

Evidence: `keeper-restoration.json` and `keeper-gpu-node{0,1,2}.json` in the shared
artifact root, plus the keeper's per-node provenance and training logs. The
initial restored steps are a liveness check, not a new steady-state MFU benchmark.
