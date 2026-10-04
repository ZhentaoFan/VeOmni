# EP balancing enabled versus disabled: verification record

> Historical ledger: statements, commands, environments and measurements below
> refer to the recorded experiment revision, not current rebased-source validation.
> See the [evidence index](qwen3_5_moe_ep_evidence.md) for current status and later results.

Status: final production comparisons and 24-GPU keeper restoration verified.

The requested local on/off comparison is complete. The numerical repair passes
all six 128-step comparisons without changing original-off curves. Performance
is workload-dependent: E256 hotspot saves about 1.65% time averaged across AB/BA,
while E256 native costs about 3.14% more time. There is no general native-route
speedup, and this is not evidence for enabling balancing globally.

## Question and frozen comparison

The requested comparison is balancing enabled versus disabled, not old-enabled
versus new-enabled. The disabled numerical algorithm is unchanged. Heatmap
rendering savings from the previous task are present on both sides and are not
credited to balancing.

Production code: `fix/issue-1038-ordered-wgrad`, `aa12260`, in `ordered/`.
All paired jobs use the same code, seed-0 checkpoint, fixed dummy data, BF16
parameters/activations, FP32 FSDP reductions, GBS8/MBS1, EP8/FSDP2/SP1, eight H200
GPUs on one host, sequence length 2048, no recompute, one replica slot per rank,
monitor interval 1 and WandB disabled. Only enablement, output location and port
differ within a pair. The 256-expert fixture has four full-attention layers,
H2048/I512/top-8; it is not the full 40-layer Qwen3.5-35B model or a convergence run.
The 16-expert/top-2 fixture is kept as a separate negative/overhead control.

Hotspot replay is applied identically to both arms: E16 IDs [0,2] (two original
owners); E256 IDs [0..7] (one original owner). Native routing is unmodified top-k.
These workloads must not be combined into a single general speedup claim.

Each arm runs 128 optimizer steps. The primary predeclared timing window is
steps 9-64 (56 intervals), with steps 9-128 (120 intervals) also reported.
Differences between rank-0 completed-step timestamps include the next data fetch, train step, monitoring,
logging and identical measurement overhead. The inherited harness adds two
device synchronizations and a max-rank elapsed-time reduction per step; therefore
this is instrumented complete-step throughput, not an unsynchronized native run.
Max-rank forward/backward/optimizer time is recorded separately and excludes data
and end callbacks. Time reduction is `1 - on/off`; throughput gain is `off/on - 1`.

All 128 loss and gradient-norm values must pass the unchanged gate
`abs(on-off) <= 1e-4 + 1e-3 * abs(off)`. Missing, failed or nonfinite evidence must
fail the reporting process. Replica participation and moved tokens are checked;
turning the balancer into a no-op cannot count as balancing benefit.

## Numerical repair

Three distinct issues were isolated with matched-state, repeated-execution tests:

1. A FP32 custom-backward result returned through a BF16 concatenation input is
   rounded before the preceding autograd node sees it. Replica reduction now
   stays inside the merged expert backward boundary until the final owner cast.
2. Physical alias sorting changes duplicated-input and top-k accumulation order.
   The enabled path now restores the original logical expert order.
3. Adding independently computed FP32 weight-gradient partials still changes the
   original GEMM reduction tree. Merely lowering local relative-L2 error did not
   prevent 128-step Adam/router trajectory divergence.

The final CUDA implementation passes an FP32 accumulator along consecutive
original token prefixes, then lets the owner continue with its suffix. Replica
boundaries align to the configured BLOCK_K least common multiple, and every
segment uses the original owner's kernel configuration key. The existing
histogram collective rejects inconsistent tile configurations across EP ranks.
The disabled GEMM path remains untouched; this is not a higher-precision baseline.

FP64 canonical-reduction prototypes were investigated but rejected as the final
solution: they changed disabled-side numerical results, and higher precision alone
is not a proof of order invariance. Their raw artifacts remain for audit.

## Evidence before final training reruns

- CPU/Gloo core regression: 36 passed; final broader related suite 280 passed/20 skipped.
- Real CUDA seeded-GEMM tail tests: 16 passed (including empty suffix and M/N tails).
- Final-source eight-GPU matched-state diagnostics: 72 case/rank records;
  output, input/router gradients and both weight gradients all elementwise exactly equal;
  repeated enabled/enabled and disabled/disabled results also equal.
- Independent four-rank CPU/Gloo verification: seven prefix/empty/multi-chain/
  tail/config-selection cases plus metadata-consensus tests passed. The finite
  cancellation example [2^60,1,0,-2^60] returns the original 0; an incorrectly
  reordered partial-sum control returns 1.
- Prefix prototype 128-step E16 training: hotspot and native loss/grad-norm curves
  each exactly matched off; off also exactly matched the earlier unmodified off.
  Hotspot wall9-128 was 1.596201 -> 1.576365 s; native was 1.575102 -> 1.613036 s.
  These are prototype results, not a replacement for production reruns below.
- Earlier FP32-partial implementation failed strict native training gates. Those
  measurements do not qualify as valid balancing speedups.

This evidence is scoped to the tested BF16 kernels, shapes and duration. It is
not an all-backend or unrestricted-input bitwise guarantee. The diagnostic uses
numeric differences, not a bytewise tensor comparison (e.g. signed zero). NPU was not retested.

## Final production results

E256 runs use AB then BA on the same host for each routing mode; E16 uses a
fresh AB pair. Sources and full commands are recorded in every launch.json.

| Experts / routing | Order | Off wall9-128 (s) | On wall9-128 (s) | Time reduction 9-64 | Time reduction 9-128 |
|---|---|---:|---:|---:|---:|
| E256 hotspot | AB | 1.617131 | 1.598373 | +0.929% | +1.160% |
| E256 hotspot | BA | 1.638325 | 1.603453 | +2.291% | +2.129% |
| E256 native | AB | 1.590495 | 1.650438 | -2.377% | -3.769% |
| E256 native | BA | 1.588800 | 1.628791 | -3.083% | -2.517% |
| E16 hotspot | AB | 1.573662 | 1.572843 | -1.050% | +0.052% |
| E16 native | AB | 1.582967 | 1.610994 | -1.433% | -1.771% |

All six comparisons have 128 exactly matching recorded loss and
gradient-norm values, and the off curves match the historical original-off runs.
The two-round E256 mean wall9-128 is 1.627728 -> 1.600913 s for hotspot
(1.647% time reduction, 1.675% throughput increase), and 1.589648 -> 1.639615 s
for native (3.143% more time, 3.047% lower throughput). These are measured means
and repeat ranges, not a statistical-significance or full-model claim.
E16 hotspot changes sign between the two predeclared windows and its long-window
difference is effectively zero: it does not pass the observed-net-benefit check.
E16 native regresses in both windows. The final matrix is 12 runs x 128 steps =
1536 optimizer steps, with no failed or outlier steps removed from the windows.

E256 replica execution is real, not a no-op: hotspot has 28 replica events every step
and mean imbalance 1.75 -> 0; native has 17-31 events per step and mean imbalance
0.398501 -> 0.035056. Native becomes more balanced but slower end to end. Thus
load balance alone is insufficient evidence of performance benefit, and enabling
this implementation globally is not justified by these results.

The numerical fixes are useful independently of throughput. Likely optimization
targets include replica weight/gradient communication, host-side planning and
ordered weight-gradient scheduling; this run does not attribute measured wall
regression quantitatively to any one of them. No native speedup is claimed.

Raw evidence under `/volume/tzhen/tmps/veomni-ep1038-onoff-momo-0916`:
results/, production-comparison.json, final-audit.json, diagnosis-prefix-final/, prefix-core-tests.log,
diagnosis-prefix-final-seeded-gpu-tests.log and verification-prefix/.

Recheck all explicitly selected production pairs with `bash compare_production.sh`.
`compare_onoff.py` separates `precision_gate_pass` from `observed_net_benefit`;
`--require-net-benefit` additionally fails a precise but slower comparison.
Historical failures and the negative-performance control are retained, not hidden.

## Resource restoration

The original 24-GPU keeper on test25/test26/test27 is restored as
`warmup-continuous24-momo-0913-a00003` (generation 3). By 2026-09-16 05:04:05 +08,
the audit verified all 24 GPU processes as descendants of the three supervisors,
fresh runtime module hashes, and five completed optimizer steps. The final
snapshot's last three steps have no skipped or NaN iterations. Step 3 activates
the original early-send schedule (32/32/40
jobs across PP ranks 0/1/2; no remaining pre-send jobs).

Pins remain Megatron `c0b4c2d935571a589da74dbe202a6902f7a7b167` and kernels
`8003de5c44350aa80c598e0a50c99f1e8442051d`; the original 48-layer/512K/GBS8,
TP8/PP3/VPP4/EP8 setup and no-recompute configuration are unchanged. Test28 was
not used. Only this task's STOP was archived as `STOP.veomni-onoff-20260916`;
older stop flags were not removed. Proof is in `keeper-restoration.json` and
`keeper-gpu-node{0,1,2}.json` under the artifact root.

This restarts the performance keeper from its original checkpoint, not from the
paused generation's optimizer/RNG state. The keeper continues beyond this audit;
its first cold steps are health checks, not a new warmup MFU benchmark.
