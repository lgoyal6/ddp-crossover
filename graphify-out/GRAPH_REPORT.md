# Graph Report - worktree  (2026-11-05)

## Corpus Check
- 30 files · ~25,463 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 178 nodes · 284 edges · 17 communities (15 shown, 2 thin omitted)
- Extraction: 98% EXTRACTED · 2% INFERRED · 0% AMBIGUOUS · INFERRED: 7 edges (avg confidence: 0.85)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `38a1b330`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- adaptive_power_sgd_hook
- 05_learnable_train.py
- load_checkpoint
- app.js
- ddp-crossover
- run_final_campaign.py
- 02_ddp_sweep.py
- kaggle_combined.py
- test_final_campaign.py
- kaggle_step5.py
- 06_compare.py
- 01_nccl_bench.py
- 03_report.py
- test_report_keeps_accumulation_and_fixed_rank_arms_separate
- run_all.sh

## God Nodes (most connected - your core abstractions)
1. `main()` - 15 edges
2. `AdaptivePowerSGDState` - 14 edges
3. `adaptive_power_sgd_hook()` - 14 edges
4. `load_checkpoint()` - 12 edges
5. `run_restart_correctness()` - 8 edges
6. `main()` - 8 edges
7. `CifarData` - 7 edges
8. `render()` - 7 edges
9. `save_checkpoint()` - 7 edges
10. `main()` - 6 edges

## Surprising Connections (you probably didn't know these)
- `main()` --indirect_call--> `adaptive_power_sgd_hook()`  [INFERRED]
  05_learnable_train.py → adaptive_powersgd.py
- `main()` --calls--> `AdaptivePowerSGDState`  [EXTRACTED]
  05_learnable_train.py → adaptive_powersgd.py
- `main()` --calls--> `load_checkpoint()`  [EXTRACTED]
  05_learnable_train.py → training_checkpoint.py
- `main()` --calls--> `save_checkpoint()`  [EXTRACTED]
  05_learnable_train.py → training_checkpoint.py
- `test_configuration_and_checkpoint_mismatch_are_rejected()` --calls--> `AdaptivePowerSGDState`  [EXTRACTED]
  tests/test_adaptive_powersgd.py → adaptive_powersgd.py

## Import Cycles
- None detected.

## Communities (17 total, 2 thin omitted)

### Community 0 - "adaptive_power_sgd_hook"
Cohesion: 0.13
Nodes (18): adaptive_power_sgd_hook(), AdaptivePowerSGDState, _move_power_state_to_device(), Any, Tensor, Adaptive, checkpointable per-bucket PowerSGD hook for torch DDP. PyTorch owns…, Lazily relocate checkpoint-loaded CPU state to the active bucket device., Run per-bucket PowerSGD; adapt ranks at a synchronized fixed cadence. (+10 more)

### Community 1 - "05_learnable_train.py"
Cohesion: 0.14
Nodes (15): CifarData, CommunicationMeter, _counted_hook(), main(), _power_sgd_hook(), Path, Tensor, Matched CIFAR-10 training arms for dense, FP16, fixed and adaptive DDP.… (+7 more)

### Community 2 - "load_checkpoint"
Cohesion: 0.18
Nodes (17): _assert_tree_equal(), _run(), _spawn_bounded(), test_adaptive_learnable_smoke_and_full_checkpoint_resume(), test_dense_checkpoint_resume_matches_uninterrupted(), _state(), test_atomic_checkpoint_requires_integrity_and_all_rank_states(), test_compare_detects_randomstate_difference() (+9 more)

### Community 3 - "app.js"
Cohesion: 0.28
Nodes (14): buildTable(), commMeasured(), compact(), css(), ctx, draw(), el(), main() (+6 more)

### Community 4 - "ddp-crossover"
Cohesion: 0.13
Nodes (14): Adaptive PowerSGD and time-to-quality, Bandwidth is reported NCCL-style, not naively, Caveats, ddp-crossover, Frozen calibration and final campaign, Mitigations swept, and why each should move the crossover, Results, The all-reduce fraction is measured, not inferred (+6 more)

### Community 5 - "run_final_campaign.py"
Cohesion: 0.33
Nodes (13): Arm, common(), ensure_gpu(), latest_row(), launch(), main(), Path, Run the preregistered CIFAR-10 DDP calibration and final arm matrix. The final… (+5 more)

### Community 6 - "02_ddp_sweep.py"
Cohesion: 0.22
Nodes (10): build(), fp16_compress_hook(), main(), MLP, n_params(), Step 5.2/5.3/5.4 -- where does the second GPU stop paying, and what moves it.…, Plain stack of square linears, so parameter count is a clean dial. Deliberately…, Cast the bucket to fp16 for the wire, all-reduce, cast back. torch ships… (+2 more)

### Community 7 - "kaggle_combined.py"
Cohesion: 0.36
Nodes (8): diagnose(), launch(), main(), materialize(), Step 5 on Kaggle: where the second GPU stops paying, 2x Tesla T4 over PCIe.…, Report what the container actually gave us before failing on it., single(), _worker()

### Community 8 - "test_final_campaign.py"
Cohesion: 0.44
Nodes (8): Path, _run(), test_calibration_is_cifar10_dense_three_seed_and_preregistered(), test_campaign_orders_gpu_restart_gate_before_calibration_and_quality_matrix(), test_campaign_uses_external_watchdog_without_elapsed_time_flags(), test_final_matrix_matches_accumulation_reference_and_profiles_separately(), test_restart_correctness_dispatches_verifier_without_torchrun(), test_wall_clock_timeout_logs_and_kills_entire_process_group()

### Community 9 - "kaggle_step5.py"
Cohesion: 0.43
Nodes (6): launch(), main(), Kaggle entry point for the 2x T4 study. A Kaggle kernel is a single process, so…, Run a script as a plain 1-GPU process (no distributed env set)., single(), _worker()

### Community 10 - "06_compare.py"
Cohesion: 0.50
Nodes (4): main(), Path, Summarize matched training traces, including arms censored before target., summarize()

### Community 12 - "01_nccl_bench.py"
Cohesion: 0.67
Nodes (3): bench(), main(), Step 5.1 -- raw NCCL collective bandwidth on 2x T4 over PCIe. This is the…

### Community 13 - "03_report.py"
Cohesion: 0.67
Nodes (3): load(), main(), Step 5 gate: the crossover, the all-reduce fraction curve, and what each…

## Knowledge Gaps
- **16 isolated node(s):** `plot`, `ctx`, `VARIANTS`, `state`, `run_all.sh script` (+11 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **2 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `load_checkpoint()` connect `load_checkpoint` to `05_learnable_train.py`?**
  _High betweenness centrality (0.047) - this node is a cross-community bridge._
- **Why does `main()` connect `05_learnable_train.py` to `adaptive_power_sgd_hook`, `load_checkpoint`?**
  _High betweenness centrality (0.046) - this node is a cross-community bridge._
- **Why does `AdaptivePowerSGDState` connect `adaptive_power_sgd_hook` to `05_learnable_train.py`?**
  _High betweenness centrality (0.038) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `adaptive_power_sgd_hook()` (e.g. with `main()` and `_distributed_worker()`) actually correct?**
  _`adaptive_power_sgd_hook()` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `plot`, `ctx`, `VARIANTS` to the rest of the system?**
  _16 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `adaptive_power_sgd_hook` be split into smaller, more focused modules?**
  _Cohesion score 0.13230769230769232 - nodes in this community are weakly interconnected._
- **Should `05_learnable_train.py` be split into smaller, more focused modules?**
  _Cohesion score 0.14130434782608695 - nodes in this community are weakly interconnected._