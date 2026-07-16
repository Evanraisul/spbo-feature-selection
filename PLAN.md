# Plan: Integrating Epsilon-Greedy & Levy Flight into SPBO

## Objective
Improve exploration/exploitation in the existing Student Psychology-Based Optimization (SPBO) feature-selection by adding:
- An epsilon-greedy exploration policy
- Levy-flight steps (Mantegna) during exploration

## Short analysis
- SPBO already separates population into `ga`, `gb`, `av`, `rd` groups and applies different update rules.
- Best integration points: the Average (`av`) and Random (`rd`) student update rules where exploration/exploitation choices are meaningful.
- Use epsilon-greedy to choose between exploitation (existing behavior) and exploration (random/Levy perturbation). Use Levy steps to permit occasional long jumps.

## Design decisions
- Epsilon (`eps`): probability of exploration vs exploitation. Default `0.1`.
- Levy: implemented via Mantegna's algorithm. Parameter `levy_beta` default `1.5`.
- Levy amplitude `levy_A` scales the Levy step; default `1.0`.
- Levy-on probability `Pthreshold`: when exploring, use Levy step if `Plevy >= Pthreshold` (default `0.6`).
- Defaults are exposed through `opts` passed to `jfs` (keys: `eps`, `levy_beta`, `levy_A`, `Pthreshold`).

## Implementation notes
- Modified file: `final_spbo_rf.py`.
  - Added `levy_flight(dim, beta)` using Mantegna.
  - Added `update_average_student_epsilon(...)` and `update_random_student_epsilon(...)` implementing epsilon-greedy + Levy-enhanced exploration.
  - In `jfs`, wired new opts keys and replaced the old `update_average_student` and `update_random_student` calls with epsilon-aware versions.
- No changes required to `SPBO_RF.py` to run; you can tune `opts` there before calling `jfs`.

## Experiment plan
1. Baseline: run the existing SPBO (with our defaults) and record:
   - Convergence curve (`c`), final fitness, number of selected features, train/test accuracy and classification report.
2. Ablation grid: vary one parameter at a time (recommended values):
   - `eps`: [0.01, 0.05, 0.1, 0.2]
   - `levy_beta`: [1.2, 1.5, 1.8]
   - `levy_A`: [0.1, 0.5, 1.0, 2.0]
   - `Pthreshold`: [0.4, 0.6, 0.8]
3. For each config, run 5 independent trials (different random seeds) and report mean ± std of metrics.
4. Metrics: best fitness, convergence curve, number of selected features, train/test accuracy, precision/recall/F1, and runtime.

## How to run (quick)
Edit `SPBO_RF.py` and set `opts = {'k':k,'fold':fold, 'N':N, 'T':T, 'mn':mn, 'mx':mx, 'eps':0.1, 'levy_beta':1.5, 'levy_A':1.0, 'Pthreshold':0.6}` then run:

```bash
python3 SPBO_RF.py
```

## Notes & next steps
- If experiments show improvements, consider:
  - Tuning `eps` as a schedule (decay over iterations).
  - Applying Levy only to a subset of dimensions (feature-wise scaling).
  - Adding logging to save convergence and selected subsets per run.

---
Created to guide reproducible experiments for SPBO + epsilon-greedy + Levy.
