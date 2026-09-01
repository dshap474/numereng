# Anti-Patterns

12 Numerai-specific gotchas consolidated from research briefs, Phase 7 learnings, and community experience. Review before and during each campaign.

## 1. Wrong Payout Target

**Mistake:** Evaluating ensembles on `target` (alias for `target_cyrusd_20`) instead of the actual payout target (`target_ender_20`).

**Impact:** Training Sharpe doesn't transfer to tournament payout. Models optimized for cyrusd may underperform on ender.

**Fix:** Always set `target: target_ender_20` in evaluation configs. Use the `--metric payout` flag for ensemble optimization.

## 2. Cherry-Picking Seeds

**Mistake:** Training many seeds and only keeping the best-performing ones, rather than averaging all seeds.

**Impact:** Selects for lucky variance rather than reducing it. Cherry-picked models regress to mean in live.

**Fix:** Pre-commit seed list (42, 123, 456, 789, 1011) before training. Average ALL seeds, never select.

## 3. Over-Neutralization

**Mistake:** Using neutralization proportion > 0.7 or blindly maximizing MMC.

**Impact:** Removes real signal along with common signal. CORR drops below 0.005, destroying payout.

**Fix:** Start with 0.5. Only increase to 0.7 if feature exposure is still > 0.10. Monitor live CORR carefully after submission.

## 4. High-Correlation Ensembles

**Mistake:** Ensembling models with pairwise prediction correlation > 0.85.

**Impact:** Redundant models add compute cost but no diversity benefit. Ensemble is effectively a single model.

**Fix:** Check pairwise correlations before ensembling. Drop redundant models (keep the stronger one). Target mean pairwise correlation 0.40-0.70.

## 5. Separate Seed Submissions

**Mistake:** Submitting individual seeds as separate Numerai models instead of averaging.

**Impact:** Each model gets evaluated independently, increasing stake risk. No variance reduction benefit.

**Fix:** Always average seeds into one prediction set, submit one model per strategy.

## 6. Ignoring BMC

**Mistake:** Optimizing only for CORR and ignoring BMC (Benchmark Model Contribution) in the payout formula.

**Impact:** Since payout = 0.75*CORR + 2.25*BMC, BMC has 3x the weight of CORR. High-CORR models that are identical to the benchmark earn 0 BMC.

**Fix:** Optimize for the full payout metric. Include diverse model types (especially Ridge) that differ from the LGBM-heavy benchmark.

## 7. Overfitting Forward Selection

**Mistake:** Running forward selection on all validation eras without a held-out test set.

**Impact:** Selected weights overfit to validation. The 50%+ Sharpe degradation from hillclimb to holdout is typical.

**Fix:** Use hillclimb/holdout split. Evaluate final ensemble on holdout eras only. If degradation > 60%, use equal-weight blend.

## 8. Feature Set Too Large Without Regularization

**Mistake:** Training on `all` ~2,376 features without sufficient `colsample_bytree` regularization.

**Impact:** Model memorizes correlated feature noise. Poor out-of-sample generalization.

**Fix:** Always use `colsample_bytree: 0.1` (or `rsm: 0.1` for CatBoost). This is non-negotiable for Numerai's correlated feature space.

## 9. Parallel Local Training

**Mistake:** Running multiple training jobs simultaneously on the local machine (36GB RAM, ~20-30GB in use).

**Impact:** OOM crash. Machine becomes unresponsive. Training results may be corrupted.

**Fix:** Run ONE local training job at a time. Use cloud backends for parallel or full-data training.

## 10. Training on Main EC2

**Mistake:** Running ML training on the main EC2 instance (3.7GB RAM).

**Impact:** OOM crash. Disrupts other services running on the instance.

**Fix:** Use `--backend ec2 --tier r7i.4xlarge` to launch a dedicated spot instance. Or use RunPod for GPU training.

## 11. Ignoring Walk-Forward CV

**Mistake:** Using random cross-validation instead of temporal (walk-forward) CV.

**Impact:** Look-ahead bias. Model sees future data during training. Validation metrics are inflated.

**Fix:** Always use era-aware walk-forward CV with proper embargo gaps (8 eras for 20-day targets, 16 for 60-day). Set `training.engine.profile: purged_walk_forward`; numereng applies official defaults automatically.

## 12. BMC Volatility Blindness

**Mistake:** Over-optimizing for current BMC composition without considering that Numerai updates benchmark models periodically.

**Impact:** A model heavily optimized for the current benchmark may underperform when the benchmark changes.

**Fix:** Maintain model diversity. Don't stake everything on high BMC. Track BMC composition changes in the Numerai forum. Consider running an equal-weight blend alongside your optimized blend as insurance.

## Quick Reference Card

| # | Anti-Pattern | One-Line Fix |
|---|-------------|-------------|
| 1 | Wrong payout target | Use `target_ender_20` for evaluation |
| 2 | Cherry-picking seeds | Pre-commit seeds, average ALL |
| 3 | Over-neutralization | Start at 0.5, monitor live CORR |
| 4 | High-corr ensembles | Pairwise < 0.85, target 0.40-0.70 |
| 5 | Separate seed submissions | Average seeds, submit one model |
| 6 | Ignoring BMC | Optimize full payout (0.75C + 2.25BMC) |
| 7 | Overfitting forward selection | Use hillclimb/holdout split |
| 8 | No feature regularization | `colsample_bytree: 0.1` always |
| 9 | Parallel local training | One job at a time locally |
| 10 | Training on main EC2 | Launch spot instances |
| 11 | Random CV | Walk-forward with embargo |
| 12 | BMC volatility | Maintain diversity, track changes |
