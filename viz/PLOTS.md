# Plot Priorities

## Numereng recommended plots

Prioritize the dashboard by decision value, not by total metric count.

### Primary

- cumulative `corr_ender20`
- per-era `corr_ender20`
- cumulative `bmc_ender20`
- per-era `bmc_ender20`
- cumulative `corr_native`
- per-era `corr_native`
- fold snapshot chart for:
  - `corr_native.fold_mean`
  - `corr_ender20.fold_mean`
  - `bmc_ender20.fold_mean`

### Secondary

- cumulative `mmc_ender20`
- per-era `mmc_ender20`
- cumulative `corr_delta_vs_baseline_ender20`
- cumulative `corr_with_benchmark`
- cumulative `baseline_corr_ender20`

### Diagnostics

- cumulative / per-era `fnc_ender20`
- cumulative / per-era `fnc_native`
- cumulative / per-era `feature_exposure`
- cumulative / per-era `max_feature_exposure`

### Advanced / debug

- `bmc_native`
- `mmc_native`
- `baseline_corr_native`
- `corr_delta_vs_baseline_native`
- `cwmm`

## Vendored Numerai example recommendations

The vendored Numerai example is narrower and more BMC-centric.

### Metrics it emphasizes

- primary: `bmc_last_200_eras.mean`
- secondary: `bmc_mean`
- sanity checks: `corr_mean`, `avg_corr_with_benchmark`
- stability checks: `sharpe`, `max_drawdown`

### Plots it recommends generating

- standard `show_experiment` plot with:
  - cumulative per-era `corr`
  - delta vs baseline cumulative `corr`
  - cumulative per-era `BMC`
- `plot_benchmark_corrs` when comparing official benchmark model columns

### Short comparison

- Numerai example: optimize for recent `BMC`, sanity-check `corr`
- Numereng: lead with `corr_ender20` and `bmc_ender20`, keep FNC/exposure in diagnostics, keep helper metrics secondary

## Prospective plots

Additional experiment-level comparison charts that would add real decision value:

### Benchmark similarity sanity chart

- `bmc_last_200_eras` vs `avg_corr_with_benchmark` or `corr_with_benchmark`
- useful for spotting runs that are too benchmark-like despite decent `corr`

### Recent vs full BMC chart

- `bmc_mean` vs `bmc_last_200_eras.mean`
- useful for separating improving runs from fading runs

### Seed stability chart

- one point per target-horizon
- x = seed mean of `bmc_last_200_eras`
- y = seed std or spread
- useful because many experiments are intentionally multi-seed

### Payout value vs distinctiveness chart

- `corr_ender20` vs `bmc_ender20`
- useful as a cleaner payout-plane complement to the `corr_ender20` vs `mmc_ender20` proxy view

### Feature robustness chart

- `corr_ender20` vs `fnc_ender20`
- only when feature-neutral metrics are available
- useful for separating fragile payout-plane winners from robust ones

### Best next additions

- benchmark similarity sanity chart
- seed stability chart
