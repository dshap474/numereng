# ChatGPT Pro Prompt: Numereng Experiment Analysis

You are analyzing a completed Numerai Classic research experiment produced by the local `numereng` workflow. Treat the attached pack as the complete evidence bundle for one experiment: the finalized `EXPERIMENT.md` narrative plus a run table built from manifest-listed scoring artifacts.

Your job is to produce a rigorous second-pass research readout. Do not merely restate the table. Identify what the experiment actually teaches, what remains uncertain, which conclusions are supported by the evidence, which conclusions are weak, and what the next experiment or production handoff should be.

## Numerai Context

Numerai Classic is a financial machine-learning tournament. Participants build models that rank a universe of stocks using obfuscated, point-in-time tabular features. Numerai combines submitted predictions into meta models that inform trading. Validation performance is useful but not sufficient: live performance can degrade, and experiments should be interpreted with overfitting, era structure, target choice, benchmark correlation, and uniqueness in mind.

Rows in the dataset represent stocks within eras. Eras are the time unit that matter most for validation. Metrics are usually computed per era and then summarized across eras with means, standard deviations, Sharpes, and drawdowns. A model that performs well in aggregate but concentrates gains in a few eras, or has high drawdown, may be less useful than a steadier model with a slightly lower mean.

Targets are future stock-specific returns over different horizons and residualization schemes. Numerai provides a main target plus many auxiliary targets. Auxiliary target models can outperform or complement the main target, but target choice can dominate the result. A 20-day target and a 60-day target may behave very differently. Treat target families and horizons as central experimental variables, not incidental config labels.

Features are encrypted stock-market signals. They are point-in-time and designed to avoid leakage, but individual features can be unstable. Models that depend too heavily on simple feature exposure can look good in validation and fail live. Feature-neutral and benchmark-relative diagnostics matter because the tournament rewards useful, robust signal rather than generic replication of common model behavior.

## Metric Context

`CORR` measures Numerai-style correlation between predictions and targets. Predictions are rank-normalized; the tails matter. Strong CORR is useful, but CORR alone can reward behavior that overlaps with common benchmark or meta-model signal.

`MMC` measures Meta Model Contribution: the model's unique contribution after neutralizing to the stake-weighted meta model. Positive MMC suggests additive uniqueness, but MMC can be noisy and sparse. Do not overfit only to MMC unless the experiment was explicitly designed for uniqueness.

`BMC` measures Benchmark Model Contribution: target covariance after neutralizing against benchmark model predictions. In this repo's experiment workflow, BMC and especially late-era BMC are treated as high-value evidence because they ask whether the model adds signal beyond benchmark-style behavior.

`bmc_last_200_eras_mean` is the primary research metric in these numereng experiments unless the report says otherwise. It focuses on more recent validation eras and is intended to reduce the chance that older-regime performance drives the conclusion. `bmc_mean` is the default tie-break. Use `corr_mean`, `corr_sharpe`, `mmc_mean`, `cwmm_mean`, FNC metrics, and drawdown as supporting context.

`FNC` measures feature-neutral correlation. It helps distinguish real predictive signal from simple linear feature exposure. Strong FNC can support robustness claims; missing FNC should limit claims about feature-neutral robustness.

`CWMM` is correlation with the meta model. High CWMM can mean the model agrees with common meta-model signal; that can be useful for CORR but may reduce uniqueness. Interpret CWMM together with BMC and MMC.

`max_drawdown` and metric standard deviations matter. Avoid choosing a run solely by the highest mean if it has unstable era behavior, poor supporting metrics, or bad drawdown relative to alternatives.

## How To Analyze This Pack

Use the experiment narrative as the researcher's current interpretation, not as unquestionable truth. Use the run table as the artifact-backed evidence. Preserve exact run IDs, config names, targets, and metrics when making concrete claims.

Prioritize:

- whether the stated hypothesis was supported, partially supported, or rejected
- whether the winner is a single run, target family, seed family, ensemble candidate set, or "no champion"
- whether the metric pattern is robust across seeds, targets, horizons, and supporting diagnostics
- whether high CORR conflicts with weak BMC/MMC, suggesting benchmark overlap or non-unique signal
- whether positive BMC is broad enough to justify follow-up work or too isolated to trust
- whether any result looks like an outlier that needs confirmation before promotion
- what the next experiment should test, with the smallest useful change

Be skeptical about validation overfit. A completed validation matrix is evidence, not proof. Do not recommend a live submission or champion handoff unless the artifact evidence supports it and the experiment's own rules allow it. If a champion is absent, explain whether that is appropriate.

## Desired Output

Return a structured analysis with:

1. Executive conclusion: the strongest supported takeaway in a few sentences.
2. Evidence-backed findings: bullets tied to exact metrics, targets, or run IDs.
3. Winner/candidate assessment: single-run, target-family, ensemble, or no-champion recommendation.
4. Risks and weak claims: what not to over-believe.
5. Next experiment: the most useful follow-up with a clear rationale.
6. Questions you would ask before production or submission.
