# ANN-MCS framework -- pseudocode

**Accompanying manuscript:** Ljubinković et al. (2026), *ANN-based Monte
Carlo simulation framework for reliability assessment of steel structures*
(submitted to Structures; under revision).

**Contact:** Filip Ljubinković — filj@uc.pt. Full citations are provided
in `README.md` and `references.bib`.

This is the step-by-step outline of the ANN-based Monte Carlo simulation
(ANN-MCS) framework implemented in this repository, matching the structure
of `CWC_ANN_MCS.py` and `I_girder_ANN_MCS.py` (each script is labelled with
`[STEP n]` comments corresponding to the steps below).

```
1. Define the stochastic input variables and their distributions.

2. Generate n input combinations using Latin Hypercube Sampling (LHS).
   Run the FE model for each combination to obtain resistance values.
   (The FE analysis itself is external to this code; the scripts generate
   the LHS sampling plan and expect the FE results to be filled in
   afterwards.)

3. For each ANN hyperparameter configuration:
   a. Perform five-fold cross-validation.
   b. Train an additional model using the specified train/test split.
   c. Evaluate that model on its test subset to obtain R2_test, and
      calculate the pooled development-dataset resistance-ratio statistics.

4. Select the ANN using the prescribed selection criteria.

5. Assess its predictions against a separate FE benchmark,
   including resistance statistics and lower-tail quantiles.

6. Generate N new stochastic input combinations using LHS.
   Predict their resistances using the selected ANN.

7. Calculate failure probabilities, reliability indices,
   partial factors, and confidence intervals.
```

## Notes on steps 3-4 (hyperparameter search and model selection)

Step 3 is a configurable grid search over neurons, epochs, batch size,
learning rate, and train/test split. The example settings in both scripts
hold epochs fixed and use 24 combinations; these defaults do not reproduce
the paper's full CWC grid. For every combination in the grid:

- Inputs are standardised using the complete development dataset before
  partitioning, matching the disclosed implementation in the manuscript.
- 5-fold cross-validation is run on the **full** development dataset (not
  just the retained model's training subset), giving an average validation
  R<sup>2</sup> (`Avg_KFold_R2`) across the five folds. Preprocessing is not
  fitted separately within each fold.
- One additional "final" model is trained on a single train/test split of
  the same dataset. This final model (not the fold models, which are
  discarded after cross-validation) is the one evaluated and potentially
  kept.
- `R2_test` is calculated from this retained model's predictions on
  `X_test` compared with `y_test` and written to the grid summary. It is a
  separate diagnostic from `Avg_KFold_R2` and does not alter the selection
  rule below.
- The FEM/ANN resistance ratio (Max, Min, Mean, StDev, CoV%) is computed
  for the final model's predictions over the full dataset.

The retained model's test subset therefore contributes to preprocessing
and configuration selection; it is not an untouched final assessment of
the overall selection procedure.

Step 4 selects one configuration per training-dataset size using the
following rule: discard any configuration whose `Avg_KFold_R2` falls below
a minimum threshold, then rank the remainder primarily by FEM/ANN CoV%
(ascending -- tightest scatter first), then by the K-fold R<sup>2</sup>
deviation from 1 (ascending), then by the summed deviation of
Max/Min/Mean from 1 (ascending). The top-ranked configuration is retained.

## Notes on step 5 (benchmark assessment)

The selected model is evaluated on a separate, independent FE benchmark
set (not used anywhere in steps 3-4), using two complementary diagnostics:

- **Marginal quantiles**: the p-th percentile of the FEM distribution and
  of the ANN distribution are computed independently, for a small set of
  lower-tail probability levels (e.g. 5%, 1%, and the target failure
  probability implied by the reliability target `beta_target`).
- **Tail-conditional ratio statistics**: a single, FEM-defined threshold
  (the FEM p-th percentile) is used to select the same subset of points
  from both the FEM and ANN arrays, and the FEM/ANN ratio statistics
  (mean, CoV%, max, min) are computed on that shared subset. This uses the
  FEM data as the sole reference for "which points count as the tail",
  consistent with the resistance model's benchmark convention used
  elsewhere in the framework.

## Notes on steps 6-7 (Monte Carlo reliability assessment)

Step 6 draws `N` new Latin Hypercube samples directly from the input
variables' defined distributions (as opposed to step 2, which samples a
smaller set of points to be run through the FE model) and predicts their
resistance with the selected ANN -- this is the "Monte Carlo" part of
ANN-MCS: a large resistance sample obtained at negligible computational
cost compared to running the FE model N times.

Step 7 computes, from that resistance sample `R`:

- the observed failure count `NF` at the Eurocode design-resistance
  threshold (named `nominal_resistance` in the code), with
  `Pf_count = NF / N`, and the corresponding count-based reliability index
  `beta_count = -Phi_inverse(Pf_count)` (the direct inverse-normal
  transform of the observed failure probability -- not derived from the
  moment/normal-fit route below). A finite `beta_count` requires
  `0 < Pf_count < 1`. At zero observed failures, the code exports the
  mathematical limit `+inf`; if every case fails, it exports `-inf`.
  These limits are not finite reliability estimates, and zero observed
  failures do not establish zero population failure probability;
- separately, `beta_norm = (mean(R) - R_nominal) / std(R)` and
  `Pf_norm = Phi(-beta_norm)` under the normal-resistance approximation.
  `beta_norm` and `beta_count` are two different reliability indices from
  the same Monte Carlo sample -- one from a normal-distribution fit to
  `R`, the other a direct transform of the observed failure probability -- and
  are not expected to coincide exactly;
- the partial factor `gammaM` via two routes:
  - **Route A** (moment/normal-fit): `Rd = mean(R) - beta_target * std(R)`,
    `gammaM = R_nominal / Rd`;
  - **Route B** (direct empirical evaluation):
    `gammaM = R_nominal / R_{p}`, where `R_p` is the empirical p-th
    percentile of `R` at the target failure probability;
- a nonparametric confidence interval on that empirical quantile (order
  statistics / binomial method), propagated to a confidence interval on
  `gammaM` (Route B).

The rank-based interval assumes independent observations, so it is an
approximate reference interval for the LHS samples. It concerns finite
sample size, not FE-model or ANN-prediction uncertainty. The two routes
evaluate failure probabilities at the design-resistance threshold; the
target probability is used separately to determine the target resistance
and partial factors.
