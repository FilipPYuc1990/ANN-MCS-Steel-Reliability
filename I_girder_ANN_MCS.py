"""
I_girder_ANN_MCS.py
--------------------------------------------------------------------------
Accompanies: ANN-based Monte Carlo simulation framework for reliability
assessment of steel structures (Ljubinković et al., 2026).
Status: manuscript submitted to Structures; under revision, no DOI yet.
Maintainer: Filip Ljubinković, University of Coimbra (filj@uc.pt).
SPDX-License-Identifier: CC-BY-NC-4.0
Licence text: LICENSE. Copyright 2026 Filip Ljubinković.
Full manuscript and related-study citations: README.md / references.bib.

ANN-based Monte Carlo simulation (ANN-MCS) framework for the stochastic
characterisation and reliability assessment of a steel resistance model,
applied here to a welded I-girder cross-section under axial compression.

This script implements the full pipeline described in `pseudocode.md`:

    1. Define the stochastic input variables and their distributions.
    2. Generate n input combinations using LHS; run the FE model for each
       combination to obtain resistance values (the FE analysis itself is
       external to this script -- see STEP 2 below).
    3. For each ANN hyperparameter configuration: 5-fold cross-validation,
       a final train/test model, and the associated performance indicators.
    4. Select the ANN using the prescribed selection criteria.
    5. Assess its predictions against a separate FE benchmark, including
       resistance statistics and lower-tail quantiles.
    6. Generate N new stochastic input combinations using LHS and predict
       their resistances with the selected ANN.
    7. Calculate failure probabilities, reliability indices, partial
       factors, and confidence intervals.

This is a cleaned, configurable implementation of the paper's workflow,
not an exact reproduction of every reported result. Folder bookkeeping,
exploratory cells and in-house plot styling have been removed. The six
input variables match the I-girder application, but the geometry and
statistics below are illustrative and must match the supplied FE data.
For example, the default web thickness differs from WS1 in the paper.

Equivalent script for the CWC application: CWC_ANN_MCS.py (same 7-step
structure; the two are kept separate because the underlying FE problems
differ in dimensionality, FE formulation and resistance mechanism, so a
single generic script would obscure more than it shares).
"""
import os
from itertools import product

import numpy as np
import pandas as pd
from pyDOE import lhs
from scipy.spatial import distance
from scipy.stats import norm, binom
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.optimizers import Adam

# ==========================================================================
# CONFIG -- edit this block for your own cross-section / problem
# ==========================================================================

# ---- STEP 1: stochastic input variables ----
FEATURE_COLUMNS = ["hw", "bf", "tw", "tf", "E", "fy"]
TARGET_COLUMN = "F_FEM"          # FE-computed axial resistance column

# Nominal (deterministic) cross-section geometry and material properties.
NOMINAL_VALUES = {"hw": 540.0, "bf": 300.0, "tw": 15.0, "tf": 30.0,
                   "E": 210000.0, "fy": 460.0}

# Distribution of each input variable: mean = bias_factor * nominal,
# cov = coefficient of variation (JCSS Probabilistic Model Code-style
# values for hot-rolled/welded steel sections). Replace with your own
# cross-section and statistics as needed.
BIAS_FACTORS = {"hw": 0.98, "bf": 0.98, "tw": 0.98, "tf": 0.98, "E": 1.00, "fy": 1.15}
COV = {"hw": 0.009, "bf": 0.009, "tw": 0.025, "tf": 0.025, "E": 0.03, "fy": 0.045}
DIST_TYPE = {k: "normal" for k in FEATURE_COLUMNS}

# ---- data files ----
TRAIN_DATA_FILE = "example_data/Igirder_training_database.xlsx"   # n rows, FEATURE_COLUMNS + TARGET_COLUMN
BENCHMARK_DATA_FILE = "example_data/Igirder_FE_benchmark.xlsx"    # separate, larger FE benchmark set (STEP 5)
RESULTS_DIR = "Igirder_ANN_MCS_Results"

# ---- STEP 3: hyperparameter grid ----
NEURONS_LIST = [8, 16, 32]
EPOCHS_LIST = [1500]
BATCH_LIST = [8, 16]
LR_LIST = [0.05, 0.01]
TESTSIZE_LIST = [0.3, 0.2]
K_FOLDS = 5
RANDOM_STATE = 42

# ---- STEP 4: model selection ----
MIN_KFOLD_R2 = 0.90             # discard configurations below this average K-fold R2

# ---- STEP 5: lower-tail assessment levels ----
BETA_TARGET = 3.04              # EN1990 Annex C target reliability index (component)
TAIL_LEVELS_PCT = [5.0, 1.0, norm.cdf(-BETA_TARGET) * 100]   # % cumulative probability

# ---- STEP 6/7: Monte Carlo reliability assessment ----
N_MCS_SAMPLES = 100_000
CONFIDENCE_LEVEL = 0.95         # for the empirical-quantile confidence interval


# ==========================================================================
# Nominal resistance function (EN1993-1-5 effective-width method for a
# welded I-section under uniform axial compression -- replace with your
# own limit-state / nominal-resistance expression for a different problem)
# ==========================================================================
def nominal_axial_resistance(hw, bf, tw, tf, E, fy):
    Ag = hw * tw + 2 * (bf * tf)
    lam_w = hw / tw
    lam_w_bar = lam_w / (28.4 * np.sqrt(235 / fy) * 2)
    rho_w = (lam_w_bar - 0.055 * 4) / lam_w_bar ** 2
    Aw_eff = hw * tw * rho_w
    lam_f = (bf - tw) / (2 * tf)
    lam_f_bar = lam_f / (28.4 * np.sqrt(235 / fy) * np.sqrt(0.43))
    rho_f = min((lam_f_bar - 0.188) / lam_f_bar ** 2, 1.0)
    Af_eff = bf * tf * rho_f
    Aeff = Aw_eff + 2 * Af_eff
    return Aeff * fy * 1e-3   # kN


# ==========================================================================
# STEP 1 -- helper: sample one variable from its defined distribution
# ==========================================================================
def _lhs_sample_variable(mean, cov, dist_type, n_samples, lhs_points=None):
    if lhs_points is None:
        lhs_points = lhs(1, samples=n_samples).flatten()
    if dist_type == "normal":
        sigma = cov * mean
        return norm(loc=mean, scale=sigma).ppf(lhs_points)
    elif dist_type == "lognormal":
        sigma_ln = np.sqrt(np.log(1 + cov ** 2))
        mu_ln = np.log(mean) - 0.5 * sigma_ln ** 2
        from scipy.stats import lognorm
        return lognorm(s=sigma_ln, scale=np.exp(mu_ln)).ppf(lhs_points)
    else:
        raise ValueError(f"Unsupported distribution type: {dist_type}")


# ==========================================================================
# STEP 2 -- generate n candidate input combinations for the FE analysis
# ==========================================================================
def generate_fe_sampling_plan(n, out_path, candidate_pool=None):
    """Latin Hypercube sample n input combinations. If `candidate_pool`
    (a DataFrame of already-available FE points, e.g. a larger existing
    database) is given, each LHS point is matched to its nearest
    not-yet-used neighbour in that pool instead of being sampled fresh --
    this is useful when drawing a smaller, well-spread sub-sample out of a
    larger existing dataset. Otherwise, points are sampled directly from
    the input distributions and must be run through the external FE model."""
    names = FEATURE_COLUMNS
    design = lhs(len(names), samples=n)

    if candidate_pool is not None:
        pool_vals = candidate_pool[names].values
        scaler = StandardScaler().fit(pool_vals)
        pool_norm = scaler.transform(pool_vals)
        design_scaled = design * (pool_norm.max(axis=0) - pool_norm.min(axis=0)) + pool_norm.min(axis=0)
        used = set()
        selected = []
        for point in design_scaled:
            dists = distance.cdist([point], pool_norm, metric="euclidean")[0]
            for idx in np.argsort(dists):
                if idx not in used:
                    selected.append(idx)
                    used.add(idx)
                    break
        df = candidate_pool.iloc[selected].reset_index(drop=True)
    else:
        cols = {}
        for j, name in enumerate(names):
            mean = BIAS_FACTORS[name] * NOMINAL_VALUES[name]
            cols[name] = _lhs_sample_variable(mean, COV[name], DIST_TYPE[name], n, lhs_points=design[:, j])
        df = pd.DataFrame(cols)
        df[TARGET_COLUMN] = np.nan

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    df.to_excel(out_path, index=False)
    print(f"[STEP 2] {n}-point LHS sampling plan written to {out_path}.")
    return df


# ==========================================================================
# STEP 3 -- grid search: 5-fold CV + final train/test model, per combination
# ==========================================================================
def build_model(n_features, neurons, learning_rate):
    model = Sequential([
        Dense(neurons, activation="relu", input_shape=(n_features,)),
        Dense(neurons, activation="relu"),
        Dense(1, activation="linear"),
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss="mean_squared_error", metrics=["mse"])
    return model


def run_hyperparameter_grid(df_train, results_dir):
    X = df_train[FEATURE_COLUMNS].values
    y = df_train[TARGET_COLUMN].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)   # fit on the full training set, before any split

    rows = []
    grid = list(product(NEURONS_LIST, EPOCHS_LIST, BATCH_LIST, LR_LIST, TESTSIZE_LIST))
    for neurons, epochs, batch_size, lr, test_size in grid:
        lr_str = f"{lr:.3f}".replace(".", "")
        combo_name = f"N{neurons}_E{epochs}_B{batch_size}_L{lr_str}_T{int(test_size*100)}"
        combo_dir = os.path.join(results_dir, combo_name)
        os.makedirs(combo_dir, exist_ok=True)

        kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        fold_r2 = []
        for tr_idx, val_idx in kf.split(X_scaled):
            m = build_model(X_scaled.shape[1], neurons, lr)
            m.fit(X_scaled[tr_idx], y[tr_idx], epochs=epochs, batch_size=batch_size, verbose=0)
            pred = m.predict(X_scaled[val_idx], verbose=0).flatten()
            fold_r2.append(r2_score(y[val_idx], pred))
        avg_kfold_r2 = float(np.mean(fold_r2))

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=RANDOM_STATE)
        final_model = build_model(X_scaled.shape[1], neurons, lr)
        final_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

        # ---- report retained-model accuracy on its test subset ----
        y_pred_test = final_model.predict(X_test, verbose=0).flatten()
        r2_test = float(r2_score(y_test, y_pred_test))

        y_pred_full = final_model.predict(X_scaled, verbose=0).flatten()
        ratio = y / y_pred_full
        model_path = os.path.join(combo_dir, f"ANN_{combo_name}.keras")
        final_model.save(model_path)

        rows.append(dict(
            Folder=combo_name, Neurons=neurons, Epochs=epochs, Batch=batch_size,
            LearningRate=lr, TestSize=test_size, Avg_KFold_R2=avg_kfold_r2,
            R2_test=r2_test,
            FEM_ANN_Max=ratio.max(), FEM_ANN_Min=ratio.min(), FEM_ANN_Mean=ratio.mean(),
            FEM_ANN_StDev=ratio.std(), FEM_ANN_CoV=100 * ratio.std() / ratio.mean(),
            ModelPath=model_path,
        ))
        print(f"[STEP 3] {combo_name}: Avg K-fold R2={avg_kfold_r2:.4f}, "
              f"test R2={r2_test:.4f}, "
              f"FEM/ANN CoV%={rows[-1]['FEM_ANN_CoV']:.3f}")

    summary = pd.DataFrame(rows)
    summary.to_excel(os.path.join(results_dir, "Hyperparameter_Summary.xlsx"), index=False)
    return summary, scaler


# ==========================================================================
# STEP 4 -- select the ANN using the prescribed selection criteria
# ==========================================================================
def select_best_model(summary_df, min_kfold_r2=MIN_KFOLD_R2):
    df = summary_df[summary_df.Avg_KFold_R2 >= min_kfold_r2].copy()
    if df.empty:
        raise ValueError(f"No configuration reached Avg_KFold_R2 >= {min_kfold_r2}.")
    df["KFold_Dev"] = 1.0 - df.Avg_KFold_R2
    df["Total_Dev"] = (
        (df.FEM_ANN_Max - 1).abs() + (df.FEM_ANN_Min - 1).abs() + (df.FEM_ANN_Mean - 1).abs()
    )
    df = df.sort_values(by=["FEM_ANN_CoV", "KFold_Dev", "Total_Dev"])
    best = df.iloc[0]
    print(f"[STEP 4] Selected: {best.Folder} "
          f"(Avg K-fold R2={best.Avg_KFold_R2:.4f}, FEM/ANN CoV%={best.FEM_ANN_CoV:.3f})")
    return best


# ==========================================================================
# STEP 5 -- assess the selected model against a separate FE benchmark
# ==========================================================================
def assess_against_benchmark(best_row, scaler, df_benchmark):
    model = load_model(best_row.ModelPath)
    X_bench = df_benchmark[FEATURE_COLUMNS].values
    y_fem = df_benchmark[TARGET_COLUMN].values
    y_ann = model.predict(scaler.transform(X_bench), verbose=0).flatten()

    ratio = y_fem / y_ann
    overall_stats = dict(
        Max=ratio.max(), Min=ratio.min(), Mean=ratio.mean(),
        StDev=ratio.std(), CoV_pct=100 * ratio.std() / ratio.mean(),
        R2=r2_score(y_fem, y_ann),
    )

    quantile_rows = []
    for p in TAIL_LEVELS_PCT:
        quantile_rows.append(dict(
            p_pct=p, Q_FEM=np.percentile(y_fem, p), Q_ANN=np.percentile(y_ann, p),
        ))

    tail_rows = []
    for p in TAIL_LEVELS_PCT:
        mask = y_fem <= np.percentile(y_fem, p)
        r = y_fem[mask] / y_ann[mask]
        tail_rows.append(dict(
            p_pct=p, n_points=int(mask.sum()), ratio_mean=r.mean(),
            ratio_cov_pct=100 * r.std() / r.mean() if r.mean() else np.nan,
            ratio_max=r.max(), ratio_min=r.min(),
        ))

    print(f"[STEP 5] Benchmark assessment: R2={overall_stats['R2']:.4f}, "
          f"FEM/ANN CoV%={overall_stats['CoV_pct']:.3f}")
    return overall_stats, pd.DataFrame(quantile_rows), pd.DataFrame(tail_rows), y_fem, y_ann


# ==========================================================================
# STEP 6 -- generate N new MCS input combinations and predict resistance
# ==========================================================================
def run_mcs_predictions(best_row, scaler, n_mcs=N_MCS_SAMPLES):
    model = load_model(best_row.ModelPath)
    design = lhs(len(FEATURE_COLUMNS), samples=n_mcs)
    cols = {}
    for j, name in enumerate(FEATURE_COLUMNS):
        mean = BIAS_FACTORS[name] * NOMINAL_VALUES[name]
        cols[name] = _lhs_sample_variable(mean, COV[name], DIST_TYPE[name], n_mcs, lhs_points=design[:, j])
    X_new = pd.DataFrame(cols)[FEATURE_COLUMNS].values
    resistance = model.predict(scaler.transform(X_new), verbose=0).flatten()
    print(f"[STEP 6] {n_mcs} MCS resistance realisations generated.")
    return resistance


# ==========================================================================
# STEP 7 -- failure probability, reliability index, partial factors, CIs
# ==========================================================================
def empirical_quantile_ci(data, p, conf=CONFIDENCE_LEVEL):
    n = len(data)
    sorted_d = np.sort(data)
    lo_rank = int(np.clip(binom.ppf((1 - conf) / 2, n, p), 1, n))
    hi_rank = int(np.clip(binom.ppf(1 - (1 - conf) / 2, n, p), 1, n))
    return sorted_d[lo_rank - 1], sorted_d[hi_rank - 1]


def reliability_assessment(resistance, beta_target=BETA_TARGET):
    nominal_resistance = nominal_axial_resistance(**NOMINAL_VALUES)

    R = np.asarray(resistance, dtype=float)
    mean, std = R.mean(), R.std(ddof=1)
    target_pf = norm.cdf(-beta_target)

    n_fail = int((R < nominal_resistance).sum())
    pf_count = n_fail / len(R)
    beta_norm = (mean - nominal_resistance) / std
    pf_norm = norm.cdf(-beta_norm)

    # count-based reliability index: the inverse-standard-normal transform
    # of the observed (counting) failure probability pf_count above. This
    # is reported alongside beta_norm in the manuscript and is NOT derived
    # from the moment/normal-fit route -- it is the direct count analogue
    # of beta_norm. At finite N it is only defined for 0 < pf_count < 1;
    # pf_count == 0 (no failures observed in this sample) formally gives
    # beta_count = +inf, which simply reflects that N was not large enough
    # to observe a failure at this reliability level.
    if 0 < pf_count < 1:
        beta_count = -norm.ppf(pf_count)
    else:
        beta_count = np.inf if pf_count == 0 else -np.inf

    Rd_req_momentA = mean - beta_target * std
    gammaM_A = nominal_resistance / Rd_req_momentA

    R_target_quantile = np.percentile(R, target_pf * 100, method="linear")
    gammaM_B = nominal_resistance / R_target_quantile
    q_lo, q_hi = empirical_quantile_ci(R, target_pf)

    results = dict(
        Nominal_resistance=nominal_resistance,
        N=len(R), Mean=mean, StDev=std,
        Fails=n_fail, Pf_count_pct=pf_count * 100, beta_count=beta_count,
        beta_norm=beta_norm, Pf_norm_pct=pf_norm * 100,
        beta_target=beta_target, target_Pf_pct=target_pf * 100,
        gammaM_RouteA_moment=gammaM_A,
        R_at_target_Pf_empirical=R_target_quantile,
        gammaM_RouteB_empirical=gammaM_B,
        empirical_quantile_CI_lo=q_lo, empirical_quantile_CI_hi=q_hi,
        gammaM_RouteB_CI_lo=nominal_resistance / q_hi,
        gammaM_RouteB_CI_hi=nominal_resistance / q_lo,
    )
    print(f"[STEP 7] Nominal resistance={nominal_resistance:.2f}, beta_count={beta_count:.3f}, "
          f"beta_norm={beta_norm:.3f}, gammaM (Route A)={gammaM_A:.4f}, "
          f"gammaM (Route B)={gammaM_B:.4f}")
    return results


# ==========================================================================
# MAIN -- run the full pipeline end to end
# ==========================================================================
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if not os.path.exists(TRAIN_DATA_FILE):
        print(f"\n[STEP 2] No training database found at '{TRAIN_DATA_FILE}'.")
        print("Generating an LHS sampling plan instead -- run your FE model on it, "
              "fill in the target column, and re-run this script.")
        generate_fe_sampling_plan(n=250, out_path=TRAIN_DATA_FILE.replace(".xlsx", "_PLAN.xlsx"))
        return

    df_train = pd.read_excel(TRAIN_DATA_FILE)

    summary, scaler = run_hyperparameter_grid(df_train, RESULTS_DIR)
    best = select_best_model(summary)

    if os.path.exists(BENCHMARK_DATA_FILE):
        df_bench = pd.read_excel(BENCHMARK_DATA_FILE)
        overall_stats, quantiles, tail_stats, y_fem, y_ann = assess_against_benchmark(
            best, scaler, df_bench)
        with pd.ExcelWriter(os.path.join(RESULTS_DIR, "Benchmark_Assessment.xlsx")) as xw:
            pd.DataFrame([overall_stats]).to_excel(xw, sheet_name="Overall", index=False)
            quantiles.to_excel(xw, sheet_name="Marginal_quantiles", index=False)
            tail_stats.to_excel(xw, sheet_name="Tail_conditional_ratio", index=False)
    else:
        print(f"[STEP 5] No separate benchmark file found at '{BENCHMARK_DATA_FILE}' -- skipping.")

    resistance = run_mcs_predictions(best, scaler)
    reliability = reliability_assessment(resistance)
    pd.DataFrame([reliability]).to_excel(
        os.path.join(RESULTS_DIR, "Reliability_Assessment.xlsx"), index=False)

    print(f"\nDone. Results written to '{RESULTS_DIR}/'.")


if __name__ == "__main__":
    main()
