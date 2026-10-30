# ANN-MCS-Steel-Reliability

Python implementation of the ANN-based Monte Carlo simulation (ANN-MCS)
framework for the stochastic characterisation and reliability assessment
of steel resistance models, as described in:

> Ljubinković, F., de Carvalho, A. S., Conde, J., Freire, F., Martins, J. P.,
> and Simões da Silva, L. (2026). *ANN-based Monte Carlo simulation framework
> for reliability assessment of steel structures*. Manuscript submitted to
> *Structures*; under revision. A publication DOI has not yet been assigned.

**Corresponding author and repository contact:** Filip Ljubinković,
University of Coimbra — [filj@uc.pt](mailto:filj@uc.pt).

This repository accompanies the paper as supplementary material for
reproducibility, and implements the framework for its two applications:
a Column-Web-in-Compression (CWC) connection component, and a welded
I-girder cross-section under axial compression.

## What's here

```
ANN-MCS-Steel-Reliability/
├── README.md
├── CITATION.cff               machine-readable citation information
├── references.bib             manuscript and related published study
├── LICENSE                    CC BY-NC 4.0 legal terms
├── requirements.txt
├── pseudocode.md              step-by-step outline of the framework
├── CWC_ANN_MCS.py             full pipeline for the CWC application
├── I_girder_ANN_MCS.py        full pipeline for the I-girder application
└── example_data/              expected input file format (see its README)
```

The two scripts are kept separate rather than merged into one generic
tool: the two underlying FE problems differ in input dimensionality, FE
formulation and resistance mechanism, so a single configurable script
would obscure more than it would share. Both follow the same structure,
however, with each step labelled to match `pseudocode.md`:

1. Define the stochastic input variables and their distributions.
2. Generate `n` input combinations via LHS for the FE analysis (external
   to this code -- see below).
3. For each hyperparameter configuration: 5-fold cross-validation, a
   final train/test model, and the performance indicators, including
   test-subset `R2_test`.
4. Select the ANN using the prescribed selection criteria.
5. Assess the selected model against a separate FE benchmark set,
   including lower-tail quantiles.
6. Generate `N` new stochastic input combinations via LHS and predict
   their resistance with the selected ANN (the actual Monte Carlo step).
7. Compute failure probabilities, reliability indices, partial factors
   (`gammaM`, via both a normal-distribution approximation and a
   direct empirical-quantile route), and confidence intervals.

See `pseudocode.md` for the full outline and the reasoning behind steps
4, 5 and 7 specifically.

## What this code does *not* do

The finite element (FE) analysis itself is external to this repository --
these scripts generate the LHS sampling plan for step 2 and expect the
corresponding FE results back as an Excel file (format described in
`example_data/README.md`). No FE solver, mesh, or model is included.

No trained models or datasets are included. Each script trains its own
models from a training-data Excel file you provide. These are configurable
implementations of the workflow, rather than a packaged reproduction of
every manuscript table and figure. The supplied geometry, material
statistics, resistance threshold, and reduced 24-configuration grids are
illustrative defaults: check them against your FE database before use.
In particular, the CWC default grid is not the paper's full
144-configuration search, and each script processes one development
dataset per run.

Preprocessing follows the disclosed development procedure: input scaling
is fitted to the complete development dataset, and the resistance-ratio
statistics used for selection include that dataset. The train/test split
is separate from five-fold cross-validation. A supplied benchmark file
is used only after model selection.

## Installation

```bash
pip install -r requirements.txt
```

Use Python 3.10–3.12 with the pinned dependencies in `requirements.txt`,
including TensorFlow/Keras, scikit-learn, pandas, scipy and pyDOE. These
pins specify the direct dependencies rather than a complete environment
lock; hardware and numerical libraries can still affect results. `RANDOM_STATE` fixes the data partitions, but the current scripts
do not fix all LHS and neural-network random generators, so retraining
need not reproduce identical numerical values.

## Usage

1. Work from the repository root and edit the `CONFIG` block at the top
   of the relevant script: input
   variable names, their distributions, the nominal/limit-state
   resistance, the data file paths, and (optionally) the hyperparameter
   grid.
2. Provide a training database (`example_data/README.md` describes the
   expected format) -- or run the script once with no training file
   present, which instead writes a 250-row LHS sampling plan to run through
   your own FE model. Fill its `F_FEM` column with FE resistances in kN,
   then save it under the configured training filename (without `_PLAN`),
   or point `TRAIN_DATA_FILE` to the completed file.
3. Run the script:

```bash
python CWC_ANN_MCS.py
# or
python I_girder_ANN_MCS.py
```

Each run writes its outputs (hyperparameter grid summary, trained models,
benchmark assessment, Monte Carlo reliability results) to a results
folder (`CWC_ANN_MCS_Results/` or `Igirder_ANN_MCS_Results/`).

`Hyperparameter_Summary.xlsx` reports `Avg_KFold_R2` (the mean fold
validation score), `R2_test` (the retained model's score on its own test
subset), and pooled development-dataset FEM/ANN ratio statistics. These
are distinct metrics. `R2_test` is reported for diagnosis; the existing
configuration-selection rule remains the CV threshold followed by
ranking on the pooled ratio CoV and the documented secondary criteria.
Because preprocessing and selection also use the development dataset,
`R2_test` is not an untouched test of the complete selection procedure.

## Reliability outputs

The reliability results report both `beta_norm` (Route A) and
`beta_count = -Phi_inverse(Pf_count)` (Route B), alongside
`Pf_norm_pct` and `Pf_count_pct`. Probability output columns ending in
`_pct` are percentages; the inverse-normal calculation uses probabilities
between zero and one.

The confidence bounds `empirical_quantile_CI_lo/hi` concern the empirical
target resistance, and `gammaM_RouteB_CI_lo/hi` concern the corresponding
Route B partial factor. They are not confidence intervals on failure
probability or either reliability index. The binomial rank method assumes
independent observations and is used as an approximate reference for LHS;
it does not quantify ANN or FE-model uncertainty.

At zero observed failures, the code exports `beta_count = +inf`; at failures
in every case it exports `-inf`. These are mathematical boundary limits,
not finite reliability estimates. Zero observed failures do not establish
zero population failure probability.

## Citation

If you use this implementation, please cite the manuscript above. Full
author information is provided in `CITATION.cff` and a BibTeX entry is
included in `references.bib`. The manuscript entry will be updated with
the final journal details and DOI after publication.

The preceding investigation of stochastic joint stiffness is:

> Ljubinković, F., de Carvalho, A., Ribeiro, F., Conde, J., and Simões da
> Silva, L. (2025). Towards Stochastic Characterization of Welded
> Beam-to-Column Steel Joints Using Artificial Neural Networks.
> *Structural Engineering International*, 35(4), 617–633.
> https://doi.org/10.1080/10168664.2025.2526010

This related publication motivates the present work; its DOI does not
identify the current manuscript or this code release.

## License

Copyright 2026 Filip Ljubinković.

This repository is licensed under the **Creative Commons
Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)** licence.
You may share and adapt the material for non-commercial purposes under
the licence terms, including attribution, a licence link, and an
indication of changes. See [LICENSE](LICENSE) for the complete legal text
and the [licence summary](https://creativecommons.org/licenses/by-nc/4.0/).

For uses outside these permissions, contact Filip Ljubinković at
[filj@uc.pt](mailto:filj@uc.pt). Third-party packages remain subject to
their respective licences.

