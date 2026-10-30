# example_data

No datasets are included in this repository yet. This folder documents
the file format each script expects, so you can drop in your own data
(or a future released example dataset). Match the column names and paths
in the script configuration to the supplied files.

These file specifications accompany *ANN-based Monte Carlo simulation
framework for reliability assessment of steel structures* (Ljubinković et
al., 2026; manuscript under revision). Contact: Filip Ljubinković,
filj@uc.pt.

## CWC_ANN_MCS.py

- **`CWC_training_database.xlsx`** -- one row per FE run used to train the
  ANN. Columns: the names listed in `FEATURE_COLUMNS` (12 predictor
  variables), plus the target column `F_FEM` (FE-computed resistance).
- **`CWC_FE_benchmark.xlsx`** -- a separate, larger set of FE runs (not
  used in training) with the same columns, used only for the step 5
  benchmark assessment.

## I_girder_ANN_MCS.py

- **`Igirder_training_database.xlsx`** -- one row per FE run. Columns:
  `hw, bf, tw, tf, E, fy` (the six predictor variables), plus the target
  column `F_FEM` (FE-computed axial resistance).
- **`Igirder_FE_benchmark.xlsx`** -- a separate, larger FE benchmark set
  with the same columns, used only for the step 5 benchmark assessment.

If no training file is found at the configured path, each script instead
writes an LHS sampling plan (`*_PLAN.xlsx`) that can be run through your
own FE model to produce a training file in the expected format.

The generated plan has 250 rows by default. After filling in its resistance
column, save it under the configured training filename without `_PLAN`,
or update `TRAIN_DATA_FILE` to point to the completed plan. Empty target
values must be filled before training; the scripts do not run Abaqus.

Use consistent physical units: the CWC input names specify MPa and mm;
I-girder inputs use mm for dimensions and MPa for `E` and `fy`. Resistance
values in `F_FEM` and the design-resistance threshold must be in kN.
Benchmark files must retain their paired input and FE-output rows.

