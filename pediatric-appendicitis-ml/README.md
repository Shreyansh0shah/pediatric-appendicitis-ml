# Interpretable and Intervenable ML Models for Pediatric Appendicitis

A student machine learning project exploring pediatric appendicitis classification with K-Nearest Neighbors (KNN) and a Multi-Layer Perceptron (MLP). The supplied PDF is a printout of notebook screenshots; the Python source here is a careful reconstruction, not the original notebook file.

<img width="1832" height="982" alt="image" src="https://github.com/user-attachments/assets/1df8f6d3-4baa-4e26-8a2c-5aed90fe1d90" />


<img width="1700" height="856" alt="image" src="https://github.com/user-attachments/assets/e4e285f6-7f6a-43ca-b11b-9926c2fb0eb3" />

<img width="1782" height="1010" alt="image" src="https://github.com/user-attachments/assets/58d6cc7c-908e-4921-ad24-731fcc4ed00d" />


## What the PDF shows

- A CSV named `APPENDICITIS.csv` and target column `DiagnosisByCriteria`.
- A 20% train/test split and categorical encoding.
- KNN and MLP classifiers, with accuracy, macro precision, macro recall, F1, a classification report, and a confusion matrix.
- Saved models and a separate `test1.csv` prediction step.

The PDF does not show actual metric outputs, so this repository reports no project performance claims. It does not demonstrate SHAP explanations, clinician intervention, or clinical validation. The title is retained from the supplied notebook; those capabilities should not be inferred from it.

## Files

```text
README.md
requirements.txt
.gitignore
src/train.py
```

## Setup

Use Python 3.10 or newer. In a terminal opened in this repository:

```bash
python -m venv .venv
```

Activate the environment, then install packages:

- macOS/Linux: `source .venv/bin/activate`
- Windows: `.venv\Scripts\activate`

```bash
python -m pip install -r requirements.txt
```

## Data

Place your authorized training CSV in the project folder, named `D-7 APPENDICITIS.csv`, or provide another path with `--data`. It must contain the target column `DiagnosisByCriteria`. The original data files were not included with the PDF, so this project cannot be run until you supply the appropriate CSV. Do not publish patient-level or otherwise restricted data in a public repository. Confirm the dataset's source, license, and sharing permission before distributing it.

## Train and evaluate

```bash
python src/train.py --data "D-7 APPENDICITIS.csv"
```

The script creates `models/knn.joblib` and `models/mlp.joblib`, prints holdout metrics, and uses the first model-independent 20% split shown in the notebook. A fixed random seed and stratification are added where possible to make the reconstruction repeatable. Preprocessing is fitted with each model so categories and missing values are handled consistently without fitting encoders separately on the holdout set.

To train and score a separate feature CSV with the MLP model:

```bash
python src/train.py --data "D-7 APPENDICITIS.csv" --predict test1.csv --predictions-out predictions.csv
```

The prediction file should have the same predictor columns as the training file; if it contains the target column, that column is ignored for prediction. The generated CSV includes a `predict` column containing the model's original class labels.

## Reconstruction notes and uncertainties

The PDF screenshots crop code at the right edge and show incomplete notebook context. In particular:

- KNN visibly specifies `n_neighbors=4` and `algorithm='ball_tree'`; the remaining constructor arguments are cut off. The reconstruction uses the library defaults for other KNN options.
- The MLP visibly specifies one hidden layer of 100 units and `max_iter=300`; the value after `random_state=` is cropped. A seed of 42 is used here for repeatability.
- The split screenshot crops the `random_state` argument; this version sets it to 42.
- The notebook selects `dataset.iloc[:, 0:38]` as features. This reconstruction instead uses every column except the named target, which avoids silently omitting predictors if the CSV schema differs.
- The PDF appears to fit label encoders separately to the training and prediction tables, which can assign different codes to the same category. The reconstruction uses one fitted preprocessing pipeline per model for consistent encoding.
- A mapping screenshot associates `0` with “Appendicitis detected” and `1` with “no_Appendicitis detected”, but the visible code also applies `LabelEncoder`; class ordering and label meaning should be confirmed against the actual CSV before interpreting predictions. This version preserves target labels as read and does not impose that mapping.
- The PDF's standalone prediction snippet refers to `test1` in a line where only `test` is visibly defined. This version uses the loaded prediction table directly.

Review these assumptions against the actual CSV and original notebook if it becomes available. This is an educational reconstruction and is not intended for diagnosis or treatment decisions.

## GitHub

Repository: [Shreyansh0shah/pediatric-appendicitis-ml](https://github.com/Shreyansh0shah/pediatric-appendicitis-ml)
