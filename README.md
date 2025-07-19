# MLOps Artifact Pipeline

This repo implements a full MLOps CI/CD pipeline for digit classification using Logistic Regression and GitHub Actions. It automates testing, training, and inference, while promoting reproducibility, modularity, and artifact tracking.

---

## Project Structure

```
mlops-artifact-pipeline/
├── config/
│   └── config.json              # Model hyperparameters
├── src/
│   ├── train.py                 # Trains model from config
│   ├── inference.py            # Loads model and makes predictions
│   └── __init__.py             # Makes src a package
├── tests/
│   └── test_train.py           # Unit tests for training
├── .github/
│   └── workflows/
│       ├── test.yml            # Pytest CI job
│       ├── train.yml           # Train model and upload artifact
│       └── inference.yml       # Runs test → train → inference chain
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

---

## Dataset and Model

- Dataset: `sklearn.datasets.load_digits` (8x8 grayscale images of digits)
- Task: Multiclass classification (digits 0–9)
- Model: `LogisticRegression` from `sklearn.linear_model`

---

## Configuration

Hyperparameters are stored in `config/config.json`:

```json
{
  "C": 1.0,
  "solver": "lbfgs",
  "max_iter": 1000
}
```

---

## Usage

### Local Setup

```bash
git clone https://github.com/imrram/mlops-artifact-pipeline.git
cd mlops-artifact-pipeline

python -m venv venv
source venv/bin/activate  # For Windows: venv\Scripts\activate

pip install -r requirements.txt
```

---

### Train Model Locally

```bash
python src/train.py
```

This will create `model_train.pkl` in the root directory.

---

### Run Tests Locally

```bash
export PYTHONPATH=.
pytest
```

If you're using Windows, use `set PYTHONPATH=.` instead.

---

### Run Inference Locally

```bash
python src/inference.py
```

Prints accuracy and F1 score on the digits dataset.

---

## GitHub Actions Workflows

### test.yml

- Runs all test cases using `pytest`

### train.yml

- Trains `LogisticRegression` model
- Uploads `model_train.pkl` as an artifact

### inference.yml

Workflow with three dependent jobs:
```
test → train → inference
```

- Uses `needs:` to chain jobs
- Downloads model artifact and makes predictions

---

## Unit Tests in test_train.py

- Validates config file loading
- Ensures training function returns a valid model
- Confirms training accuracy > 85%

---

## Model Performance

| Metric     | Value (approx) |
|------------|----------------|
| Accuracy   | 0.98–0.99      |
| F1 Score   | 0.98–0.99      |

---

## Branching Strategy

```
main → classification → test → inference
```

Do not merge back to `main`. Each branch builds on the previous one.

---

## Screenshots to Include in Report

1. Git command-line operations (clone, branches)
2. Workflow runs and success output
3. pytest test run locally
4. GitHub artifact generated

---

## Notes

- Do not hardcode hyperparameters. Use the config file.
- `src` folder must contain `__init__.py` for testing to work.
- Do not delete any branch.
- `model_train.pkl` must be generated and uploaded as an artifact.
- All workflows must pass.

---

## Author
- Name: Ramashankar Mishra