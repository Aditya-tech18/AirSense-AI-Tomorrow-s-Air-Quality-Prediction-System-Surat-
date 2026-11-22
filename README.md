# AirSense AI

Bennett University SCSE3040 MLOps mini project, group G180, problem MP252.
The goal is to predict next-day PM2.5 in Surat, evaluated with MAE.

| Member | Enrollment |
| --- | --- |
| Aditya Chaubey | S24CSEU2150 |
| Prejval Yadav | S24CSEU1585 |
| Tanisha Gupta | S24CSEU0578 |

## Data pipeline

Use Python 3.12 and install requirements.txt in a virtual environment.
Place the original faculty train.csv and later.csv under data/raw. Their
DVC pointers identify the original bytes. This repository has no configured
DVC remote. When the original inputs are supplied, run dvc status to check
that they match their pointers.

Run the audit from the repository root.

```bash
PYTHONPATH=src python scripts/audit_data.py
PYTHONPATH=src python scripts/audit_data.py --write
python -m pytest
```

The write option updates only the data issues in reports/findings.yaml.
Audit and preprocessing keep the raw input files intact.

Use prepare_split to obtain ordered features, optional labels and metadata.
Fit numeric imputation and categorical encoding on the training partition
only. Training-specific repairs use verified data issue findings. For later
and holdout, input rows and IDs keep their original order. Holdout labels
are absent. Metadata includes record ID, observation date and station zone.

The data-pipeline implementation was produced with AI assistance as
requested by Aditya.

## Model training and MLflow

From the repository root, with the original training CSV in place:

```bash
OMP_NUM_THREADS=1 PYTHONPATH=src python scripts/train.py
python -m pytest
```

Training compares a median baseline, random forest and histogram gradient
boosting by MAE on held-out dates, with a one-day purge for the next-day
label. Candidate preprocessing is fitted inside the training window.
The winner is refitted on the full cleaned training split and exported to
`models/model.joblib`. The service should pass the eight feature columns
directly to the exported sklearn pipeline, which owns imputation and encoding.

The SQLite database records the runs, validation metrics, and champion alias.
MLflow model artifact files require the local artifact directory used during
training. The exported joblib serves as the portable inference artifact.
Run the training command in a new checkout to recreate its local artifacts.
This model training implementation was produced with AI assistance.

### Serving handoff

Load `ChampionPredictor` from `airsense.models.inference` once at service
startup. Pass the configured ordered feature names and the trusted
`models/model.joblib` path to `ChampionPredictor.load`. After HTTP request
validation, convert request data to a DataFrame and pass it to `predict`.
The adapter verifies the saved model's feature contract on load and returns
one finite prediction per row. It accepts shuffled feature columns and
restores training order. The API owns HTTP validation and error translation.
