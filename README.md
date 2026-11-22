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
