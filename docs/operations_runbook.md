# AirSense AI: Stage 1–3 handoff

Group G180, MP252. Serving/Ops contribution prepared for Tanisha Gupta,
S24CSEU0578, with AI assistance. This document records tested behavior;
it does not claim independent student authorship or a completed Stage 4.

## Integration

The working branch incorporates main `30b34e54accc6180c6528160bacea06727330e43`
and Prejval's model/handoff commit `7922220` by ancestry, preserving Aditya's data code,
model code, dependency pins, pytest settings, findings and fitted artifact.
No teammate branch or main is a push target. New work belongs on
`feature/tanisha` only. Stage 1/2 authorship stays with the existing commits.

The champion is a fitted sklearn pipeline at `models/model.joblib`:
registry `mp252-pm25-regressor`, alias `champion`, version 1. API inputs use
this exact feature order:

`pm25_today, wind_kmph, humidity_pct, temp_c, traffic_index, stubble_fires, day_type, station_zone`.

The first six columns become float64, matching `prepare_split`. Day type
and station zone retain their canonical strings. The exported pipeline owns
imputation and categorical encoding. API requests never receive training-only
unit corrections, sentinel replacement, deduplication or target repairs.
Only load trusted team artifacts: joblib deserialization can execute code.

## Reproduce

Use Python 3.12 from the repository root:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
OMP_NUM_THREADS=1 python -m pytest -q
PYTHONPATH=src OMP_NUM_THREADS=1 uvicorn app.main:app --host 0.0.0.0 --port 8000
```

In another terminal:

```bash
curl --fail http://localhost:8000/health
curl --fail http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"pm25_today":42.5,"wind_kmph":12,"humidity_pct":65,"temp_c":29,"traffic_index":55,"stubble_fires":30,"day_type":"weekday","station_zone":"Central"}'
```

The response is `{"prediction": <finite number>}`. Configure an alternative
trusted artifact with `AIRSENSE_MODEL_PATH=/absolute/path/model.joblib`.
Load happens in application lifespan, once per process. A missing, corrupt,
unfitted or incompatible sklearn artifact leaves `/health` and valid
`/predict` requests at 503 with a diagnostic JSON startup event. Installing
or replacing an artifact requires restarting the application process.
Invalid requests return 422 without invoking the model. Failed or non-finite
model output returns 500 with a generic response, not internal exception text.

## Container and Kubernetes

```bash
docker build -t mp252-api:1.0 .
docker run --rm -p 8000:8000 mp252-api:1.0
# Optional: replace the packaged artifact with an approved read-only mount.
docker run --rm -p 8000:8000 \
  -v "$PWD/models/model.joblib:/opt/airsense/models/model.joblib:ro" mp252-api:1.0
kubectl apply --dry-run=client -f k8s/
# Only when targeting your intended local cluster:
kubectl apply -f k8s/
```

Use one container command at a time. For a local Kubernetes cluster, load
`mp252-api:1.0` into the cluster's container runtime before applying the YAML.
The image uses Python 3.12-slim and a non-root account. Requirements install
before source files, so editing code can reuse the dependency layer. The
Docker context excludes data, local environments, caches, credentials and
MLflow storage. The portable champion stays in `models/` and is copied.

The Deployment is `mp252-api`, has four replicas and exposes 8000. Service
`mp252-svc` maps 80 to 8000 with matching Pod selectors. Both probes call
`/health`; liveness waits 30 seconds, readiness waits 5. A permanently missing
model fails both probes until a valid image/artifact is deployed.

## Verified locally on 22 November 2026

- All 50 tests passed, including data/model tests and TestClient fixtures.
  API tests create test doubles or small temporary artifacts and need no CSVs
  or checked-in production model. Deployment tests check the MP252 YAML contract.
- Real checked-in champion: `/health` returned 200 and nine predictions
  (all canonical day-type/station-zone combinations) were finite and returned 200.
- Dependency resolver: `pip check` found no broken requirements.
- Original training SHA-256:
  `bd7f9b90d0bf0d8e8570e4534b3a40505d353a34497ef5801b2498a9851c6c05`.
  All three stored MLflow runs record this fingerprint. Their validation MAEs
  are 44.75424528301886 (baseline), 15.69284167757004 (random forest),
  and 11.292679766536876 (histogram gradient boosting champion).
- The seven data audit results agree with the existing findings. DVC cache
  was populated from the unchanged original ZIP inputs; `dvc status` reports
  data and pipelines up to date. Raw train/later CSVs remain outside Git.
- Docker and kubectl executables are unavailable in the execution environment.
  No image build, container smoke test or cluster application is claimed.
  Kubernetes validation here is local YAML parsing and contract tests.
- GitHub CI needs a successful push; inspect the Actions run for the final
  branch head. A local test pass is not a GitHub CI pass.

## Hinglish viva notes

- Request schema gatekeeper hai: missing/extra fields aur out-of-range values
  par 422 milta hai; prediction function tak request nahi jaati.
- Champion startup par ek baar load hota hai. Har request par disk se dobara
  load nahi hota. Path env variable se change kar sakte hain.
- Logging formatter event ko JSON line banata hai. Hum status, UTC timestamp
  aur latency record karte hain; raw request ya personal details nahi.
- Docker mein requirements pehle copy karte hain. Sirf code badalne par pip
  install wali cached layer dobara banane ki zaroorat nahi padti.
- Readiness decide karti hai Pod ko traffic milega ya nahi. Liveness decide
  karti hai container restart karna chahiye ya nahi; uska initial wait zyada hai.
- CI har push par dependencies install karke pytest chalata hai. Green run
  evidence hai ki us remote commit ke tests pass hue.

Stage 4 PSI, fairness MAE, retraining/promotion and holdout predictions have
not been performed as part of this serving contribution.
