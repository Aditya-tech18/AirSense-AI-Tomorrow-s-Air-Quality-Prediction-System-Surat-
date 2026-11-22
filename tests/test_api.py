import json
import logging
from unittest.mock import Mock

from fastapi.testclient import TestClient
import joblib
import numpy as np
import pytest

from airsense.api.main import create_app
from airsense.api.schemas import FEATURES
from airsense.logging_config import JsonFormatter


@pytest.fixture
def payload():
    return {
        "pm25_today": 42.5, "wind_kmph": 12.0, "humidity_pct": 65.0,
        "temp_c": 29.0, "traffic_index": 55.0, "stubble_fires": 30,
        "day_type": "weekday", "station_zone": "Central",
    }


@pytest.fixture
def model():
    instance = Mock(spec=["predict"])
    instance.predict.return_value = np.array([48.25])
    return instance


def test_health_and_prediction_with_ordered_clean_features(model, payload, rules):
    with TestClient(create_app(model=model)) as client:
        assert client.get("/health").json() == {"status": "ok"}
        # Input JSON property order must not control the training feature order.
        response = client.post("/predict", json=dict(reversed(list(payload.items()))))
    assert response.status_code == 200
    assert response.json() == {"prediction": 48.25}
    frame = model.predict.call_args.args[0]
    assert frame.columns.tolist() == rules["features"] == list(FEATURES)
    assert frame.iloc[0].tolist() == list(payload.values())
    assert all(str(frame[column].dtype) == "float64" for column in FEATURES[:6])
    assert frame["station_zone"].iloc[0] == "Central"


@pytest.mark.parametrize("changes", [
    {"pm25_today": 7.5}, {"stubble_fires": 750},
    {"pm25_today": 7.5, "stubble_fires": 750},
])
def test_exact_boundaries_are_accepted(model, payload, changes):
    with TestClient(create_app(model=model)) as client:
        assert client.post("/predict", json=payload | changes).status_code == 200
    model.predict.assert_called_once()


@pytest.mark.parametrize("changes", [
    {"pm25_today": 7.499999}, {"stubble_fires": 751},
    {"stubble_fires": -1}, {"stubble_fires": 30.5},
    {"stubble_fires": True}, {"wind_kmph": "12"},
    {"day_type": "wee."}, {"station_zone": "central"},
    {"temp_c": None}, {"extra": "private detail"},
])
def test_invalid_inputs_never_reach_model(model, payload, changes):
    with TestClient(create_app(model=model)) as client:
        response = client.post("/predict", json=payload | changes)
    assert response.status_code == 422
    assert all("input" not in error for error in response.json()["detail"])
    assert "private detail" not in response.text
    model.predict.assert_not_called()


@pytest.mark.parametrize("field", FEATURES)
def test_every_field_is_required(model, payload, field):
    del payload[field]
    with TestClient(create_app(model=model)) as client:
        assert client.post("/predict", json=payload).status_code == 422
    model.predict.assert_not_called()


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity"])
def test_non_finite_inputs_are_rejected_without_serialization_error(model, payload, value):
    body = json.dumps(payload).replace('"temp_c": 29.0', '"temp_c": ' + value)
    with TestClient(create_app(model=model)) as client:
        assert client.post("/predict", content=body, headers={
            "content-type": "application/json",
        }).status_code == 422
    model.predict.assert_not_called()


def test_one_allowlisted_json_line_per_success(model, payload):
    from io import StringIO

    stream = StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    logger = logging.getLogger("airsense.serving")
    logger.addHandler(handler)
    try:
        with TestClient(create_app(model=model)) as client:
            client.post("/predict", json=payload)
            client.post("/predict", json=payload | {"stubble_fires": 751})
    finally:
        logger.removeHandler(handler)
    lines = stream.getvalue().splitlines()
    assert len(lines) == 1
    event = json.loads(lines[0])
    assert event["event"] == "prediction" and event["status"] == "ok"
    assert event["timestamp"].endswith("+00:00")
    assert event["latency_ms"] >= 0
    assert set(event) == {"event", "status", "timestamp", "level", "latency_ms"}
    model.predict.assert_called_once()


def test_missing_and_corrupt_model_are_unavailable(tmp_path, payload):
    path = tmp_path / "absent.joblib"
    for corrupt in (False, True):
        if corrupt:
            path.write_text("not a model", encoding="utf-8")
        with TestClient(create_app(model_path=path)) as client:
            assert client.get("/health").status_code == 503
            assert client.post("/predict", json=payload).status_code == 503


@pytest.mark.parametrize("kind", ["mapping", "unfitted"])
def test_non_predictor_and_unfitted_artifacts_are_unavailable(kind, tmp_path):
    from sklearn.dummy import DummyRegressor

    artifact = {} if kind == "mapping" else DummyRegressor()
    path = tmp_path / "invalid.joblib"
    joblib.dump(artifact, path)
    with TestClient(create_app(model_path=path)) as client:
        assert client.get("/health").status_code == 503


def test_load_once_and_environment_path(monkeypatch, model, payload, tmp_path):
    import airsense.api.main as service

    path = tmp_path / "champion.joblib"
    monkeypatch.setenv("AIRSENSE_MODEL_PATH", str(path))
    loader = Mock(return_value=model)
    monkeypatch.setattr(service, "load_model", loader)
    with TestClient(create_app()) as client:
        client.post("/predict", json=payload)
        client.post("/predict", json=payload)
    loader.assert_called_once_with(path)
    assert model.predict.call_count == 2


@pytest.mark.parametrize("result", [[float("nan")], [float("inf")], [], [1, 2], [[1]], ["48"]])
def test_invalid_model_outputs_fail_cleanly(model, payload, result):
    model.predict.return_value = result
    with TestClient(create_app(model=model)) as client:
        response = client.post("/predict", json=payload)
    assert response.status_code == 500
    assert response.json() == {"detail": "Prediction failed"}


def test_model_exception_is_not_exposed(model, payload):
    model.predict.side_effect = RuntimeError("secret internal detail")
    with TestClient(create_app(model=model)) as client:
        response = client.post("/predict", json=payload)
    assert response.status_code == 500
    assert "secret" not in response.text


def test_wrong_feature_order_artifact_is_unavailable(tmp_path, payload, rules):
    import pandas as pd
    from airsense.models.train import build_candidates

    frame = pd.DataFrame([payload], columns=list(reversed(FEATURES)))
    settings = {"random_seed": 252, "random_forest": {}, "hist_gradient_boosting": {}}
    reversed_rules = rules | {"features": list(reversed(FEATURES))}
    artifact = build_candidates(reversed_rules, settings)["median_baseline"].fit(frame, [19.0])
    path = tmp_path / "mismatched.joblib"
    joblib.dump(artifact, path)
    with TestClient(create_app(model_path=path)) as client:
        assert client.get("/health").status_code == 503
        assert client.post("/predict", json=payload).status_code == 503


def test_serialized_test_pipeline_loads_without_production_artifact(tmp_path, payload, rules):
    import pandas as pd
    from airsense.models.train import build_candidates

    frame = pd.DataFrame([payload], columns=list(FEATURES))
    settings = {"random_seed": 252, "random_forest": {}, "hist_gradient_boosting": {}}
    artifact = build_candidates(rules, settings)["median_baseline"].fit(frame, [19.0])
    path = tmp_path / "test-only.joblib"
    joblib.dump(artifact, path)
    with TestClient(create_app(model_path=path)) as client:
        assert client.get("/health").status_code == 200
        assert client.post("/predict", json=payload).json() == {"prediction": 19.0}
