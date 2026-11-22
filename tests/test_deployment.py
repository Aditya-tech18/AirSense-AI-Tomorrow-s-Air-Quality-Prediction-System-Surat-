from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_mp252_deployment_contract():
    deployment = yaml.safe_load((ROOT / "k8s/deployment.yaml").read_text())
    service = yaml.safe_load((ROOT / "k8s/service.yaml").read_text())
    assert deployment["metadata"]["name"] == "mp252-api"
    assert deployment["spec"]["replicas"] == 4
    labels = deployment["spec"]["template"]["metadata"]["labels"]
    assert deployment["spec"]["selector"]["matchLabels"] == labels
    assert service["metadata"]["name"] == "mp252-svc"
    assert service["spec"]["selector"] == labels
    container = deployment["spec"]["template"]["spec"]["containers"][0]
    assert container["image"] == "mp252-api:1.0"
    assert container["ports"][0]["containerPort"] == 8000
    assert service["spec"]["ports"][0] == {"port": 80, "targetPort": 8000}
    for probe in ("readinessProbe", "livenessProbe"):
        assert container[probe]["httpGet"] == {"path": "/health", "port": 8000}
    assert container["livenessProbe"]["initialDelaySeconds"] > container["readinessProbe"]["initialDelaySeconds"]
