"""Unit and regression tests for R23 Security, Token Rotation, and Deployment Contracts."""

from __future__ import annotations

import time
from pathlib import Path
import pytest

from urbangraph.data import DataError
from urbangraph.security.token_rotation import TokenRotationEngine


def test_token_rotation_and_grace_period() -> None:
    # 5-second grace period for fast testing
    engine = TokenRotationEngine(rotation_ttl_seconds=10.0, grace_period_seconds=2.0)

    token_v1 = engine.get_primary_token("worker")
    assert engine.validate_token("worker", token_v1) is True

    # Rotate token
    token_v2 = engine.rotate_token("worker")
    assert token_v2 != token_v1
    assert engine.get_primary_token("worker") == token_v2

    # Both new token and old token (during grace period) are valid
    assert engine.validate_token("worker", token_v2) is True
    assert engine.validate_token("worker", token_v1) is True

    # Wait for grace period to expire
    time.sleep(2.2)
    # Old token expired
    assert engine.validate_token("worker", token_v1) is False
    # New token still valid
    assert engine.validate_token("worker", token_v2) is True


def test_role_isolation_and_negative_authorization() -> None:
    engine = TokenRotationEngine()

    reader_token = engine.get_primary_token("reader")
    admin_token = engine.get_primary_token("admin")

    # Reader token cannot authorize worker or admin
    assert engine.validate_token("reader", reader_token) is True
    assert engine.validate_token("worker", reader_token) is False
    assert engine.validate_token("admin", reader_token) is False

    # Empty / forged tokens fail
    assert engine.validate_token("admin", "") is False
    assert engine.validate_token("admin", "forged-token-abc123") is False


def test_token_audit_log_redaction() -> None:
    engine = TokenRotationEngine()
    engine.rotate_token("admin")

    audit = engine.get_audit_log()
    assert len(audit) > 0
    for record in audit:
        # Full secret must not be logged
        assert "token" not in record
        assert "new_token_fingerprint" in record
        assert len(record["new_token_fingerprint"]) < 15


from urbangraph.security.k8s_validator import (
    parse_simple_yaml,
    validate_helm_chart_package,
    validate_k8s_deployment,
    validate_k8s_service_and_netpol,
)


def test_kubernetes_restricted_pod_security_standards() -> None:
    manifest_path = Path("infra/k8s/deployment.yaml")
    assert manifest_path.is_file()

    content = manifest_path.read_text(encoding="utf-8")
    docs = parse_simple_yaml(content)
    assert len(docs) == 1

    deployment = docs[0]
    result = validate_k8s_deployment(deployment)
    assert result["status"] == "VALID"
    assert result["pod_security_standard"] == "restricted"
    assert result["containers_verified"] >= 1


def test_kubernetes_service_and_netpol_integration() -> None:
    deploy_doc = parse_simple_yaml(Path("infra/k8s/deployment.yaml").read_text(encoding="utf-8"))[0]
    svc_docs = parse_simple_yaml(Path("infra/k8s/service.yaml").read_text(encoding="utf-8"))
    assert len(svc_docs) == 2

    service_doc = next(d for d in svc_docs if d.get("kind") == "Service")
    netpol_doc = next(d for d in svc_docs if d.get("kind") == "NetworkPolicy")

    result = validate_k8s_service_and_netpol(service_doc, netpol_doc, deploy_doc)
    assert result["status"] == "VALID"
    assert result["service_type"] == "ClusterIP"


def test_helm_chart_package_validation() -> None:
    helm_dir = Path("infra/helm/energraph-ai")
    result = validate_helm_chart_package(helm_dir)
    assert result["status"] == "VALID"
    assert result["chart_name"] == "energraph-ai"
    assert "deployment.yaml" in result["templates"]
    assert "service.yaml" in result["templates"]


def test_kubernetes_negative_security_rejections() -> None:
    deploy_doc = parse_simple_yaml(Path("infra/k8s/deployment.yaml").read_text(encoding="utf-8"))[0]

    # Negative 1: runAsNonRoot disabled
    insecure = dict(deploy_doc)
    insecure["spec"] = dict(deploy_doc["spec"])
    insecure["spec"]["template"] = dict(deploy_doc["spec"]["template"])
    insecure["spec"]["template"]["spec"] = dict(deploy_doc["spec"]["template"]["spec"])
    insecure["spec"]["template"]["spec"]["securityContext"] = {"runAsNonRoot": False}
    with pytest.raises(DataError, match="runAsNonRoot: true"):
        validate_k8s_deployment(insecure)

    import copy

    # Negative 2: writable root filesystem
    insecure_c = copy.deepcopy(deploy_doc)
    insecure_c["spec"]["template"]["spec"]["containers"][0]["securityContext"]["readOnlyRootFilesystem"] = False
    with pytest.raises(DataError, match="readOnlyRootFilesystem: true"):
        validate_k8s_deployment(insecure_c)

    # Negative 3: allowPrivilegeEscalation enabled
    insecure_priv = copy.deepcopy(deploy_doc)
    insecure_priv["spec"]["template"]["spec"]["containers"][0]["securityContext"]["allowPrivilegeEscalation"] = True
    with pytest.raises(DataError, match="allowPrivilegeEscalation: false"):
        validate_k8s_deployment(insecure_priv)

    # Negative 4: missing drop ALL capabilities
    insecure_caps = copy.deepcopy(deploy_doc)
    insecure_caps["spec"]["template"]["spec"]["containers"][0]["securityContext"]["capabilities"]["drop"] = ["NET_RAW"]
    with pytest.raises(DataError, match="drop ALL capabilities"):
        validate_k8s_deployment(insecure_caps)

    # Negative 5: missing health probes
    insecure_probes = copy.deepcopy(deploy_doc)
    insecure_probes["spec"]["template"]["spec"]["containers"][0]["livenessProbe"] = {}
    with pytest.raises(DataError, match="livenessProbe"):
        validate_k8s_deployment(insecure_probes)

    # Negative 6: schema validation error when required field is missing
    insecure_schema = copy.deepcopy(deploy_doc)
    del insecure_schema["spec"]["template"]["spec"]["containers"][0]["image"]
    with pytest.raises(DataError, match="Kubernetes schema validation error"):
        validate_k8s_deployment(insecure_schema)



