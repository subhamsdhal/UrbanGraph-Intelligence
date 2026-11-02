"""Kubernetes and Helm Manifest Security and Structural Validator.

Independently validates Kubernetes resources, Pod Security Standards (Restricted),
service selectors, network policies, health probes, and Helm chart specifications
using formal Pydantic schema contracts and security policies.
Satisfies Requirement R23.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from urbangraph.data import DataError

try:
    import yaml  # type: ignore
except ImportError:
    yaml = None


# -----------------------------------------------------------------------------
# Kubernetes Formal Pydantic Schema Contracts
# -----------------------------------------------------------------------------


class K8sCapabilities(BaseModel):
    drop: list[str] = Field(default_factory=list)


class K8sContainerSecurityContext(BaseModel):
    allowPrivilegeEscalation: bool = False
    readOnlyRootFilesystem: bool = True
    capabilities: K8sCapabilities = Field(default_factory=K8sCapabilities)


class K8sSeccompProfile(BaseModel):
    type: str = "RuntimeDefault"


class K8sPodSecurityContext(BaseModel):
    runAsNonRoot: bool = True
    runAsUser: int = 65534
    seccompProfile: K8sSeccompProfile = Field(default_factory=K8sSeccompProfile)


class K8sProbe(BaseModel):
    httpGet: dict[str, Any] | None = None
    initialDelaySeconds: int | None = None
    periodSeconds: int | None = None


class K8sResourceConstraints(BaseModel):
    cpu: str | None = None
    memory: str | None = None


class K8sResourceRequirements(BaseModel):
    limits: K8sResourceConstraints
    requests: K8sResourceConstraints


class K8sContainer(BaseModel):
    name: str
    image: str
    securityContext: K8sContainerSecurityContext
    livenessProbe: K8sProbe
    readinessProbe: K8sProbe
    resources: K8sResourceRequirements


class K8sPodSpec(BaseModel):
    securityContext: K8sPodSecurityContext
    containers: list[K8sContainer]


class K8sPodTemplateSpec(BaseModel):
    metadata: dict[str, Any]
    spec: K8sPodSpec


class K8sDeploymentSpec(BaseModel):
    replicas: int = 1
    selector: dict[str, Any]
    template: K8sPodTemplateSpec


class K8sDeployment(BaseModel):
    apiVersion: Literal["apps/v1"]
    kind: Literal["Deployment"]
    metadata: dict[str, Any]
    spec: K8sDeploymentSpec


class K8sServiceSpec(BaseModel):
    type: str = "ClusterIP"
    selector: dict[str, str]
    ports: list[dict[str, Any]]


class K8sService(BaseModel):
    apiVersion: Literal["v1"]
    kind: Literal["Service"]
    metadata: dict[str, Any]
    spec: K8sServiceSpec


class K8sNetworkPolicy(BaseModel):
    apiVersion: Literal["networking.k8s.io/v1"]
    kind: Literal["NetworkPolicy"]
    metadata: dict[str, Any]
    spec: dict[str, Any]


# -----------------------------------------------------------------------------
# YAML Parser with Safe Fallback
# -----------------------------------------------------------------------------


def parse_simple_yaml(text: str) -> list[dict[str, Any]]:
    """Parse YAML documents using PyYAML if available, else structural fallback."""
    if yaml is not None:
        try:
            loaded = list(yaml.safe_load_all(text))
            return [doc for doc in loaded if isinstance(doc, dict)]
        except Exception:
            pass

    return _parse_simple_yaml_fallback(text)


def _parse_simple_yaml_fallback(text: str) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    raw_docs = text.split("\n---")

    for raw_doc in raw_docs:
        lines = [line for line in raw_doc.splitlines() if line.strip() and not line.strip().startswith("#")]
        if not lines:
            continue

        doc: dict[str, Any] = {}
        stack: list[tuple[int, dict[str, Any] | list[Any], str | None]] = [(0, doc, None)]

        i = 0
        while i < len(lines):
            line = lines[i]
            indent = len(line) - len(line.lstrip())
            stripped = line.strip()

            while len(stack) > 1 and indent < stack[-1][0]:
                stack.pop()

            curr_indent, curr_obj, _ = stack[-1]

            if stripped.startswith("- "):
                val_str = stripped[2:].strip()
                if isinstance(curr_obj, list):
                    if ":" in val_str and not val_str.endswith(":"):
                        sub_dict: dict[str, Any] = {}
                        k, v = val_str.split(":", 1)
                        sub_dict[k.strip()] = _parse_val(v.strip())
                        curr_obj.append(sub_dict)
                        stack.append((indent + 2, sub_dict, k.strip()))
                    elif val_str.endswith(":"):
                        sub_dict = {}
                        curr_obj.append(sub_dict)
                        stack.append((indent + 2, sub_dict, val_str[:-1].strip()))
                    else:
                        curr_obj.append(_parse_val(val_str))
            elif ":" in stripped:
                key, rest = stripped.split(":", 1)
                key = key.strip()
                val = rest.strip()

                if isinstance(curr_obj, dict):
                    if not val:
                        if i + 1 < len(lines) and lines[i + 1].strip().startswith("- "):
                            new_list: list[Any] = []
                            curr_obj[key] = new_list
                            stack.append((indent + 2, new_list, key))
                        else:
                            new_dict: dict[str, Any] = {}
                            curr_obj[key] = new_dict
                            stack.append((indent + 2, new_dict, key))
                    elif val == "{}":
                        curr_obj[key] = {}
                    elif val == "[]":
                        curr_obj[key] = []
                    else:
                        curr_obj[key] = _parse_val(val)
            i += 1

        if doc:
            documents.append(doc)

    return documents


def _parse_val(val: str) -> Any:
    if val.startswith('"') and val.endswith('"'):
        return val[1:-1]
    if val.startswith("'") and val.endswith("'"):
        return val[1:-1]
    if val.lower() == "true":
        return True
    if val.lower() == "false":
        return False
    if val.lower() == "null" or val == "~":
        return None
    try:
        if "." in val:
            return float(val)
        return int(val)
    except ValueError:
        return val


# -----------------------------------------------------------------------------
# Validation Engines
# -----------------------------------------------------------------------------


def validate_k8s_deployment(deployment: dict[str, Any]) -> dict[str, Any]:
    """Validate Kubernetes deployment resource against restricted security profile."""
    try:
        model = K8sDeployment.model_validate(deployment)
    except ValidationError as exc:
        raise DataError(f"Kubernetes schema validation error: {exc}") from exc

    match_labels = model.spec.selector.get("matchLabels", {})
    pod_labels = model.spec.template.metadata.get("labels", {})

    # 1. Selector reconciliation
    for k, v in match_labels.items():
        if pod_labels.get(k) != v:
            raise DataError(f"Deployment selector mismatch for label {k}: {v} != {pod_labels.get(k)}")

    pod_sec = model.spec.template.spec.securityContext

    # 2. Pod-level Restricted Security Standards
    if not pod_sec.runAsNonRoot:
        raise DataError("Pod must enforce runAsNonRoot: true")
    if pod_sec.runAsUser != 65534:
        raise DataError("Pod must run as non-root UID 65534 (nobody)")
    if pod_sec.seccompProfile.type != "RuntimeDefault":
        raise DataError("Pod must enforce seccompProfile type: RuntimeDefault")

    # 3. Container-level Security Standards
    containers = model.spec.template.spec.containers
    if not containers:
        raise DataError("Deployment has no containers")

    for c in containers:
        c_sec = c.securityContext
        if not c_sec.readOnlyRootFilesystem:
            raise DataError(f"Container {c.name} must enforce readOnlyRootFilesystem: true")
        if c_sec.allowPrivilegeEscalation is not False:
            raise DataError(f"Container {c.name} must set allowPrivilegeEscalation: false")
        if "ALL" not in c_sec.capabilities.drop:
            raise DataError(f"Container {c.name} must drop ALL capabilities")

        # 4. Probes
        if not c.livenessProbe.httpGet:
            raise DataError(f"Container {c.name} missing valid httpGet livenessProbe")
        if not c.readinessProbe.httpGet:
            raise DataError(f"Container {c.name} missing valid httpGet readinessProbe")

        # 5. Resources
        res = c.resources
        if not res.limits.cpu or not res.limits.memory:
            raise DataError(f"Container {c.name} missing resource limits")
        if not res.requests.cpu or not res.requests.memory:
            raise DataError(f"Container {c.name} missing resource requests")

    return {
        "status": "VALID",
        "pod_security_standard": "restricted",
        "containers_verified": len(containers),
        "reconciled_labels": match_labels,
    }


def validate_k8s_service_and_netpol(
    service: dict[str, Any],
    netpol: dict[str, Any],
    deployment: dict[str, Any],
) -> dict[str, Any]:
    """Validate Service and NetworkPolicy integration with Deployment."""
    try:
        svc_model = K8sService.model_validate(service)
        net_model = K8sNetworkPolicy.model_validate(netpol)
    except ValidationError as exc:
        raise DataError(f"Kubernetes service/netpol schema error: {exc}") from exc

    pod_labels = deployment.get("spec", {}).get("template", {}).get("metadata", {}).get("labels", {})

    # Validate Service
    svc_selector = svc_model.spec.selector
    if not svc_selector:
        raise DataError("Service missing spec.selector")
    for k, v in svc_selector.items():
        if pod_labels.get(k) != v:
            raise DataError(f"Service selector does not match deployment pod label {k}")

    # Validate NetworkPolicy
    net_selector = net_model.spec.get("podSelector", {}).get("matchLabels", {})
    for k, v in net_selector.items():
        if pod_labels.get(k) != v:
            raise DataError(f"NetworkPolicy selector does not match deployment pod label {k}")

    return {
        "status": "VALID",
        "service_name": svc_model.metadata.get("name"),
        "netpol_name": net_model.metadata.get("name"),
        "service_type": svc_model.spec.type,
        "policy_types": net_model.spec.get("policyTypes"),
    }


def validate_helm_chart_package(chart_dir: Path | str) -> dict[str, Any]:
    """Validate Helm chart package files and configuration values."""
    path = Path(chart_dir)
    chart_yaml = path / "Chart.yaml"
    values_yaml = path / "values.yaml"
    templates_dir = path / "templates"

    if not chart_yaml.is_file():
        raise DataError("Helm Chart.yaml missing")
    if not values_yaml.is_file():
        raise DataError("Helm values.yaml missing")
    if not templates_dir.is_dir():
        raise DataError("Helm templates directory missing")

    chart_docs = parse_simple_yaml(chart_yaml.read_text(encoding="utf-8"))
    if not chart_docs or chart_docs[0].get("apiVersion") != "v2":
        raise DataError("Helm chart must specify apiVersion: v2")

    values_docs = parse_simple_yaml(values_yaml.read_text(encoding="utf-8"))
    if not values_docs:
        raise DataError("Helm values.yaml is empty")

    val = values_docs[0]
    if not val.get("podSecurityContext", {}).get("runAsNonRoot"):
        raise DataError("Helm values must specify podSecurityContext.runAsNonRoot: true")
    if not val.get("securityContext", {}).get("readOnlyRootFilesystem"):
        raise DataError("Helm values must specify securityContext.readOnlyRootFilesystem: true")
    if not val.get("resources", {}).get("limits"):
        raise DataError("Helm values must specify resource limits")

    template_files = [f.name for f in templates_dir.glob("*.yaml")]
    if "deployment.yaml" not in template_files or "service.yaml" not in template_files:
        raise DataError("Helm templates must contain deployment.yaml and service.yaml")

    return {
        "status": "VALID",
        "chart_name": chart_docs[0].get("name"),
        "chart_version": chart_docs[0].get("version"),
        "templates": template_files,
    }
