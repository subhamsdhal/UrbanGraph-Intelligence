# EVIDENCE 12: Helm Chart & Kubernetes Deployment Validation

## 1. Validation Overview
- **Requirement Addressed:** R23 (Kubernetes Packaging & Declarative Orchestration)
- **Status:** VERIFIED (Schema Validation, Security Standards Audit & Negative Testing)
- **External Gate:** Client Kubernetes Staging / Production Cluster Provisioning (External Gate G-03)

UrbanGraph Intelligence packages a production Helm chart (`infra/helm/energraph-ai`) and raw Kubernetes manifests (`infra/k8s/`) conforming to the Kubernetes **Restricted Pod Security Standard**.

---

## 2. Pod Security Standards (Restricted Profile) Compliance
All manifests and Helm templates were audited against the Pydantic schema validation engine in `urbangraph.security.k8s_validator`:

| Security Standard Rule | Value Enforced | Verification Mechanism | Status |
| :--- | :--- | :--- | :--- |
| `runAsNonRoot` | `true` | Checked at Pod spec level | PASS |
| `runAsUser` | `65534` (`nobody`) | Non-root UID enforcement | PASS |
| `readOnlyRootFilesystem` | `true` | Container filesystem immutability | PASS |
| `allowPrivilegeEscalation` | `false` | SUID privilege escalation blocked | PASS |
| Capabilities Drop | `ALL` | All Linux kernel capabilities dropped | PASS |
| Seccomp Profile | `RuntimeDefault` | Seccomp syscall filtering enforced | PASS |

Negative tests verify that any violation (e.g. `runAsNonRoot: false`, writable root filesystem, missing drop ALL) is rejected immediately with a `DataError`.

---

## 3. Service & NetworkPolicy Reconciliation
1. **Deployment Selector:** `app.kubernetes.io/name: energraph-ai`
2. **Service Target:** ClusterIP service targeting port 8000 matches Deployment labels exactly.
3. **NetworkPolicy:** Ingress restricted to authorized namespaces and ports, egress restricted to DNS and external API dependencies. Pod selector matches Deployment pod labels.

---

## 4. Helm Chart Structure
- Chart Name: `energraph-ai`
- Chart Version: `0.2.0`
- API Version: `v2`
- Templates Validated:
  - `deployment.yaml`
  - `service.yaml`
  - `ingress.yaml`
  - `networkpolicy.yaml`
  - `pdb.yaml` (PodDisruptionBudget)
  - `serviceaccount.yaml`
  - `configmap.yaml`
