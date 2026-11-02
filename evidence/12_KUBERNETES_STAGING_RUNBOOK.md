# EVIDENCE 12: Kubernetes Staging & Production Deployment Runbook

## 1. Prerequisites
- Kubernetes cluster v1.26+ (AKS, EKS, GKE, or vanilla Kubernetes).
- Helm v3.10+ installed.
- Ingress controller (e.g. ingress-nginx or cloud load balancer) configured with TLS termination.
- Production container image `urbangraph:0.2.0rc2` pushed to client internal container registry.

---

## 2. Dry-Run & Linting
Prior to staging deployment, validate chart syntax and templates:
```bash
helm lint infra/helm/energraph-ai
helm template energraph infra/helm/energraph-ai --namespace urbangraph-staging --debug
```

---

## 3. Staging Installation
```bash
# 1. Create staging namespace
kubectl create namespace urbangraph-staging

# 2. Deploy secret containing production API master key
kubectl create secret generic urbangraph-auth   --from-literal=token="prod_sec_initial_master_key_991823"   --namespace urbangraph-staging

# 3. Install Helm release
helm install energraph infra/helm/energraph-ai   --namespace urbangraph-staging   --set image.repository="registry.internal/urbangraph"   --set image.tag="0.2.0rc2"   --set replicas=2
```

---

## 4. Health Verification & Rollout Status
```bash
# Watch deployment status
kubectl rollout status deployment/energraph-ai -n urbangraph-staging

# Verify pod security contexts
kubectl get pods -n urbangraph-staging -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.spec.securityContext}{"\n"}{end}'

# Test readiness probe
kubectl exec -it deployment/energraph-ai -n urbangraph-staging -- curl -s http://localhost:8000/readyz
```

---

## 5. Automated Rollback Procedure
If deployment health checks fail or alerts fire post-rollout:
```bash
helm rollback energraph -n urbangraph-staging
```
