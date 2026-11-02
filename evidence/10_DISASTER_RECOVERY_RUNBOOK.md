# UrbanGraph Intelligence — Disaster Recovery Runbook

## 1. Emergency Model Rollback / Corrupted State Restoration

If a model checkpoint is corrupted or an unauthorized weight change is detected:
```bash
# 1. Stop active server PID gracefully
kill -15 $PID

# 2. Restore golden backup checkpoint
cp /opt/urbangraph/checkpoints/urbangraphformer_synth.pt.bak /opt/urbangraph/checkpoints/urbangraphformer_synth.pt

# 3. Verify SHA-256 integrity
sha256sum /opt/urbangraph/checkpoints/urbangraphformer_synth.pt

# 4. Run preflight verification
urbangraph-server --check-only

# 5. Restart production server
urbangraph-server --port 8080 --workers 1 &
```

## 2. Hard Crash Recovery Procedure
If the node crashes abruptly:
1. Docker / Kubernetes `RestartPolicy: Always` automatically restarts the container.
2. The server binds cleanly to the port using `SO_REUSEADDR`.
3. Kubernetes readiness probe polls `/readyz` every 2 seconds until healthy before routing traffic.
