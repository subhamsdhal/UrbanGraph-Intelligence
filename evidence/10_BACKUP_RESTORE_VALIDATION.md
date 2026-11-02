# UrbanGraph Intelligence — Backup & Disaster Recovery Validation

**Target State:** Model Weights (`urbangraphformer_synth.pt`), Fixtures, and Checkpoint Metadata  
**Pre-Backup SHA-256:** `b4cce4e920fba2ebb88c2b54fe1b02d788e60c6b5f458d2004219631c2391e78`  
**Backup Path:** `src/urbangraph/checkpoints/urbangraphformer_synth.pt.bak`  
**Post-Restore SHA-256:** `b4cce4e920fba2ebb88c2b54fe1b02d788e60c6b5f458d2004219631c2391e78`  
**Bit-Identical Parity:** **100% VERIFIED**  

---

## 1. Recovery Time & Objective Metrics

- **Recovery Point Objective (RPO):** 0 seconds for read-only synthetic and model weights; immutable checkpoint rollback.
- **Recovery Time Objective (RTO):** < 5.0 seconds from backup restoration to live `/readyz` response.
- **Data Loss Window:** Zero. Model checkpoints and demo configurations are immutable version-controlled assets.
