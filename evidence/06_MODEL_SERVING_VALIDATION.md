# UrbanGraph Intelligence — Model Serving & Inference Validation

**Model Checkpoint:** `src/urbangraph/checkpoints/urbangraphformer_synth.pt`  
**Checkpoint SHA-256:** `b4cce4e920fba2ebb88c2b54fe1b02d788e60c6b5f458d2004219631c2391e78`  
**Architecture:** `UrbanGraphFormer` (Multi-Head Spatial Attention + Causal Temporal Attention)  
**Trained Weights Version:** `urbangraphformer-trained-v1`  

---

## 1. Inference Engine Conformance

- **Input Tensor Dimensions:** `[batch_size=1, sequence_length=12, num_nodes=3, num_features=1]`
- **Spatial Graph Dimensions:** `[num_nodes=3, num_nodes=3]` (Normalized adjacency list of length `num_graphs=3`)
- **Output Forecast Dimensions:** `[batch_size=1, horizon=24, num_nodes=3]`
- **Numerical Finiteness:** 100% of tensor predictions are strictly finite (`torch.isfinite() == True`). Zero NaN or Inf values.
- **Conformal Uncertainty:** Split-conformal quantile intervals computed with coverage guarantee.
- **Dropout Degradation Factor:** When sensors are dropped, the `RuntimeResilienceController` applies causal-carry imputation and inflates interval width dynamically by `uncertainty_inflation_factor`.

---

## 2. Checkpoint Provenance & Security

- Checked against safe-unpickling filters: Checkpoint loads cleanly with weights verified.
- Corruption Rejection: Severely corrupted files trigger a clean preflight exit code `1`.
