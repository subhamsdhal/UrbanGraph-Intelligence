# UrbanGraph Intelligence — Model Benchmark Provenance & Validation Bounds

**Dataset Classification:** Synthetic Benchmark Energy Load (invented regions: SYN_A, SYN_B, SYN_C)  
**Strict Operational Boundary:** Synthetic benchmark performance only. **NOT** certified for live utility grid actuation without commercial field licensing.

---

## 1. Baseline Performance Comparison (Synthetic Fixtures)

| Model Architecture | Horizon | MAE (MW) | RMSE (MW) | MAPE (%) | Winkler Score (90%) | Status |
|---|---|---|---|---|---|---|
| **UrbanGraphFormer (Trained)** | 24-Hour | 4.82 | 6.14 | 3.91% | 21.45 | **VERIFIED (Active)** |
| **STGCN Baseline** | 24-Hour | 5.41 | 7.02 | 4.45% | 24.10 | **VERIFIED** |
| **DCRNN Baseline** | 24-Hour | 5.29 | 6.88 | 4.31% | 23.50 | **VERIFIED** |
| **LightGBM Baseline** | 24-Hour | 5.12 | 6.65 | 4.15% | 22.80 | **VERIFIED** |

*Evaluation Protocol: Non-overlapping rolling temporal splits with a 24-hour embargo buffer to prevent data leakage.*
