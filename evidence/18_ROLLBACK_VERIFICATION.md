# EVIDENCE 18: Release Rollback Verification Report (RC3 -> RC2)

## 1. Executive Summary
- **Current Version:** `0.2.0rc3` (SHA256: `44cfa7a640ba8592cb1e32df3d344fa6a9a456a6bc8a0c20b772b37ce107999f`)
- **Target Rollback Version:** `0.2.0rc2` (SHA256: `f41ef0dc08b63ea2fe277edbb5d46a20cef73e388266b5178d88930a6f4d0217`)
- **Rollback Feasibility:** VERIFIED & SAFE
- **Compatibility Breakages:** ZERO breaking schema or persistence changes between RC2 and RC3.

---

## 2. Artifact Provenance & Preservation
Both release artifacts are preserved in `dist/` and cryptographically verified:

| Release Artifact | SHA-256 Digest | File Size | Description |
| :--- | :--- | :---: | :--- |
| `urbangraph-0.2.0rc2-py3-none-any.whl` | `f41ef0dc08b63ea2fe277edbb5d46a20cef73e388266b5178d88930a6f4d0217` | 1,259,602 bytes | Preserved baseline RC2 release package |
| `urbangraph-0.2.0rc3-py3-none-any.whl` | `44cfa7a640ba8592cb1e32df3d344fa6a9a456a6bc8a0c20b772b37ce107999f` | 2,111,639 bytes | Corrected release package with updated static bundle |

---

## 3. Rollback Verification Suite Results

| Test Check | Condition Verified | Result |
| :--- | :--- | :---: |
| Distinct Artifact Checksums | RC2 and RC3 wheels have distinct SHA-256 digests and file sizes | PASS |
| Asset Version Isolation | RC2 wheel contains prior static assets; RC3 wheel contains patched bundle | PASS |
| Schema Backward-Compatibility | All API routes, Pydantic response models, and headers match | PASS |
| Model Checkpoint Portability | `urbangraphformer_synth.pt` is bit-identical and loads identically on both | PASS |

**All 4 rollback verification tests passed.**

---

## 4. Rollback Execution Procedure
In the event of an operational need to roll back:
```bash
# Helm rollback:
helm rollback energraph -n urbangraph-staging

# Or Wheel reinstall:
pip install dist/urbangraph-0.2.0rc2-py3-none-any.whl --force-reinstall
```
