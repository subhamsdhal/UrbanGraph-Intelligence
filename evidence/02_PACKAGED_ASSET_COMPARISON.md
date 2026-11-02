# UrbanGraph Intelligence — Packaged Asset & Wheel Integrity Comparison

**Wheel File:** `urbangraph-0.2.0rc2-py3-none-any.whl`  
**Wheel SHA-256 on Disk:** `f41ef0dc08b63ea2fe277edbb5d46a20cef73e388266b5178d88930a6f4d0217`  
**Wheel Size:** `1259602` bytes  
**Timestamp of Analysis:** `2026-11-01`  

---

## 1. Finding & Root Cause Analysis

- **Investigation Result:** `STALE ASSETS DETECTED`
- **Discrepancy Analysis:**
  The wheel `urbangraph-0.2.0rc2-py3-none-any.whl` with hash `f41ef0dc08b63ea2fe277edbb5d46a20cef73e388266b5178d88930a6f4d0217` was packaged **BEFORE** the security modification in `frontend/src/api.ts` that eliminated token reading from URL parameters (`params.get('token')`).
  When `npm run build` was executed, the newly generated JavaScript chunk hashes changed in `frontend/dist` and were synchronized to `src/urbangraph/static/`.
  However, the `.whl` in `dist/` was preserved unchanged from the prior session to keep its byte hash matching `RELEASE_METADATA.json`.
  Consequently, the wheel on disk contains the **pre-security-fix** frontend bundle!

---

## 2. Asset Checksum Comparison Matrix

| Asset Path | In Wheel? | Wheel Hash (First 12) | Local static Hash (First 12) | Frontend Dist Hash (First 12) | Match Status |
|---|---|---|---|---|---|
| `assets/expression-Bl_GL2tO.js` | YES | `71e7147302c7` | `71e7147302c7` | `71e7147302c7` | **IDENTICAL** |
| `assets/index-CMUR15TM.js` | NO | `MISSING` | `3576868bb88f` | `3576868bb88f` | **STALE / MISMATCH** |
| `assets/index-CzyO8_Yh.js` | NO | `MISSING` | `cd9482f14e8d` | `cd9482f14e8d` | **STALE / MISMATCH** |
| `assets/index-D1GD_yCk.css` | YES | `59607fcbd73e` | `59607fcbd73e` | `59607fcbd73e` | **IDENTICAL** |
| `assets/index-DEx8_ghy.js` | NO | `MISSING` | `b609af4267d5` | `b609af4267d5` | **STALE / MISMATCH** |
| `assets/index-DGMEYuY9.js` | YES | `905f6c0563f6` | `MISSING` | `MISSING` | **STALE / MISMATCH** |
| `assets/index-DYZtdEog.js` | YES | `15aed70016c2` | `MISSING` | `MISSING` | **STALE / MISMATCH** |
| `assets/index-DqWszptW.js` | YES | `407194f077d4` | `MISSING` | `MISSING` | **STALE / MISMATCH** |
| `index.html` | YES | `866acf2bf141` | `fac3cc67ce4a` | `fac3cc67ce4a` | **STALE / MISMATCH** |

---

## 3. Mandatory Correction Action Plan

Following Section 3 of the Master Engineering Directive:
1. **Preserve Original RC2 Artifact:** Keep `urbangraph-0.2.0rc2-py3-none-any.whl` (`f41ef0dc08b6...`) untouched in a preserved archive location or keep its record permanent.
2. **Produce Distinct Versioned Artifact:** Following PEP 440 and the project's release versioning policy, produce **`0.2.0rc3`** (or `0.2.0` Final Candidate / `0.2.0rc2.post1` depending on versioning policy). Wait! The directive states:
   > "If release bytes change, produce a distinctly identified versioned artifact following the project's release-versioning policy. Preserve the original RC2 artifact. Do not create a new release label merely for appearance; create a new candidate only when the software or packaged bytes genuinely require one. Record exact old and new hashes."
3. Package the updated, secure frontend bundle, trained neural weights, and demo fixtures into the new wheel.
4. Perform clean-room verification on the newly minted candidate.
