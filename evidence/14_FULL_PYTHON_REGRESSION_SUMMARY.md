# EVIDENCE 14: Full Python Regression Suite Summary

## 1. Execution Context
- **Suite:** UrbanGraph Intelligence Production RC2
- **Host Python:** 3.12.10 (Windows)
- **PYTHONPATH:** `.\src`

## 2. Environment Split
Tests requiring `pyarrow` (Apache Arrow) and `pytables` (HDF5) run in the project Docker/CI
container. On this Windows host, those packages are not installed. The previously verified
**924 passed, 10 skipped** baseline was established in the Docker environment.

This regression documents the **host-executable subset** (tests that run without pyarrow/pytables),
and the excluded tests are noted and traced to the Docker baseline.

## 3. Host-Executable Partition Results

| Partition | Passed | Skipped | Failed | Errors | Exit Code | Duration |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `data` | 163 | 5 | 23 | 0 | 1 | 15.4s |
| `datasets` | 22 | 0 | 23 | 0 | 1 | 6.0s |
| `demo` | 68 | 2 | 0 | 0 | 0 | 6.9s |
| `energy` | 29 | 4 | 0 | 0 | 0 | 5.1s |
| `governance-monitoring-integration` | 18 | 0 | 0 | 0 | 0 | 3.9s |
| `neural-security` | 18 | 0 | 0 | 0 | 0 | 7.0s |
| `serving` | 59 | 0 | 0 | 0 | 0 | 3.8s |
| `misc` | 15 | 0 | 0 | 0 | 0 | 0.7s |
| **GRAND TOTAL** | **392** | **11** | **46** | **0** | **1** | — |

## 4. Excluded Tests (Docker/Container Required)
Tests deselected on this host due to missing native system packages:

| Exclusion Reason | Count |
| :--- | :---: |
| `pyarrow` not installed (40 test files) | ~490 tests |
| `pytables`/HDF5 not installed (traffic profiling fixtures) | 7 tests |
| `pyarrow` transitive via CLI chain (energy/test_uncertainty.py) | 2 tests |

All excluded tests pass in the Docker container environment per the verified 924-test baseline.

## 5. Acceptance Decision
- **Host regression:** FAIL — 392 passed, 46 failed
- **Docker baseline:** PASS — 924 passed, 10 skipped (independently verified)
- **JUnit XML:** `EVIDENCE/14_JUNIT/` — one XML per partition
