# UrbanGraph Intelligence — Master Backup, Verification & Disk Cleanup Guide

## 1. Overview
This runbook guides administrators through safely archiving, pushing to GitHub, and cleaning up legacy development directories to recover disk space following the creation of the self-contained `UrbanGraph-Intelligence-Master` directory.

---

## 2. Pre-Cleanup Verification Procedure

Before deleting any legacy files, verify that the master package is complete:

1. **Confirm Master Package Integrity**:
   Verify that the authoritative release wheel exists and matches the frozen hash:
   - Expected File: `release/urbangraph-0.2.0rc3-py3-none-any.whl`
   - Expected SHA-256: `b0cfc7297bbc6c5f76b790f15bd5cca40873e48c7d34089cc32460fb360b3c97`
   - Expected Size: `1,267,084 bytes`

2. **Verify Reports and Presentation**:
   - `reports/UrbanGraph_Intelligence_Complete_Project_Report.md` (40 sections complete)
   - `reports/UrbanGraph_Intelligence_Complete_Project_Report.pdf` (compiled PDF report)
   - `presentation/UrbanGraph_Intelligence_Project_Presentation.pptx` (20 slides with embedded screenshots)

3. **Verify All 17 Technical Guides**:
   Check that `documentation/` contains all 17 numbered guides.

4. **Verify Acceptance Screenshots**:
   Confirm that all 5 PNG screenshots exist in `acceptance/screenshots/`.

---

## 3. GitHub Publication Runbook

To initialize and push the master directory to a clean GitHub repository:

```bash
# 1. Navigate into the master directory
cd UrbanGraph-Intelligence-Master

# 2. Initialize fresh Git repository
git init -b main

# 3. Verify Git Large File Storage (Git LFS) is active for weights and wheels
git lfs install
git lfs track "*.whl" "*.pt" "*.onnx" "*.pdf" "*.pptx" "*.png"

# 4. Stage all files
git add .

# 5. Commit with definitive release message
git commit -m "feat(release): definitive UrbanGraph Intelligence v0.2.0rc3 release package"

# 6. Set remote repository
git remote add origin https://github.com/<your-org-or-username>/urbangraph-intelligence.git

# 7. Push to GitHub
git push -u origin main
```

---

## 4. Archival Backup (Recommended Prior to Deletion)

Before permanently deleting prior development directories, create a single compressed archival backup:

```powershell
# In PowerShell:
Compress-Archive -Path "UrbanGraph-Intelligence-Master" -DestinationPath "D:\UrbanGraph_Intelligence_0.2.0rc3_Authoritative_Archive.zip" -CompressionLevel Optimal
```

---

## 5. Safe Cleanup of Legacy Development Directories

Once the master directory is safely pushed to GitHub or archived:

### Directories Eligible for Safe Deletion:
- Historical intermediate folders in `new-chat`:
  - `urbangraph_clean_rc2_test` (Temporary cleanroom test workspace)
  - `atf2f4b98b`, `b28`, `b29` (Historical test scratch directories)
  - `v414-git-inventory-mismatch-investigation-20261017`
  - `.pytest_cache`, `.mypy_cache`, `.ruff_cache` in root
  - `urbangraph-production-rc2-2026-10-31/build`
  - `urbangraph-production-rc2-2026-10-31/logs`
  - `urbangraph-production-rc2-2026-10-31/frontend/node_modules` (can recover 300+ MB)
  - `urbangraph-production-rc2-2026-10-31/URBANGRAPH_LOCALHOST_FINAL` (assets now in Master)

### Absolute Preservation Warning:
> **CAUTION**: DO NOT delete `UrbanGraph-Intelligence-Master` or any archive zip files until GitHub publication has been independently verified.
