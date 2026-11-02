# 15 - Backup and Recovery Runbook

## 1. Backup Strategy
- **Security Database (`tokens.db`)**: Perform daily online SQLite backups using SQLite backup API or `.backup` command:
  ```bash
  sqlite3 /app/security/tokens.db ".backup '/app/security/backup_tokens.db'"
  ```
- **Model Weights & Fixtures**: Static files stored in version control (Git LFS) and packaged into the release wheel.

## 2. Disaster Recovery Procedure
1. Stop running services.
2. Restore `tokens.db` from latest verified snapshot.
3. Validate database integrity:
   ```bash
   sqlite3 tokens.db "PRAGMA integrity_check;"
   ```
4. Restart services and execute verification script.
