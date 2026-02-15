"""Upgrade MLflow database schema safely.

Usage:
    python scripts/upgrade_mlflow_db.py --db-uri sqlite:///./mlruns.db --backup

This script will:
- For sqlite URIs, create a safe file copy before applying upgrade
- Run `mlflow db upgrade <db_uri>` via subprocess (requires mlflow installed)
- Provide a --check flag to only print the current mlflow version and recommended upgrade command

Note: Always backup your production DBs (pg_dump for Postgres) before use.
"""
from __future__ import annotations
import argparse
import os
import shutil
import subprocess
import sys
from urllib.parse import urlparse
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)


def is_sqlite_uri(uri: str) -> bool:
    return uri.startswith("sqlite:")


def backup_sqlite(uri: str, backup_dir: Path) -> Path:
    # sqlite:///relative/path or sqlite:////absolute/path
    parsed = urlparse(uri)
    # Path portion of sqlite URI
    if parsed.path:
        db_path = Path(parsed.path)
    else:
        raise ValueError(f"Cannot identify sqlite path from URI: {uri}")

    if not db_path.exists():
        raise FileNotFoundError(f"SQLite DB not found: {db_path}")

    backup_dir.mkdir(parents=True, exist_ok=True)
    target = backup_dir / (db_path.name + ".bak")
    shutil.copy2(db_path, target)
    logging.info("Backed up SQLite DB %s -> %s", db_path, target)
    return target


def run_mlflow_db_upgrade(uri: str) -> int:
    cmd = [sys.executable, "-m", "mlflow", "db", "upgrade", uri]
    logging.info("Running: %s", " ".join(cmd))
    return subprocess.call(cmd)


def main():
    parser = argparse.ArgumentParser(description="Safely upgrade MLflow DB schema")
    parser.add_argument("--db-uri", required=True, help="SQLAlchemy/MLflow DB URI (e.g. sqlite:///mlruns.db or postgresql://user:pass@host/db)")
    parser.add_argument("--backup", action="store_true", help="Create a backup before upgrading (sqlite only)")
    parser.add_argument("--backup-dir", default="./backups/mlflow", help="Backup directory")
    parser.add_argument("--check", action="store_true", help="Show mlflow version and exit")

    args = parser.parse_args()

    try:
        import mlflow
    except Exception as e:
        logging.error("mlflow is not installed in the active environment: %s", e)
        return 2

    logging.info("Detected mlflow version: %s", getattr(mlflow, '__version__', 'unknown'))

    if args.check:
        logging.info("Recommended upgrade command: python -m mlflow db upgrade %s", args.db_uri)
        return 0

    if is_sqlite_uri(args.db_uri) and args.backup:
        try:
            backup_sqlite(args.db_uri, Path(args.backup_dir))
        except Exception as e:
            logging.error("Failed to backup sqlite DB: %s", e)
            return 3

    rc = run_mlflow_db_upgrade(args.db_uri)
    if rc != 0:
        logging.error("mlflow db upgrade exited with code %d", rc)
        return rc

    logging.info("mlflow db upgrade completed successfully")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())