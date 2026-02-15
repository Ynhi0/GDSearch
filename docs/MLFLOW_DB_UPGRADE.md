# MLflow DB Schema Upgrades (local SQLite / Postgres) 🗄️🔧

When MLflow is upgraded between major releases, the DB schema may need to be migrated.
This file documents safe, repeatable steps.

## Backup your DB (CRITICAL)
- SQLite: copy the file
  - cp mlruns.db mlruns.db.bak
  - or use: python -c "import shutil; shutil.copy2('mlruns.db', 'mlruns.db.bak')"
- Postgres: use `pg_dump -Fc -f mlruns.dump -U <user> <db>` or your cloud provider's backup

## Upgrade commands (MLflow 2.x)
- Using the MLflow CLI (recommended):

  python -m mlflow db upgrade <SQLALCHEMY_DATABASE_URI>

  Example for sqlite:
  python -m mlflow db upgrade sqlite:///./mlruns.db

- Alternatively, use Alembic directly (advanced):
  - cd $(python -c "import mlflow; import os; print(os.path.dirname(mlflow.__file__))")
  - alembic -c mlflow/alembic.ini upgrade head

> The recommended and supported command is `python -m mlflow db upgrade` for the MLflow release you have installed.

## Automation helper
We added `scripts/upgrade_mlflow_db.py` which:
- creates backups for sqlite URIs if `--backup` is passed
- runs `python -m mlflow db upgrade <uri>`

Usage:

```
python scripts/upgrade_mlflow_db.py --db-uri sqlite:///./mlruns.db --backup
```

## CI Recommendations
- Add a CI job that runs `scripts/check_dependencies.py` to detect MLflow version changes and warn maintainers.
- When upgrading the pinned `mlflow==` in `requirements.txt`, add a PR checklist item to run the DB upgrade locally and update this doc if the upgrade requires manual steps.

## Troubleshooting
- If you see Alembic migration errors that reference a missing revision, ensure you are running the `mlflow` CLI from the same Python environment as the pinned MLflow version.
- If in doubt, backup the DB and restore from backup before retrying the upgrade.
