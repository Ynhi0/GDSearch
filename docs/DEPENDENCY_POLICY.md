# Dependency compatibility policy 🧾

To prevent subtle runtime incompatibilities (Alembic / SQLAlchemy / MLflow / Pydantic), we use the following policy:

- Pin the experiment-tracking package: `mlflow==2.19.0`
- Constrain SQLAlchemy to a compatible range: `SQLAlchemy>=1.4,<2.2`
- Constrain Alembic to `alembic>=1.8,<2.0`
- Keep Pydantic within tested range: `pydantic>=2.0,<2.12`

Whenever `mlflow` is bumped in `requirements.txt`:
1. Run `python scripts/check_dependencies.py` and verify no new incompatibilities are reported.
2. Run `python scripts/upgrade_mlflow_db.py --check --db-uri <your-db-uri>` and perform a local upgrade in a branch.
3. Add database upgrade verification to the PR checklist.

CI recommendation:
- Add a PR check (we added `.github/workflows/dependency_checks.yml`) to run `scripts/check_dependencies.py` and fail if critical mismatches are found.

Rationale:
- SQLAlchemy 2.x has backward-incompatible changes that can break Alembic migrations or MLflow DB interactions; capping at `<2.2` is conservative and compatible with MLflow 2.x series.
