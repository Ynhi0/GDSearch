"""Check installed package versions against repository pins and flag likely incompatibilities.

Usage:
    python scripts/check_dependencies.py

This script inspects installed versions of mlflow, sqlalchemy, alembic, pydantic and reports discrepancies
and recommended pins based on the project's `requirements.txt`.
"""
from __future__ import annotations
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
import re
import logging

logging.basicConfig(level=logging.INFO)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQ_FILE = PROJECT_ROOT / "requirements.txt"


def parse_pin(pkg: str) -> str | None:
    if not REQ_FILE.exists():
        return None
    txt = REQ_FILE.read_text()
    pattern = re.compile(rf"^{re.escape(pkg)}(==[^\s#]+)", re.MULTILINE)
    m = pattern.search(txt)
    if m:
        return m.group(1).lstrip('=')
    return None


def get_installed(pkg: str) -> str | None:
    try:
        return version(pkg)
    except PackageNotFoundError:
        return None


def main():
    pkgs = ["mlflow", "sqlalchemy", "alembic", "pydantic"]
    info = {}

    for p in pkgs:
        info[p] = {
            'installed': get_installed(p),
            'pinned': parse_pin(p),
        }

    for p, v in info.items():
        logging.info("%s: installed=%s pinned=%s", p, v['installed'], v['pinned'])

    # Simple heuristic checks
    mlflow_v = info['mlflow']['installed']
    sa_v = info['sqlalchemy']['installed']
    alembic_v = info['alembic']['installed']

    if mlflow_v and sa_v:
        logging.info("Checking compatibility heuristics between mlflow (%s) and sqlalchemy (%s)", mlflow_v, sa_v)
        # Heuristic: mlflow 2.x generally requires SQLAlchemy < 2.2 in many 2.x releases
        if sa_v and sa_v.startswith('2.'):
            logging.warning("Detected SQLAlchemy 2.x (%s). Some MLflow releases have limited support for SQLAlchemy 2.x. Consider pinning sqlalchemy<2.2 if you encounter DB errors.", sa_v)

    if mlflow_v and alembic_v:
        if alembic_v and alembic_v.startswith('1.'):
            logging.info("Alembic 1.x detected (%s). This is typically compatible with MLflow 2.x, but if you see migration errors consider updating to latest alembic 1.x.", alembic_v)

    logging.info("If you see version warnings, add a CI job to run `python scripts/check_dependencies.py` and fail the build on critical mismatches.")


if __name__ == '__main__':
    main()