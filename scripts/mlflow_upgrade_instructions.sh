#!/usr/bin/env bash
# Script: mlflow_upgrade_instructions.sh
# Purpose: Print guidance and a safe example command for upgrading MLflow DB schema.
# NOTE: This script does NOT perform the migration - it only prints the command you should run
# after taking an appropriate backup of your MLflow database.

if [ -z "$1" ]; then
  echo "Usage: $0 <database_uri>"
  echo "Example: $0 sqlite:///mlflow.db" 
  exit 1
fi

DB_URI="$1"

cat <<EOF
MLflow DB Upgrade Guidance
==========================
1. BACKUP your database before applying any schema migrations. For SQLite, make a copy of the file.
   Example (SQLite): cp mlflow.db mlflow.db.bak
   Example (Postgres): pg_dump -U <user> -h <host> -d <db> -f mlflow_backup.sql

2. Run the MLflow DB upgrade command:
   mlflow db upgrade "$DB_URI"

3. Note: Schema migration may result in database downtime. Consult your DB docs and run during maintenance windows.

4. If you use hosted CI runners, consider migrating a copy of your DB in a temporary environment first.

EOF
