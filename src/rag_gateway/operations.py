import json
import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import uuid4


class OperationsStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS document_loader_jobs (
                    job_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    data_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS retrieval_evaluations (
                    run_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    top_k INTEGER NOT NULL,
                    result_json TEXT NOT NULL,
                    metrics_json TEXT NOT NULL DEFAULT '{}',
                    case_count INTEGER NOT NULL DEFAULT 0,
                    dataset_hash TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL
                )
                """
            )
            evaluation_columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(retrieval_evaluations)")
            }
            if "metrics_json" not in evaluation_columns:
                connection.execute(
                    "ALTER TABLE retrieval_evaluations "
                    "ADD COLUMN metrics_json TEXT NOT NULL DEFAULT '{}'"
                )
            if "case_count" not in evaluation_columns:
                connection.execute(
                    "ALTER TABLE retrieval_evaluations "
                    "ADD COLUMN case_count INTEGER NOT NULL DEFAULT 0"
                )
            if "dataset_hash" not in evaluation_columns:
                connection.execute(
                    "ALTER TABLE retrieval_evaluations "
                    "ADD COLUMN dataset_hash TEXT NOT NULL DEFAULT ''"
                )
            evaluations_missing_hash = connection.execute(
                "SELECT run_id, result_json FROM retrieval_evaluations WHERE dataset_hash = ''"
            ).fetchall()
            for row in evaluations_missing_hash:
                evaluation = json.loads(row["result_json"])
                connection.execute(
                    "UPDATE retrieval_evaluations SET dataset_hash = ? WHERE run_id = ?",
                    (_evaluation_dataset_hash(evaluation), row["run_id"]),
                )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_loader_jobs_tenant_created "
                "ON document_loader_jobs(tenant_id, created_at DESC)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_evaluations_tenant_created "
                "ON retrieval_evaluations(tenant_id, created_at DESC)"
            )
            interrupted_jobs = connection.execute(
                """
                SELECT job_id, data_json FROM document_loader_jobs
                WHERE status NOT IN (
                    'completed', 'completed_with_errors', 'failed', 'interrupted', 'cancelled'
                )
                """
            ).fetchall()
            for row in interrupted_jobs:
                job = json.loads(row["data_json"])
                job["status"] = "interrupted"
                job["events"].append(
                    {
                        "stage": "interrupted",
                        "message": "Gateway stopped before this job completed",
                    }
                )
                connection.execute(
                    "UPDATE document_loader_jobs SET status = ?, data_json = ?, updated_at = ? "
                    "WHERE job_id = ?",
                    ("interrupted", json.dumps(job, ensure_ascii=False), _now(), row["job_id"]),
                )
        path.chmod(0o600)

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def save_job(self, job: dict[str, Any]) -> None:
        now = _now()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO document_loader_jobs
                    (job_id, tenant_id, status, data_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(job_id) DO UPDATE SET
                    tenant_id = excluded.tenant_id,
                    status = excluded.status,
                    data_json = excluded.data_json,
                    updated_at = excluded.updated_at
                """,
                (
                    job["job_id"],
                    job["tenant_id"],
                    job["status"],
                    json.dumps(job, ensure_ascii=False),
                    job.get("created_at", now),
                    now,
                ),
            )

    def get_job(
        self, job_id: str, allowed_tenants: list[str] | None = None
    ) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT data_json, tenant_id FROM document_loader_jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
        if row is None or (
            allowed_tenants is not None and row["tenant_id"] not in allowed_tenants
        ):
            return None
        return json.loads(row["data_json"])

    def list_jobs(
        self, allowed_tenants: list[str] | None = None, limit: int = 50
    ) -> list[dict[str, Any]]:
        with self._connect() as connection:
            if allowed_tenants is None:
                rows = connection.execute(
                    "SELECT data_json FROM document_loader_jobs "
                    "ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            elif not allowed_tenants:
                return []
            else:
                placeholders = ",".join("?" for _ in allowed_tenants)
                rows = connection.execute(
                    "SELECT data_json FROM document_loader_jobs "
                    f"WHERE tenant_id IN ({placeholders}) ORDER BY created_at DESC LIMIT ?",
                    (*allowed_tenants, limit),
                ).fetchall()
        jobs = [json.loads(row["data_json"]) for row in rows]
        summary_fields = (
            "job_id",
            "tenant_id",
            "status",
            "created_at",
            "discovered",
            "eligible",
            "skipped",
            "processed",
            "succeeded",
            "failed",
            "retryable",
        )
        return [
            {
                **{key: job[key] for key in summary_fields if key in job},
                "retryable": job.get("status") in {"completed_with_errors", "failed"}
                and bool(job.get("failed"))
                and job.get("directory") != "browser upload",
            }
            for job in jobs
        ]

    def clear_completed_jobs(self, allowed_tenants: list[str] | None = None) -> int:
        terminal_statuses = ("completed", "completed_with_errors", "failed", "interrupted", "cancelled")
        status_placeholders = ",".join("?" for _ in terminal_statuses)
        with self._connect() as connection:
            if allowed_tenants is None:
                cursor = connection.execute(
                    "DELETE FROM document_loader_jobs "
                    f"WHERE status IN ({status_placeholders})",
                    terminal_statuses,
                )
            elif not allowed_tenants:
                return 0
            else:
                tenant_placeholders = ",".join("?" for _ in allowed_tenants)
                cursor = connection.execute(
                    "DELETE FROM document_loader_jobs "
                    f"WHERE status IN ({status_placeholders}) "
                    f"AND tenant_id IN ({tenant_placeholders})",
                    (*terminal_statuses, *allowed_tenants),
                )
        return cursor.rowcount

    def save_evaluation(self, result: dict[str, Any]) -> str:
        run_id = uuid4().hex
        created_at = _now()
        dataset_hash = _evaluation_dataset_hash(result)
        saved = {
            **result,
            "run_id": run_id,
            "created_at": created_at,
            "dataset_id": dataset_hash,
        }
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO retrieval_evaluations
                    (run_id, tenant_id, top_k, result_json, metrics_json, case_count,
                     dataset_hash, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    result["tenant_id"],
                    result["top_k"],
                    json.dumps(saved, ensure_ascii=False),
                    json.dumps(result["metrics"], separators=(",", ":")),
                    len(result["cases"]),
                    dataset_hash,
                    created_at,
                ),
            )
        return run_id

    def list_evaluations(
        self,
        allowed_tenants: list[str] | None = None,
        tenant_id: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        parameters: list[Any] = []
        if tenant_id is not None:
            clauses.append("tenant_id = ?")
            parameters.append(tenant_id)
        if allowed_tenants is not None:
            if not allowed_tenants:
                return []
            placeholders = ",".join("?" for _ in allowed_tenants)
            clauses.append(f"tenant_id IN ({placeholders})")
            parameters.extend(allowed_tenants)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT run_id, tenant_id, top_k, metrics_json, case_count, dataset_hash, created_at "
                "FROM retrieval_evaluations "
                f"{where} ORDER BY created_at DESC LIMIT ?",
                (*parameters, limit),
            ).fetchall()
        return [
            {
                "run_id": row["run_id"],
                "tenant_id": row["tenant_id"],
                "top_k": row["top_k"],
                "metrics": json.loads(row["metrics_json"]),
                "case_count": row["case_count"],
                "dataset_id": row["dataset_hash"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def get_evaluation(
        self, run_id: str, allowed_tenants: list[str] | None = None
    ) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT result_json, tenant_id, dataset_hash "
                "FROM retrieval_evaluations WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        if row is None or (
            allowed_tenants is not None and row["tenant_id"] not in allowed_tenants
        ):
            return None
        result = json.loads(row["result_json"])
        if "dataset_id" not in result:
            result["dataset_id"] = row["dataset_hash"] or _evaluation_dataset_hash(result)
        return result


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _evaluation_dataset_hash(result: dict[str, Any]) -> str:
    dataset = sorted(
        (
            case["query"],
            sorted(case["expected_document_ids"]),
        )
        for case in result["cases"]
    )
    serialized = json.dumps(dataset, ensure_ascii=False, separators=(",", ":"))
    return sha256(serialized.encode("utf-8")).hexdigest()
