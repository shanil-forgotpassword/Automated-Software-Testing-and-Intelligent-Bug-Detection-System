from __future__ import annotations

import logging
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("CODEGUARD_DATA_DIR", PROJECT_ROOT / "data")).resolve()
DATABASE_PATH = DATA_DIR / "codeguard.sqlite3"

SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS projects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    objectives TEXT NOT NULL DEFAULT '',
    language TEXT NOT NULL DEFAULT 'Python',
    project_type TEXT NOT NULL DEFAULT '',
    main_features TEXT NOT NULL DEFAULT '',
    expected_inputs TEXT NOT NULL DEFAULT '',
    expected_outputs TEXT NOT NULL DEFAULT '',
    constraints TEXT NOT NULL DEFAULT '',
    is_pinned INTEGER NOT NULL DEFAULT 0 CHECK (is_pinned IN (0, 1)),
    is_demo INTEGER NOT NULL DEFAULT 0 CHECK (is_demo IN (0, 1)),
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_projects_title_unique
    ON projects(lower(title));
CREATE TABLE IF NOT EXISTS requirements (
    id TEXT NOT NULL,
    project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    description TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY(project_id, id)
);
CREATE TABLE IF NOT EXISTS source_files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    filename TEXT NOT NULL,
    original_content TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(project_id, filename)
);
CREATE TABLE IF NOT EXISTS code_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    source_file_id INTEGER NOT NULL REFERENCES source_files(id) ON DELETE CASCADE,
    version_number INTEGER NOT NULL,
    content TEXT NOT NULL,
    kind TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(project_id, version_number)
);
CREATE TABLE IF NOT EXISTS test_cases (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    test_id TEXT NOT NULL,
    name TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    testing_objective TEXT NOT NULL DEFAULT '',
    complexity TEXT NOT NULL DEFAULT 'Basic',
    requirement_id TEXT,
    category TEXT NOT NULL,
    function_name TEXT NOT NULL,
    target_class TEXT NOT NULL DEFAULT '',
    target_kind TEXT NOT NULL DEFAULT 'function',
    inputs_json TEXT NOT NULL,
    keyword_inputs_json TEXT NOT NULL DEFAULT '{}',
    constructor_inputs_json TEXT NOT NULL DEFAULT '[]',
    constructor_keyword_inputs_json TEXT NOT NULL DEFAULT '{}',
    setup_json TEXT NOT NULL DEFAULT '[]',
    expected_json TEXT NOT NULL,
    database_assertions_json TEXT NOT NULL DEFAULT '[]',
    expected_exception TEXT NOT NULL DEFAULT '',
    preconditions TEXT NOT NULL DEFAULT '',
    priority TEXT NOT NULL DEFAULT 'Medium',
    generation_source TEXT NOT NULL DEFAULT 'manual',
    approval_status TEXT NOT NULL DEFAULT 'pending'
        CHECK (approval_status IN ('pending', 'approved', 'rejected')),
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(project_id, test_id)
);
CREATE TABLE IF NOT EXISTS test_executions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    code_version_id INTEGER NOT NULL REFERENCES code_versions(id),
    status TEXT NOT NULL,
    total INTEGER NOT NULL DEFAULT 0,
    passed INTEGER NOT NULL DEFAULT 0,
    failed INTEGER NOT NULL DEFAULT 0,
    skipped INTEGER NOT NULL DEFAULT 0,
    errors INTEGER NOT NULL DEFAULT 0,
    duration REAL NOT NULL DEFAULT 0,
    coverage_json TEXT NOT NULL DEFAULT '{}',
    output TEXT NOT NULL DEFAULT '',
    error_kind TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS test_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    execution_id INTEGER NOT NULL REFERENCES test_executions(id) ON DELETE CASCADE,
    test_case_id INTEGER REFERENCES test_cases(id) ON DELETE SET NULL,
    test_id TEXT NOT NULL,
    name TEXT NOT NULL,
    status TEXT NOT NULL,
    message TEXT NOT NULL DEFAULT '',
    traceback TEXT NOT NULL DEFAULT '',
    duration REAL NOT NULL DEFAULT 0,
    source_location TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS ai_analyses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    execution_id INTEGER NOT NULL REFERENCES test_executions(id) ON DELETE CASCADE,
    model TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS corrections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    execution_id INTEGER NOT NULL REFERENCES test_executions(id) ON DELETE CASCADE,
    original_version_id INTEGER NOT NULL REFERENCES code_versions(id),
    proposed_code TEXT NOT NULL,
    explanation TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'approved', 'rejected')),
    applied_version_id INTEGER REFERENCES code_versions(id),
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    decided_at TEXT
);
CREATE TABLE IF NOT EXISTS report_metadata (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    report_type TEXT NOT NULL,
    generated_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS generation_reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    source_sha256 TEXT NOT NULL,
    coverage_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_projects_activity ON projects(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_executions_project ON test_executions(project_id, id DESC);
CREATE INDEX IF NOT EXISTS idx_test_cases_project ON test_cases(project_id, id);
"""


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE_PATH, timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 10000")
    try:
        with connection:
            yield connection
    except sqlite3.Error:
        LOGGER.exception("SQLite operation failed")
        raise
    finally:
        connection.close()


def initialize_database() -> None:
    with connect() as connection:
        connection.executescript(SCHEMA)
        requirement_keys = sorted(
            (row["pk"], row["name"])
            for row in connection.execute("PRAGMA table_info(requirements)")
            if row["pk"]
        )
        if requirement_keys == [(1, "id")]:
            connection.execute(
                """CREATE TABLE requirements_new (
                    id TEXT NOT NULL,
                    project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                    description TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT (datetime('now')),
                    PRIMARY KEY(project_id, id)
                )"""
            )
            connection.execute(
                """INSERT INTO requirements_new(id, project_id, description, created_at)
                SELECT id, project_id, description, created_at FROM requirements"""
            )
            connection.execute("DROP TABLE requirements")
            connection.execute(
                "ALTER TABLE requirements_new RENAME TO requirements"
            )
        project_columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(projects)")
        }
        if "is_pinned" not in project_columns:
            connection.execute(
                "ALTER TABLE projects ADD COLUMN is_pinned INTEGER NOT NULL DEFAULT 0 "
                "CHECK (is_pinned IN (0, 1))"
            )
        connection.execute(
            "INSERT OR IGNORE INTO schema_migrations(version) VALUES (?)", (2,)
        )
        connection.execute(
            "INSERT OR IGNORE INTO schema_migrations(version) VALUES (?)", (3,)
        )
        test_case_columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(test_cases)")
        }
        for column, definition in (
            ("testing_objective", "TEXT NOT NULL DEFAULT ''"),
            ("complexity", "TEXT NOT NULL DEFAULT 'Basic'"),
            ("target_class", "TEXT NOT NULL DEFAULT ''"),
            ("target_kind", "TEXT NOT NULL DEFAULT 'function'"),
            ("keyword_inputs_json", "TEXT NOT NULL DEFAULT '{}'"),
            ("constructor_inputs_json", "TEXT NOT NULL DEFAULT '[]'"),
            ("constructor_keyword_inputs_json", "TEXT NOT NULL DEFAULT '{}'"),
            ("setup_json", "TEXT NOT NULL DEFAULT '[]'"),
            ("database_assertions_json", "TEXT NOT NULL DEFAULT '[]'"),
        ):
            if column not in test_case_columns:
                connection.execute(
                    f"ALTER TABLE test_cases ADD COLUMN {column} {definition}"
                )
        connection.execute(
            "INSERT OR IGNORE INTO schema_migrations(version) VALUES (?)", (4,)
        )
        connection.execute(
            "INSERT OR IGNORE INTO schema_migrations(version) VALUES (?)", (5,)
        )
        if connection.execute(
            "SELECT 1 FROM schema_migrations WHERE version=6"
        ).fetchone() is None:
            connection.execute(
                "UPDATE test_cases SET approval_status='pending'"
            )
            connection.execute(
                "INSERT INTO schema_migrations(version) VALUES (?)", (6,)
            )
        if connection.execute(
            "SELECT 1 FROM schema_migrations WHERE version=7"
        ).fetchone() is None:
            connection.execute(
                """UPDATE test_cases
                SET complexity=CASE
                    WHEN category IN (
                        'empty_input', 'null_value', 'invalid_type', 'boundary',
                        'edge', 'exception', 'conditional', 'combination',
                        'validation', 'failure', 'invalid'
                    ) THEN 'Intermediate'
                    ELSE 'Basic'
                END"""
            )
            connection.execute(
                "INSERT INTO schema_migrations(version) VALUES (?)", (7,)
            )


def row_dict(row: sqlite3.Row | None) -> dict | None:
    return dict(row) if row is not None else None


def rows_dict(rows: list[sqlite3.Row]) -> list[dict]:
    return [dict(row) for row in rows]
