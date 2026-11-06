from __future__ import annotations

import hashlib
import csv
import io
import json
import subprocess
import sys
from pathlib import Path

import pytest

from codeguard import database, reports, services
from codeguard import execution
from codeguard.execution import SandboxError, _test_payload


@pytest.fixture
def report_data(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DATA_DIR", tmp_path)
    monkeypatch.setattr(database, "DATABASE_PATH", tmp_path / "reports.sqlite3")
    database.initialize_database()
    project_id = services.create_project(
        title="Report sample",
        requirements=["Returns a sum"],
        code="def add(a, b):\n    return a + b\n",
    )
    services.add_test_case(project_id, {
        "name": "Addition", "requirement_id": "REQ-001", "category": "normal",
        "function_name": "add", "inputs": [2, 2], "expected": 4,
    })
    return services.project_report_data(project_id)


def test_only_approved_tests_can_be_prepared(report_data):
    with pytest.raises(services.ValidationError, match="Only approved"):
        _test_payload(report_data["test_cases"])


def test_project_report_includes_complexity_and_advanced_scope(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DATA_DIR", tmp_path)
    monkeypatch.setattr(database, "DATABASE_PATH", tmp_path / "reports.sqlite3")
    database.initialize_database()
    project_id = services.create_project(
        title="Complexity report",
        requirements=[
            "Returns a calculated result.",
            "Reject invalid input.",
            "Prevent race conditions under concurrent requests.",
        ],
    )

    data = services.project_report_data(project_id)
    exported = json.loads(reports.json_export(data))
    pdf = reports.pdf_report(data)

    assert data["requirement_analysis"]["basic"] == 1
    assert data["requirement_analysis"]["intermediate"] == 1
    assert data["requirement_analysis"]["advanced"] == 1
    assert data["requirement_analysis"]["advanced_requirements"][0]["id"] == "REQ-003"
    assert "Advanced LLM integration is not currently available" in (
        exported["requirement_analysis"]["advanced_limitation_warning"]
    )
    assert exported["requirement_analysis"]["advanced_requirements"][0]["reason"]
    assert pdf.startswith(b"%PDF")


def test_approved_execution_payload_includes_class_and_database_setup(report_data):
    project_id = services.create_project(
        title="Calculator class",
        requirements=["Adds two values"],
        code=(
            "class Calculator:\n"
            "    def __init__(self, database=':memory:'):\n"
            "        self.database = database\n"
            "    def add(self, left, right):\n"
            "        return left + right\n"
        ),
    )
    services.add_test_case(project_id, {
        "name": "Addition",
        "requirement_id": "REQ-001",
        "function_name": "add",
        "target_class": "Calculator",
        "target_kind": "instance_method",
        "inputs": [2, 2],
        "constructor_inputs": ["__CODEGUARD_TEMP_DB__"],
        "expected": 4,
        "database_assertions": [],
    })
    case = services.list_test_cases(project_id)[0]
    services.set_test_status(project_id, case["id"], "approved")
    case = services.list_test_cases(project_id)[0]
    payload = execution._test_payload([case])
    assert payload[0]["target_class"] == "Calculator"
    assert payload[0]["constructor_inputs"] == ["__CODEGUARD_TEMP_DB__"]
    assert payload[0]["database_assertions"] == []


def test_execution_refuses_pending_case_even_when_called_directly(report_data):
    case = report_data["test_cases"][0]
    with pytest.raises(services.ValidationError, match="Only approved"):
        execution.run_approved_tests(
            "def add(a, b):\n    return a + b\n", [case]
        )


def test_approved_run_passes_only_approved_cases_to_sandbox(monkeypatch, report_data):
    project_id = report_data["project"]["id"]
    case = services.list_test_cases(project_id)[0]
    services.set_test_status(project_id, case["id"], "approved")
    case = services.list_test_cases(project_id)[0]
    monkeypatch.setattr(execution, "_ensure_image", lambda: None)
    captured = {}
    response = {
        "protocol": 1,
        "status": "completed",
        "total": 1,
        "passed": 1,
        "failed": 0,
        "errors": 0,
        "skipped": 0,
        "coverage": {"available": False},
        "output": "1 passed",
        "results": [{"test_id": case["test_id"], "status": "passed"}],
    }

    def fake_run(command, **kwargs):
        mount = next(item for item in command if item.startswith("type=bind,source="))
        cases_path = Path(mount.split("source=", 1)[1].split(",target=", 1)[0])
        captured["cases"] = json.loads((cases_path / "cases.json").read_text())
        return subprocess.CompletedProcess(command, 0, json.dumps(response), "")

    monkeypatch.setattr(execution.subprocess, "run", fake_run)
    result = execution.run_approved_tests(
        "def add(a, b):\n    return a + b\n", [case]
    )

    assert captured["cases"][0]["test_case_id"] == case["id"]
    assert result["results"][0]["test_case_id"] == case["id"]


def test_stored_approved_case_is_revalidated_against_current_source(
    monkeypatch, report_data
):
    project_id = report_data["project"]["id"]
    case = services.list_test_cases(project_id)[0]
    with database.connect() as connection:
        connection.execute(
            """UPDATE test_cases SET function_name='register_customer',
            target_class='', target_kind='function', approval_status='approved'
            WHERE project_id=? AND id=?""",
            (project_id, case["id"]),
        )
    saved_case = services.list_test_cases(project_id)[0]

    def docker_must_not_be_reached():
        pytest.fail("stale approved test reached Docker")

    monkeypatch.setattr(execution, "_ensure_image", docker_must_not_be_reached)
    with pytest.raises(services.ValidationError, match="register_customer"):
        execution.run_approved_tests(
            "def add(a, b):\n    return a + b\n", [saved_case]
        )


def test_docker_payload_preserves_validated_keyword_and_setup_arguments(
    monkeypatch, report_data
):
    project_id = services.create_project(
        title="Order call payload",
        requirements=["Creates an order"],
        code=(
            "class InventorySystem:\n"
            "    def add_customer(self, name):\n"
            "        return len(name)\n"
            "    def create_order(self, customer_id, products, discount=0):\n"
            "        return customer_id\n"
        ),
    )
    services.add_test_case(project_id, {
        "name": "Create order with setup customer",
        "requirement_id": "REQ-001",
        "function_name": "create_order",
        "target_class": "InventorySystem",
        "target_kind": "instance_method",
        "inputs": [{"$setup_ref": "customer_id"}, [[1, 1]]],
        "keyword_inputs": {"discount": 0},
        "constructor_inputs": [],
        "constructor_keyword_inputs": {},
        "setup": [{
            "method": "add_customer",
            "inputs": ["Rahul"],
            "keyword_inputs": {},
            "store_as": "customer_id",
        }],
        "expected": 5,
    })
    case = services.list_test_cases(project_id)[0]
    services.set_test_status(project_id, case["id"], "approved")
    case = services.list_test_cases(project_id)[0]
    monkeypatch.setattr(execution, "_ensure_image", lambda: None)
    captured = {}
    response = {
        "protocol": 1,
        "status": "completed",
        "total": 1,
        "passed": 1,
        "failed": 0,
        "errors": 0,
        "skipped": 0,
        "coverage": {"available": False},
        "output": "1 passed",
        "results": [{"test_id": case["test_id"], "status": "passed"}],
    }

    def fake_run(command, **kwargs):
        mount = next(item for item in command if item.startswith("type=bind,source="))
        cases_path = Path(mount.split("source=", 1)[1].split(",target=", 1)[0])
        captured["case"] = json.loads(
            (cases_path / "cases.json").read_text(encoding="utf-8")
        )[0]
        return subprocess.CompletedProcess(command, 0, json.dumps(response), "")

    monkeypatch.setattr(execution.subprocess, "run", fake_run)
    execution.run_approved_tests(
        services.list_code_versions(project_id)[0]["content"], [case]
    )

    assert captured["case"]["inputs"] == [
        {"$setup_ref": "customer_id"}, [[1, 1]]
    ]
    assert captured["case"]["keyword_inputs"] == {"discount": 0}
    assert captured["case"]["setup"][0]["store_as"] == "customer_id"


def test_sandbox_image_rebuilds_when_runner_changes(tmp_path, monkeypatch):
    sandbox_dir = tmp_path / "sandbox"
    sandbox_dir.mkdir()
    dockerfile = sandbox_dir / "Dockerfile"
    runner = sandbox_dir / "runner.py"
    dockerfile.write_text("FROM python:3.11-slim\n", encoding="utf-8")
    runner.write_text("print('updated runner')\n", encoding="utf-8")
    monkeypatch.setattr(execution, "PROJECT_ROOT", tmp_path)
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if command[1:3] == ["image", "inspect"]:
            return subprocess.CompletedProcess(command, 0, "old-fingerprint\n", "")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(execution.subprocess, "run", fake_run)
    execution._ensure_image()

    expected_fingerprint = hashlib.sha256(
        dockerfile.read_bytes() + runner.read_bytes()
    ).hexdigest()
    assert len(calls) == 2
    assert "--label" in calls[1]
    assert f"codeguard.runner-sha256={expected_fingerprint}" in calls[1]


def test_report_exports_have_expected_content(report_data):
    assert json.loads(reports.json_export(report_data))["project"]["title"] == "Report sample"
    assert b"test_id" in reports.tests_csv(report_data)
    assert b"execution_id" in reports.executions_csv(report_data)
    assert reports.pdf_report(report_data).startswith(b"%PDF")


def test_csv_exports_escape_formula_like_user_content(report_data):
    test_data = {
        **report_data,
        "test_cases": [
            {**report_data["test_cases"][0], "name": "=HYPERLINK(\"https://example.invalid\")"}
        ],
        "executions": [{
            "id": 1,
            "created_at": "2026-01-01",
            "version_number": 1,
            "status": "completed",
            "results": [{
                "test_id": "TC-001",
                "name": "+SUM(1,1)",
                "status": "failed",
                "message": "@SUM(1,1)",
                "traceback": "-1",
                "duration": 0.1,
            }],
        }],
    }

    tests_rows = list(csv.DictReader(io.StringIO(
        reports.tests_csv(test_data).decode("utf-8-sig")
    )))
    execution_rows = list(csv.DictReader(io.StringIO(
        reports.executions_csv(test_data).decode("utf-8-sig")
    )))

    assert tests_rows[0]["name"].startswith("'=")
    assert execution_rows[0]["test_name"].startswith("'+")
    assert execution_rows[0]["message"].startswith("'@")
    assert execution_rows[0]["traceback"].startswith("'-")


def test_pdf_contains_stored_before_and_after_runs(report_data):
    project_id = report_data["project"]["id"]
    case = services.list_test_cases(project_id)[0]
    version_id = services.list_code_versions(project_id)[0]["id"]
    common = {
        "test_case_id": case["id"],
        "test_id": case["test_id"],
        "name": case["name"],
        "message": "",
    }
    services.save_execution(project_id, version_id, {
        "status": "completed", "total": 1, "failed": 1,
        "results": [{**common, "status": "failed", "message": "expected 4, got 3"}],
    })
    services.save_execution(project_id, version_id, {
        "status": "completed", "total": 1, "passed": 1,
        "results": [{**common, "status": "passed"}],
    })
    updated = services.project_report_data(project_id)
    pdf = reports.pdf_report(updated)
    assert pdf.startswith(b"%PDF")
    comparison = reports.execution_comparisons(updated)[0]
    assert comparison["fixed"] == [case["test_id"]]
    assert comparison["still_failing"] == []


def test_docker_failure_never_falls_back_to_host_execution(tmp_path, monkeypatch):
    marker = tmp_path / "host-execution-marker"

    def fail_docker():
        raise SandboxError("Docker is unavailable.")

    monkeypatch.setattr(execution, "_ensure_image", fail_docker)
    code = (
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('executed')\n"
        "def solve():\n"
        "    return 1\n"
    )
    case = {
        "id": 1,
        "test_id": "TC-001",
        "name": "Test",
        "function_name": "solve",
        "inputs_json": "[]",
        "expected_json": "1",
        "expected_exception": "",
        "approval_status": "approved",
    }
    with pytest.raises(SandboxError, match="Docker is unavailable"):
        execution.run_approved_tests(code, [case])
    assert not marker.exists()


def test_docker_output_uses_tolerant_utf8_decoding(monkeypatch):
    monkeypatch.setattr(execution, "_ensure_image", lambda: None)
    real_run = subprocess.run
    captured = {}
    sandbox_response = {
        "protocol": 1,
        "status": "completed",
        "total": 1,
        "passed": 1,
        "failed": 0,
        "errors": 0,
        "skipped": 0,
        "coverage": {"available": False},
        "output": "1 passed",
        "results": [{"test_id": "TC-001", "status": "passed"}],
    }

    def fake_run(command, **kwargs):
        captured.update(kwargs)
        diagnostic = real_run(
            [
                sys.executable,
                "-c",
                "import sys; sys.stderr.buffer.write(bytes([129]))",
            ],
            capture_output=True,
            text=True,
            encoding=kwargs["encoding"],
            errors=kwargs["errors"],
            check=False,
        )
        return subprocess.CompletedProcess(
            command, 0, json.dumps(sandbox_response), diagnostic.stderr
        )

    monkeypatch.setattr(execution.subprocess, "run", fake_run)
    case = {
        "id": 1,
        "test_id": "TC-001",
        "name": "Addition",
        "function_name": "add",
        "inputs_json": "[2, 3]",
        "expected_json": "5",
        "expected_exception": "",
        "approval_status": "approved",
    }
    result = execution.run_approved_tests(
        "def add(a, b):\n    return a + b\n", [case]
    )

    assert captured["encoding"] == "utf-8"
    assert captured["errors"] == "replace"
    assert "\ufffd" in result["output"]
