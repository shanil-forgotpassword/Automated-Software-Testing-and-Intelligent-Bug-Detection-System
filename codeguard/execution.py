from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from codeguard.database import PROJECT_ROOT
from codeguard.services import ValidationError
from codeguard.source_analysis import (
    SourceAnalysisError,
    analyze_source,
    validate_case_target,
)

LOGGER = logging.getLogger(__name__)
IMAGE = "codeguard-sandbox:latest"
TIMEOUT_SECONDS = 30


class SandboxError(RuntimeError):
    """Raised when Docker cannot safely execute a test run."""


def _ensure_image() -> None:
    dockerfile = PROJECT_ROOT / "sandbox" / "Dockerfile"
    runner = dockerfile.with_name("runner.py")
    fingerprint = hashlib.sha256(
        dockerfile.read_bytes() + runner.read_bytes()
    ).hexdigest()
    probe = subprocess.run(
        [
            "docker", "image", "inspect",
            "--format={{ index .Config.Labels \"codeguard.runner-sha256\" }}",
            IMAGE,
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=20,
        check=False,
    )
    if probe.returncode == 0 and probe.stdout.strip() == fingerprint:
        return
    build = subprocess.run(
        [
            "docker", "build", "--tag", IMAGE,
            "--label", f"codeguard.runner-sha256={fingerprint}",
            "--file", str(dockerfile), str(dockerfile.parent),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
        check=False,
    )
    if build.returncode != 0:
        detail = (build.stderr or build.stdout).strip()[-3000:]
        raise SandboxError(
            "Could not build the CodeGuard sandbox image. Check that Docker Desktop "
            "is running and the image can be built. "
            f"{detail or 'Docker returned no details.'}"
        )


def _test_payload(test_cases: list[dict]) -> list[dict]:
    if not test_cases:
        raise ValidationError("Add at least one test case before running tests.")
    if len(test_cases) > 200:
        raise ValidationError("A single test run is limited to 200 test cases.")
    payload = []
    for case in test_cases:
        if case["approval_status"] != "approved":
            raise ValidationError("Only approved test cases may be executed.")
        try:
            inputs = json.loads(case["inputs_json"])
            keyword_inputs = json.loads(case.get("keyword_inputs_json", "{}"))
            constructor_inputs = json.loads(
                case.get("constructor_inputs_json", "[]")
            )
            constructor_keyword_inputs = json.loads(
                case.get("constructor_keyword_inputs_json", "{}")
            )
            setup = json.loads(case.get("setup_json", "[]"))
            expected = json.loads(case["expected_json"])
            database_assertions = json.loads(
                case.get("database_assertions_json", "[]")
            )
        except json.JSONDecodeError as exc:
            raise ValidationError(
                f"Test {case['test_id']} has invalid stored JSON."
            ) from exc
        if (
            not isinstance(inputs, list)
            or not isinstance(keyword_inputs, dict)
            or not isinstance(constructor_inputs, list)
            or not isinstance(constructor_keyword_inputs, dict)
            or not isinstance(setup, list)
            or not isinstance(database_assertions, list)
        ):
            raise ValidationError(
                f"Test {case['test_id']} has invalid target or setup data."
            )
        payload.append({
            "test_id": case["test_id"],
            "test_case_id": case["id"],
            "name": case["name"],
            "function_name": case["function_name"],
            "target_class": case.get("target_class", ""),
            "target_kind": case.get("target_kind", "function"),
            "inputs": inputs,
            "keyword_inputs": keyword_inputs,
            "constructor_inputs": constructor_inputs,
            "constructor_keyword_inputs": constructor_keyword_inputs,
            "setup": setup,
            "expected": expected,
            "database_assertions": database_assertions,
            "expected_exception": case["expected_exception"],
        })
    return payload


def _run_tests(
    code: str, test_cases: list[dict]
) -> dict[str, Any]:
    if not code.strip():
        raise ValidationError("Add Python code before running tests.")
    try:
        compile(code, "main.py", "exec")
    except SyntaxError as exc:
        raise ValidationError(
            f"Python syntax error on line {exc.lineno}: {exc.msg}"
        ) from exc
    if len(code.encode("utf-8")) > 200_000:
        raise ValidationError("Python source must be 200 KB or smaller.")
    payload = _test_payload(test_cases)
    try:
        inspection = analyze_source(code)
    except SourceAnalysisError as exc:
        raise ValidationError(str(exc)) from exc
    for item in payload:
        try:
            validate_case_target(item, inspection)
        except SourceAnalysisError as exc:
            raise ValidationError(
                f"Test {item['test_id']} is incompatible with the current source: "
                f"{exc} Edit or regenerate this test before execution."
            ) from exc
    started = time.monotonic()
    try:
        _ensure_image()
    except FileNotFoundError as exc:
        raise SandboxError(
            "Docker was not found. Install Docker Desktop and start it; host execution "
            "is intentionally disabled."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise SandboxError("Docker did not respond in time.") from exc

    with tempfile.TemporaryDirectory(prefix="codeguard-") as directory:
        workdir = Path(directory)
        (workdir / "solution.py").write_text(code, encoding="utf-8")
        (workdir / "cases.json").write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )
        command = [
            "docker", "run", "--rm",
            "--network", "none",
            "--cpus=1",
            "--memory=256m",
            "--memory-swap=256m",
            "--pids-limit=64",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges:true",
            "--read-only",
            "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m",
            "--user", "65534:65534",
            "--env", "PYTHONDONTWRITEBYTECODE=1",
            "--env", "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1",
            "--mount", f"type=bind,source={workdir},target=/workspace,readonly",
            "--workdir", "/workspace",
            "--entrypoint", "python",
            IMAGE, "/opt/codeguard/runner.py",
        ]
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=TIMEOUT_SECONDS + 15,
                check=False,
            )
        except FileNotFoundError as exc:
            raise SandboxError(
                "Docker was not found. Install Docker Desktop and start it; host "
                "execution is intentionally disabled."
            ) from exc
        except subprocess.TimeoutExpired:
            return {
                "status": "timeout",
                "total": 0,
                "passed": 0,
                "failed": 0,
                "skipped": 0,
                "errors": 0,
                "duration": time.monotonic() - started,
                "coverage": {"available": False},
                "output": f"Container exceeded the {TIMEOUT_SECONDS}-second execution limit.",
                "error_kind": "timeout",
                "results": [],
            }
        duration = time.monotonic() - started
        stdout = completed.stdout.strip()
        try:
            result = json.loads(stdout.splitlines()[-1])
        except (IndexError, json.JSONDecodeError) as exc:
            detail = (completed.stderr or stdout).strip()[-3000:]
            raise SandboxError(
                "Docker could not complete the test run. "
                f"{detail or f'Container exited with status {completed.returncode}.'}"
            ) from exc
        if not isinstance(result, dict) or result.get("protocol") != 1:
            raise SandboxError("The sandbox returned an unsupported result format.")
        result["duration"] = duration
        result["output"] = (
            str(result.get("output", "")) + (
                f"\nDocker diagnostics:\n{completed.stderr[-3000:]}"
                if completed.stderr else ""
            )
        )[:40000]
        result["results"] = [
            {
                **item,
                "test_case_id": next(
                    (case["id"] for case in test_cases
                     if case["test_id"] == item.get("test_id")), None
                ),
            }
            for item in result.get("results", [])
        ]
        if completed.returncode not in (0, 1, 2):
            raise SandboxError(
                f"Docker container exited unexpectedly with status {completed.returncode}."
            )
        if result.get("status") == "infrastructure_error":
            result["error_kind"] = result.get("error_kind") or "pytest"
        LOGGER.info(
            "Sandbox execution completed: %s tests, status=%s",
            result.get("total", 0), result.get("status"),
        )
        return result


def run_approved_tests(code: str, test_cases: list[dict]) -> dict[str, Any]:
    return _run_tests(code, test_cases)
