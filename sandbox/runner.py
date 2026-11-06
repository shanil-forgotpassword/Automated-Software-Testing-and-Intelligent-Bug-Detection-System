from __future__ import annotations

import builtins
import io
import json
import re
import sqlite3
import sys
import time
import traceback
import xml.etree.ElementTree as ET
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

WORKSPACE = Path("/workspace")
RESULTS_PATH = Path("/tmp/pytest-results.xml")
CASE_PATTERN = re.compile(r"\[([A-Za-z0-9_-]{1,32})\]$")
EXCEPTIONS = {
    name for name, value in vars(builtins).items()
    if isinstance(value, type) and issubclass(value, BaseException)
}
IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,62}")


def _extract_location(text: str) -> str:
    for line in text.splitlines():
        if "solution.py" in line and "line " in line:
            return line.strip()[:500]
    return ""


def _read_results(cases: list[dict], output: str) -> list[dict]:
    case_map = {case["test_id"]: case for case in cases}
    if not RESULTS_PATH.exists():
        return []
    root = ET.parse(RESULTS_PATH).getroot()
    results = []
    for node in root.iter("testcase"):
        match = CASE_PATTERN.search(node.attrib.get("name", ""))
        test_id = match.group(1) if match else ""
        case = case_map.get(test_id, {})
        failure = node.find("failure")
        error = node.find("error")
        skipped = node.find("skipped")
        detail = failure if failure is not None else error
        failure_type = detail.attrib.get("type", "") if detail is not None else ""
        status = "passed"
        if failure is not None:
            status = (
                "failed"
                if failure_type in {"", "AssertionError", "pytest.fail.Exception"}
                else "error"
            )
        elif error is not None:
            status = "error"
        elif skipped is not None:
            status = "skipped"
        message = ""
        trace = ""
        if detail is not None:
            message = detail.attrib.get("message", "")[:4000]
            trace = (detail.text or "")[:10000]
        elif skipped is not None:
            message = skipped.attrib.get("message", "")[:4000]
        properties_node = node.find("properties")
        properties = {
            item.attrib.get("name", ""): item.attrib.get("value", "")
            for item in properties_node or ()
        }
        results.append({
            "test_id": test_id,
            "name": case.get("name", node.attrib.get("name", "")),
            "status": status,
            "message": message,
            "expected_output": properties.get("expected_output"),
            "actual_output": properties.get("actual_output"),
            "traceback": trace,
            "duration": float(node.attrib.get("time", 0)),
            "source_location": _extract_location(trace),
        })
    return results


def _generated_test_source(cases: list[dict]) -> str:
    case_json = json.dumps(cases, ensure_ascii=False)
    template = r'''
import builtins, importlib.util, json, os, pytest, re, sqlite3, sys
from pathlib import Path

sys.path.insert(0, "/workspace")
CASES = json.loads(__CASE_JSON__)
IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,62}")

def _load_solution(case):
    module_name = "codeguard_solution_" + re.sub(
        r"[^A-Za-z0-9_]", "_", case["test_id"]
    )
    spec = importlib.util.spec_from_file_location(
        module_name, "/workspace/solution.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load uploaded solution.py in the sandbox.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module

def _resolve_values(obj, setup_returns=None, setup_values=None, actual=None):
    if isinstance(obj, dict):
        if obj == {"$actual_return": True}:
            return actual
        if "$setup_return" in obj and len(obj) == 1:
            idx = obj["$setup_return"]
            if isinstance(idx, int) and not isinstance(idx, bool) and 0 <= idx < len(setup_returns or []):
                return setup_returns[idx]
            raise ValueError("Invalid setup return reference.")
        if "$setup_ref" in obj and len(obj) == 1:
            name = obj["$setup_ref"]
            if setup_values is not None and name in setup_values:
                return setup_values[name]
            raise ValueError("Unknown setup return variable: " + str(name))
        return {
            k: _resolve_values(v, setup_returns, setup_values, actual)
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [
            _resolve_values(item, setup_returns, setup_values, actual)
            for item in obj
        ]
    if isinstance(obj, tuple):
        return tuple(
            _resolve_values(item, setup_returns, setup_values, actual)
            for item in obj
        )
    return obj

def _call_arguments(func, args, kwargs, label):
    import inspect
    try:
        inspect.signature(func).bind(*args, **kwargs)
    except (TypeError, ValueError) as exc:
        raise ValueError("{} arguments do not match: {}".format(label, exc)) from exc

def _invoke(solution, case, temp_dir):
    target_class = case.get("target_class", "")
    target_kind = case.get("target_kind", "function")
    function_name = case["function_name"]
    import inspect

    instance = None
    setup_returns = []
    setup_values = {}

    if target_class:
        cls = getattr(solution, target_class, None)
        if not inspect.isclass(cls):
            raise AttributeError(
                "module {!r} has no target class {!r}".format(
                    solution.__name__, target_class
                )
            )
    elif target_kind == "function":
        func = getattr(solution, function_name, None)
        if not callable(func):
            raise AttributeError(
                "module {!r} has no module-level function {!r}".format(
                    solution.__name__, function_name
                )
            )
        args = case.get("inputs", [])
        kwargs = case.get("keyword_inputs", {})
        _call_arguments(func, args, kwargs, function_name)
        args = _resolve_values(args, setup_returns, setup_values)
        kwargs = _resolve_values(kwargs, setup_returns, setup_values)
        return func(*args, **kwargs), None, setup_returns, setup_values
    else:
        raise ValueError("A target class is required for class methods.")

    if target_kind == "instance_method":
        constructor_inputs = [
            str(temp_dir / "codeguard.sqlite3")
            if value == "__CODEGUARD_TEMP_DB__" else value
            for value in case.get("constructor_inputs", [])
        ]
        constructor_kwargs = {
            key: (
                str(temp_dir / "codeguard.sqlite3")
                if value == "__CODEGUARD_TEMP_DB__" else value
            )
            for key, value in case.get("constructor_keyword_inputs", {}).items()
        }
        _call_arguments(cls, constructor_inputs, constructor_kwargs, target_class + ".__init__")
        instance = cls(*constructor_inputs, **constructor_kwargs)

        for step in case.get("setup", []):
            step_func = getattr(instance, step["method"], None)
            if not callable(step_func):
                raise AttributeError(
                    "{} has no callable setup method {!r}".format(
                        target_class, step["method"]
                    )
                )
            step_inputs = _resolve_values(
                step.get("inputs", []), setup_returns, setup_values
            )
            step_keywords = _resolve_values(
                step.get("keyword_inputs", {}), setup_returns, setup_values
            )
            _call_arguments(
                step_func, step_inputs, step_keywords,
                target_class + "." + step["method"],
            )
            step_result = step_func(*step_inputs, **step_keywords)
            setup_returns.append(step_result)
            if step.get("store_as"):
                setup_values[step["store_as"]] = step_result

        func = getattr(instance, function_name)
    elif target_kind in {"class_method", "static_method"}:
        for step in case.get("setup", []):
            step_func = getattr(cls, step["method"], None)
            if not callable(step_func):
                raise AttributeError(
                    "{} has no callable setup method {!r}".format(
                        target_class, step["method"]
                    )
                )
            step_inputs = _resolve_values(
                step.get("inputs", []), setup_returns, setup_values
            )
            step_keywords = _resolve_values(
                step.get("keyword_inputs", {}), setup_returns, setup_values
            )
            _call_arguments(
                step_func, step_inputs, step_keywords,
                target_class + "." + step["method"],
            )
            step_result = step_func(*step_inputs, **step_keywords)
            setup_returns.append(step_result)
            if step.get("store_as"):
                setup_values[step["store_as"]] = step_result
        func = getattr(cls, function_name)
    else:
        raise RuntimeError("Unsupported callable target kind: " + target_kind)

    call_inputs = _resolve_values(case.get("inputs", []), setup_returns, setup_values)
    call_keywords = _resolve_values(
        case.get("keyword_inputs", {}), setup_returns, setup_values
    )
    _call_arguments(func, call_inputs, call_keywords, target_class + "." + function_name)
    return func(*call_inputs, **call_keywords), instance, setup_returns, setup_values

def _assert_expected(actual, expected):
    if isinstance(expected, dict) and list(expected.keys()) == ["expected"]:
        expected = expected["expected"]
    if not isinstance(expected, dict) or "$type" not in expected:
        assert actual == expected, "expected {!r}, got {!r}".format(expected, actual)
        return
    expected_type = expected["$type"]
    predicates = {
        "integer": lambda value: isinstance(value, int) and not isinstance(value, bool),
        "number": lambda value: isinstance(value, (int, float)) and not isinstance(value, bool),
        "string": lambda value: isinstance(value, str),
        "boolean": lambda value: isinstance(value, bool),
        "array": lambda value: isinstance(value, list),
        "object": lambda value: isinstance(value, dict),
        "none": lambda value: value is None,
        "row": lambda value: value is not None and (isinstance(value, tuple) or hasattr(value, "keys")),
    }
    if expected_type not in predicates:
        raise AssertionError("Unsupported expected-output contract: " + str(expected_type))
    assert predicates[expected_type](actual), (
        "expected type {}, got {!r}".format(expected_type, actual)
    )
    if "minimum" in expected:
        assert actual >= expected["minimum"], (
            "expected value >= {!r}, got {!r}".format(expected["minimum"], actual)
        )
    if "maximum" in expected:
        assert actual <= expected["maximum"], (
            "expected value <= {!r}, got {!r}".format(expected["maximum"], actual)
        )
    if expected.get("non_empty"):
        assert len(actual) > 0, "expected a non-empty {}".format(expected_type)

def _database_paths(root):
    return sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".db", ".sqlite", ".sqlite3"}
    )

def _assert_database_state(assertions, root, actual, setup_returns=None, setup_values=None):
    setup_returns = setup_returns or []
    setup_values = setup_values or {}
    for assertion in assertions:
        table = assertion["table"]
        where = assertion["where"]
        values = assertion["values"]
        if not IDENTIFIER.fullmatch(table) or any(
            not IDENTIFIER.fullmatch(column)
            for column in (*where.keys(), *values.keys())
        ):
            raise RuntimeError("Database assertion contains an unsafe identifier.")
        conditions = []
        parameters = []
        for column, value in where.items():
            parameters.append(_resolve_values(value, setup_returns, setup_values, actual))
            conditions.append('"{}" = ?'.format(column))
        columns = list(values)
        query = 'SELECT {} FROM "{}" WHERE {} LIMIT 1'.format(
            ", ".join('"{}"'.format(column) for column in columns),
            table,
            " AND ".join(conditions),
        )
        found_table = False
        failures = []
        for path in _database_paths(root):
            try:
                connection = sqlite3.connect(
                    "file:{}?mode=ro".format(path.as_posix()), uri=True
                )
                connection.row_factory = sqlite3.Row
                try:
                    table_exists = connection.execute(
                        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                        (table,),
                    ).fetchone()
                    if not table_exists:
                        continue
                    found_table = True
                    row = connection.execute(query, parameters).fetchone()
                    if row is None:
                        failures.append("No matching row in table {!r}.".format(table))
                        continue
                    for column, expected in values.items():
                        expected_val = _resolve_values(
                            expected, setup_returns, setup_values, actual
                        )
                        assert row[column] == expected_val, (
                            "database column {} expected {!r}, got {!r}".format(
                                column, expected_val, row[column]
                            )
                        )
                    break
                finally:
                    connection.close()
            except sqlite3.DatabaseError as exc:
                failures.append("{}: {}".format(path.name, exc))
        else:
            if not found_table:
                raise AssertionError(
                    "No isolated SQLite database contained table {!r}.".format(table)
                )
            raise AssertionError("; ".join(failures) or
                                 "Database state assertion failed for {!r}.".format(table))

@pytest.mark.parametrize("case", CASES, ids=[item["test_id"] for item in CASES])
def test_contract(case, record_property, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    solution = _load_solution(case)
    instance = None
    try:
        expected_exception = case.get("expected_exception", "")
        if expected_exception:
            record_property("expected_output", "raises " + expected_exception)
            error_type = getattr(builtins, expected_exception)
            with pytest.raises(error_type) as caught:
                actual, instance, setup_returns, setup_values = _invoke(solution, case, tmp_path)
            actual_str = "raised {}: {}".format(
                type(caught.value).__name__, caught.value
            )
            record_property("actual_output", actual_str)
        else:
            expected = case["expected"]
            record_property("expected_output", repr(expected))
            actual, instance, setup_returns, setup_values = _invoke(solution, case, tmp_path)
            record_property("actual_output", repr(actual))
            _assert_database_state(
                case.get("database_assertions", []), tmp_path, actual,
                setup_returns, setup_values,
            )
            _assert_expected(actual, expected)
    finally:
        if instance is not None:
            close_method = getattr(instance, "close", None)
            if callable(close_method):
                try:
                    close_method()
                except Exception:
                    pass
            conn = getattr(instance, "conn", None) or getattr(instance, "connection", None)
            if conn is not None and hasattr(conn, "close") and callable(conn.close):
                try:
                    conn.close()
                except Exception:
                    pass
        sys.modules.pop(solution.__name__, None)
'''
    return template.replace("__CASE_JSON__", repr(case_json))


def main() -> int:
    try:
        cases = json.loads((WORKSPACE / "cases.json").read_text(encoding="utf-8"))
        if not isinstance(cases, list) or not cases:
            raise ValueError("No tests were provided.")
        for case in cases:
            if (
                not isinstance(case, dict)
                or not re.fullmatch(r"[A-Za-z0-9_-]{1,32}", case.get("test_id", ""))
                or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", case.get("function_name", ""))
                or not re.fullmatch(
                    r"[A-Za-z_][A-Za-z0-9_]{0,63}",
                    case.get("target_class", "") or "_",
                )
                or case.get("target_kind", "function") not in {
                    "function", "instance_method", "class_method", "static_method"
                }
                or not isinstance(case.get("inputs"), list)
                or not isinstance(case.get("keyword_inputs", {}), dict)
                or not isinstance(case.get("constructor_inputs", []), list)
                or not isinstance(case.get("constructor_keyword_inputs", {}), dict)
                or not isinstance(case.get("setup", []), list)
                or not isinstance(case.get("database_assertions", []), list)
                or (
                    not case.get("target_class", "")
                    and case.get("target_kind", "function") != "function"
                )
            ):
                raise ValueError("A test case did not pass sandbox validation.")
            exception_name = case.get("expected_exception", "")
            if exception_name and exception_name not in EXCEPTIONS:
                raise ValueError("An expected exception is not supported.")
        generated = _generated_test_source(cases)
        generated_path = Path("/tmp/test_generated.py")
        generated_path.write_text(generated, encoding="utf-8")
        capture = io.StringIO()
        started = time.monotonic()
        coverage_error = ""
        try:
            import coverage
            cov = coverage.Coverage(
                include=[str((WORKSPACE / "solution.py").resolve())],
                data_file="/tmp/.coverage",
            )
            cov.erase()
            cov.start()
        except Exception as exc:
            cov = None
            coverage_error = f"{type(exc).__name__}: {exc}"
        try:
            import pytest
            with redirect_stdout(capture), redirect_stderr(capture):
                exit_code = int(pytest.main([
                    "-q", "--disable-warnings", "-p", "no:cacheprovider",
                    f"--junitxml={RESULTS_PATH}", str(generated_path),
                ]))
        finally:
            if cov is not None:
                cov.stop()
                cov.save()
        output = capture.getvalue()[-30000:]
        results = _read_results(cases, output)
        if not results:
            raise RuntimeError(
                "Pytest collected no tests or could not write results. " + output[-2000:]
            )
        coverage_info = {"available": False, "error": coverage_error}
        if cov is not None:
            try:
                target_path = str((WORKSPACE / "solution.py").resolve())
                measured = {
                    str(Path(path).resolve())
                    for path in cov.get_data().measured_files()
                }
                if target_path not in measured or not cov.get_data().lines(target_path):
                    raise RuntimeError(
                        "No coverage data was collected for /workspace/solution.py."
                    )
                report_path = Path("/tmp/coverage.json")
                cov.json_report(outfile=str(report_path))
                report = json.loads(report_path.read_text(encoding="utf-8"))
                file_data = next(
                    (value for key, value in report.get("files", {}).items()
                     if str(Path(key).resolve()) == target_path),
                    {},
                )
                if not file_data or not file_data.get("summary", {}).get("num_statements"):
                    raise RuntimeError(
                        "Coverage.py did not report executable lines for the solution module."
                    )
                coverage_info = {
                    "available": True,
                    "percentage": float(file_data.get("summary", {}).get("percent_covered", 0)),
                    "covered_lines": int(file_data.get("summary", {}).get("covered_lines", 0)),
                    "missed_lines": int(file_data.get("summary", {}).get("missing_lines", 0)),
                    "uncovered_lines": file_data.get("missing_lines", []),
                }
            except Exception as exc:
                coverage_info = {
                    "available": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
        passed = sum(item["status"] == "passed" for item in results)
        failed = sum(item["status"] == "failed" for item in results)
        errors = sum(item["status"] == "error" for item in results)
        skipped = sum(item["status"] == "skipped" for item in results)
        status = "completed" if exit_code in (0, 1) else "infrastructure_error"
        response = {
            "protocol": 1,
            "status": status,
            "total": len(results),
            "passed": passed,
            "failed": failed,
            "errors": errors,
            "skipped": skipped,
            "duration": time.monotonic() - started,
            "coverage": coverage_info,
            "output": output,
            "error_kind": "pytest" if status == "infrastructure_error" else "",
            "results": results,
        }
        print(json.dumps(response, ensure_ascii=False))
        return 0
    except Exception as exc:
        response = {
            "protocol": 1,
            "status": "infrastructure_error",
            "total": 0,
            "passed": 0,
            "failed": 0,
            "errors": 0,
            "skipped": 0,
            "duration": 0,
            "coverage": {"available": False},
            "output": f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=5)}",
            "error_kind": "runner",
            "results": [],
        }
        print(json.dumps(response, ensure_ascii=False))
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
