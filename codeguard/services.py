from __future__ import annotations

import json
import hashlib
import logging
import re
import sqlite3
from collections.abc import Iterable
from typing import Any

from codeguard.database import connect, row_dict, rows_dict
from codeguard.source_analysis import (
    SourceAnalysisError,
    analyze_source,
    validate_case_target,
)
from codeguard.requirement_analysis import analyze_requirements

LOGGER = logging.getLogger(__name__)
FUNCTION_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
TEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,32}$")
ALLOWED_CATEGORIES = {
    "normal", "empty_input", "null_value", "invalid_type", "boundary", "edge",
    "exception", "conditional", "combination", "validation", "failure",
    "invalid", "other",
}
INTERMEDIATE_CATEGORIES = {
    "empty_input", "null_value", "invalid_type", "boundary", "edge",
    "exception", "conditional", "combination", "validation", "failure",
    "invalid",
}
GENERATION_CATEGORIES = {
    "normal", "empty_input", "null_value", "invalid_type", "boundary", "edge",
    "exception", "conditional", "combination", "validation", "failure",
}
ALLOWED_EXCEPTIONS = {
    "", "AssertionError", "IndexError", "KeyError", "TypeError",
    "ValueError", "ZeroDivisionError", "OverflowError", "FileNotFoundError"
}


class ValidationError(ValueError):
    """Raised when user or model data does not meet the application contract."""


def _touch_project(connection: sqlite3.Connection, project_id: int) -> None:
    connection.execute(
        "UPDATE projects SET updated_at=datetime('now') WHERE id=?", (project_id,)
    )


def _inserted_id(cursor: sqlite3.Cursor) -> int:
    if cursor.lastrowid is None:
        raise RuntimeError("SQLite insert did not return a row ID.")
    return cursor.lastrowid


def _requirement_lines(text: str | Iterable[str]) -> list[str]:
    values = text.splitlines() if isinstance(text, str) else list(text)
    descriptions = [str(value).strip() for value in values if str(value).strip()]
    if len(descriptions) > 100:
        raise ValidationError("A project can contain at most 100 requirements.")
    if any(len(value) > 2000 for value in descriptions):
        raise ValidationError("Each requirement must be 2,000 characters or fewer.")
    return descriptions


def create_project(
    *,
    title: str,
    description: str = "",
    objectives: str = "",
    language: str = "Python",
    project_type: str = "",
    main_features: str = "",
    expected_inputs: str = "",
    expected_outputs: str = "",
    constraints: str = "",
    requirements: str | Iterable[str] = (),
    filename: str = "main.py",
    code: str = "",
    code_kind: str = "manual",
    is_demo: bool = False,
) -> int:
    clean_title = title.strip()
    if not clean_title or len(clean_title) > 120:
        raise ValidationError("Project title must contain 1–120 characters.")
    if filename != "main.py" and (
        not filename.endswith(".py") or "/" in filename or "\\" in filename
    ):
        raise ValidationError("Only a simple .py filename is supported.")
    if len(code.encode("utf-8")) > 200_000:
        raise ValidationError("Python source must be 200 KB or smaller.")
    if code:
        try:
            compile(code, filename, "exec")
        except SyntaxError as exc:
            raise ValidationError(
                f"Python syntax error on line {exc.lineno}: {exc.msg}"
            ) from exc
    descriptions = _requirement_lines(requirements)
    try:
        with connect() as connection:
            cursor = connection.execute(
                """INSERT INTO projects
                (title, description, objectives, language, project_type,
                 main_features, expected_inputs, expected_outputs, constraints, is_demo)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    clean_title, description.strip(), objectives.strip(),
                    language.strip() or "Python", project_type.strip(),
                    main_features.strip(), expected_inputs.strip(),
                    expected_outputs.strip(), constraints.strip(), int(is_demo),
                ),
            )
            project_id = _inserted_id(cursor)
            for index, requirement in enumerate(descriptions, start=1):
                connection.execute(
                    "INSERT INTO requirements(id, project_id, description) VALUES (?, ?, ?)",
                    (f"REQ-{index:03d}", project_id, requirement),
                )
            source_cursor = connection.execute(
                """INSERT INTO source_files(project_id, filename, original_content)
                VALUES (?, ?, ?)""",
                (project_id, filename, code),
            )
            connection.execute(
                """INSERT INTO code_versions
                (project_id, source_file_id, version_number, content, kind)
                VALUES (?, ?, 1, ?, ?)""",
                (project_id, _inserted_id(source_cursor), code, code_kind),
            )
            return project_id
    except sqlite3.IntegrityError as exc:
        if "projects.title" in str(exc).lower() or "idx_projects_title_unique" in str(exc):
            raise ValidationError(
                "A project with this title already exists. Choose a different title."
            ) from exc
        LOGGER.exception("Could not create project")
        raise


def get_project(project_id: int) -> dict | None:
    with connect() as connection:
        return row_dict(connection.execute(
            "SELECT * FROM projects WHERE id=?", (project_id,)
        ).fetchone())


def list_projects(*, include_demo: bool = False) -> list[dict]:
    query = "SELECT * FROM projects"
    parameters: tuple = ()
    if not include_demo:
        query += " WHERE is_demo=0"
    query += " ORDER BY is_pinned DESC, updated_at DESC, id DESC"
    with connect() as connection:
        return rows_dict(connection.execute(query, parameters).fetchall())


def rename_project(project_id: int, title: str) -> None:
    clean_title = title.strip()
    if not clean_title or len(clean_title) > 120:
        raise ValidationError("Project title must contain 1–120 characters.")
    try:
        with connect() as connection:
            cursor = connection.execute(
                """UPDATE projects SET title=?, updated_at=datetime('now')
                WHERE id=? AND is_demo=0""",
                (clean_title, project_id),
            )
            if cursor.rowcount != 1:
                exists = connection.execute(
                    "SELECT 1 FROM projects WHERE id=?", (project_id,)
                ).fetchone()
                if exists:
                    raise ValidationError("Demo projects cannot be renamed.")
                raise ValidationError("Project no longer exists.")
    except sqlite3.IntegrityError as exc:
        if "projects.title" in str(exc).lower() or "idx_projects_title_unique" in str(exc):
            raise ValidationError(
                "A project with this title already exists. Choose a different title."
            ) from exc
        LOGGER.exception("Could not rename project %s", project_id)
        raise


def set_project_pinned(project_id: int, pinned: bool) -> None:
    with connect() as connection:
        cursor = connection.execute(
            """UPDATE projects SET is_pinned=?, updated_at=datetime('now')
            WHERE id=? AND is_demo=0""",
            (int(pinned), project_id),
        )
        if cursor.rowcount != 1:
            exists = connection.execute(
                "SELECT 1 FROM projects WHERE id=?", (project_id,)
            ).fetchone()
            if exists:
                raise ValidationError("Demo projects cannot be pinned.")
            raise ValidationError("Project no longer exists.")


def delete_project(project_id: int) -> None:
    with connect() as connection:
        project = connection.execute(
            "SELECT is_demo FROM projects WHERE id=?", (project_id,)
        ).fetchone()
        if project is None:
            raise ValidationError("Project no longer exists.")
        if project["is_demo"]:
            raise ValidationError("Demo projects cannot be deleted.")

        connection.execute("DELETE FROM corrections WHERE project_id=?", (project_id,))
        connection.execute(
            """DELETE FROM ai_analyses
            WHERE execution_id IN (
                SELECT id FROM test_executions WHERE project_id=?
            )""",
            (project_id,),
        )
        connection.execute(
            "DELETE FROM test_executions WHERE project_id=?", (project_id,)
        )
        connection.execute("DELETE FROM projects WHERE id=?", (project_id,))


def list_requirements(project_id: int) -> list[dict]:
    with connect() as connection:
        return rows_dict(connection.execute(
            "SELECT * FROM requirements WHERE project_id=? ORDER BY id", (project_id,)
        ).fetchall())


def add_requirement(project_id: int, description: str) -> str:
    clean = description.strip()
    if not clean or len(clean) > 2000:
        raise ValidationError("Requirement must contain 1–2,000 characters.")
    with connect() as connection:
        if connection.execute(
            "SELECT 1 FROM projects WHERE id=?", (project_id,)
        ).fetchone() is None:
            raise ValidationError("Project no longer exists.")
        count = connection.execute(
            "SELECT COUNT(*) FROM requirements WHERE project_id=?", (project_id,)
        ).fetchone()[0]
        requirement_id = f"REQ-{count + 1:03d}"
        while connection.execute(
            "SELECT 1 FROM requirements WHERE project_id=? AND id=?",
            (project_id, requirement_id),
        ).fetchone():
            count += 1
            requirement_id = f"REQ-{count + 1:03d}"
        connection.execute(
            "INSERT INTO requirements(id, project_id, description) VALUES (?, ?, ?)",
            (requirement_id, project_id, clean),
        )
        _touch_project(connection, project_id)
        return requirement_id


def list_test_cases(project_id: int) -> list[dict]:
    with connect() as connection:
        return rows_dict(connection.execute(
            """SELECT t.*, r.description AS requirement_description
            FROM test_cases t LEFT JOIN requirements r
            ON r.project_id=t.project_id AND r.id=t.requirement_id
            WHERE t.project_id=? ORDER BY t.id""", (project_id,)
        ).fetchall())


def _validate_case_against_project_source(
    connection: sqlite3.Connection,
    project_id: int,
    case: dict[str, Any],
    *,
    required: bool = False,
) -> None:
    source = connection.execute(
        """SELECT content FROM code_versions WHERE project_id=?
        ORDER BY version_number DESC LIMIT 1""",
        (project_id,),
    ).fetchone()
    code = source["content"] if source is not None else ""
    if not code.strip():
        if required:
            raise ValidationError(
                "Add Python source code before approving or executing test cases."
            )
        return
    prepared = dict(case)
    for source_name, target_name in (
        ("inputs_json", "inputs"),
        ("keyword_inputs_json", "keyword_inputs"),
        ("constructor_inputs_json", "constructor_inputs"),
        ("constructor_keyword_inputs_json", "constructor_keyword_inputs"),
        ("setup_json", "setup"),
        ("expected_json", "expected"),
    ):
        if source_name in prepared:
            try:
                prepared[target_name] = json.loads(prepared.pop(source_name))
            except (TypeError, json.JSONDecodeError) as exc:
                raise ValidationError(
                    f"Test {case.get('test_id', '')} has invalid stored {target_name}."
                ) from exc
    try:
        validate_case_target(prepared, analyze_source(code))
    except SourceAnalysisError as exc:
        raise ValidationError(
            f"Test {case.get('test_id', case.get('name', 'case'))} is incompatible "
            f"with the current source: {exc} Edit or regenerate this test before "
            "approving it."
        ) from exc


def _validate_setup_references(
    value: Any, step_count: int, stored_names: set[str], label: str
) -> None:
    if isinstance(value, dict):
        if "$setup_return" in value:
            index = value["$setup_return"]
            if (
                set(value) != {"$setup_return"}
                or not isinstance(index, int)
                or isinstance(index, bool)
                or not 0 <= index < step_count
            ):
                raise ValidationError(f"{label} has an invalid setup return reference.")
            return
        if "$setup_ref" in value:
            if set(value) != {"$setup_ref"} or value["$setup_ref"] not in stored_names:
                raise ValidationError(f"{label} has an unknown setup value reference.")
            return
        if "$actual_return" in value:
            raise ValidationError(f"{label} cannot reference a target return value.")
        for item in value.values():
            _validate_setup_references(item, step_count, stored_names, label)
    elif isinstance(value, list):
        for item in value:
            _validate_setup_references(item, step_count, stored_names, label)


def _validate_test_case(case: dict[str, Any], requirement_ids: set[str]) -> dict[str, Any]:
    required = ("name", "function_name", "inputs", "expected")
    if any(field not in case for field in required):
        raise ValidationError("Test case is missing a required field.")
    name = str(case["name"]).strip()
    function_name = str(case["function_name"]).strip()
    if not name or len(name) > 160:
        raise ValidationError("Test name must contain 1–160 characters.")
    if not FUNCTION_PATTERN.fullmatch(function_name):
        raise ValidationError("Function name must be a valid Python identifier.")
    target_class = str(case.get("target_class") or "").strip()
    if target_class and not FUNCTION_PATTERN.fullmatch(target_class):
        raise ValidationError("Target class must be a valid Python identifier.")
    target_kind = str(
        case.get("target_kind")
        or ("instance_method" if target_class else "function")
    ).strip()
    allowed_target_kinds = {
        "function", "instance_method", "class_method", "static_method"
    }
    if target_kind not in allowed_target_kinds:
        raise ValidationError("Test target kind is not supported.")
    if bool(target_class) != (target_kind != "function"):
        raise ValidationError("Test target class and kind do not match.")
    inputs = case["inputs"]
    if not isinstance(inputs, list) or len(inputs) > 50:
        raise ValidationError("Test inputs must be a JSON array with at most 50 items.")
    keyword_inputs = case.get("keyword_inputs", {})
    if (
        not isinstance(keyword_inputs, dict)
        or len(keyword_inputs) > 50
        or any(not FUNCTION_PATTERN.fullmatch(str(key)) for key in keyword_inputs)
    ):
        raise ValidationError("Test keyword inputs must be an object with valid names.")
    constructor_inputs = case.get("constructor_inputs", [])
    if not isinstance(constructor_inputs, list) or len(constructor_inputs) > 50:
        raise ValidationError("Constructor inputs must be a JSON array with at most 50 items.")
    constructor_keyword_inputs = case.get("constructor_keyword_inputs", {})
    if (
        not isinstance(constructor_keyword_inputs, dict)
        or len(constructor_keyword_inputs) > 50
        or any(
            not FUNCTION_PATTERN.fullmatch(str(key))
            for key in constructor_keyword_inputs
        )
    ):
        raise ValidationError(
            "Constructor keyword inputs must be an object with valid names."
        )
    setup = case.get("setup", [])
    if not isinstance(setup, list) or len(setup) > 10:
        raise ValidationError("At most 10 setup calls are allowed per test.")
    stored_names: set[str] = set()
    for index, action in enumerate(setup):
        if (
            not isinstance(action, dict)
            or not FUNCTION_PATTERN.fullmatch(str(action.get("method", "")))
            or not isinstance(action.get("inputs", []), list)
            or len(action.get("inputs", [])) > 50
            or not isinstance(action.get("keyword_inputs", {}), dict)
            or len(action.get("keyword_inputs", {})) > 50
            or any(
                not FUNCTION_PATTERN.fullmatch(str(key))
                for key in action.get("keyword_inputs", {})
            )
        ):
            raise ValidationError("Test setup calls must have a method and input array.")
        store_as = action.get("store_as", "")
        if not isinstance(store_as, str):
            raise ValidationError("Setup return variable names must be strings.")
        if store_as:
            if not FUNCTION_PATTERN.fullmatch(store_as):
                raise ValidationError("Setup return variable names must be valid identifiers.")
            if store_as in stored_names:
                raise ValidationError("Setup return variable names must be unique.")
            stored_names.add(store_as)
        action["keyword_inputs"] = action.get("keyword_inputs", {})
        action["store_as"] = store_as
        _validate_setup_references(
            [action["inputs"], action["keyword_inputs"]],
            index,
            stored_names - ({store_as} if store_as else set()),
            f"setup step {index + 1}",
        )
    _validate_setup_references(
        [constructor_inputs, constructor_keyword_inputs], 0, set(),
        "constructor arguments",
    )
    _validate_setup_references(
        [inputs, keyword_inputs], len(setup), stored_names, "test arguments"
    )
    database_assertions = case.get("database_assertions", [])
    if not isinstance(database_assertions, list) or len(database_assertions) > 10:
        raise ValidationError("At most 10 database assertions are allowed per test.")
    identifier = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,62}$")

    def valid_assertion_reference(value: Any) -> bool:
        if value == {"$actual_return": True}:
            return True
        if not isinstance(value, dict):
            return False
        if set(value) == {"$setup_return"}:
            index = value["$setup_return"]
            return (
                isinstance(index, int)
                and not isinstance(index, bool)
                and 0 <= index < len(setup)
            )
        if set(value) == {"$setup_ref"}:
            return value["$setup_ref"] in stored_names
        return False

    for assertion in database_assertions:
        if (
            not isinstance(assertion, dict)
            or not identifier.fullmatch(str(assertion.get("table", "")))
            or not isinstance(assertion.get("where"), dict)
            or not isinstance(assertion.get("values"), dict)
            or not assertion["where"]
            or not assertion["values"]
            or len(assertion["where"]) > 10
            or len(assertion["values"]) > 20
            or any(not identifier.fullmatch(str(key)) for key in (
                *assertion["where"], *assertion["values"]
            ))
        ):
            raise ValidationError(
                "Database assertions must contain a table and safe column names."
            )
        if any(
            isinstance(value, dict) and not valid_assertion_reference(value)
            for value in assertion["where"].values()
        ):
            raise ValidationError(
                "Database assertion values may only reference the actual return "
                "with {\"$actual_return\": true}."
            )
        if any(
            isinstance(value, (dict, list))
            for value in (*assertion["where"].values(), *assertion["values"].values())
            if not valid_assertion_reference(value)
        ):
            raise ValidationError(
                "Database assertions must compare SQLite scalar values."
            )
    expected = case["expected"]
    if isinstance(expected, dict) and "$type" in expected:
        supported_types = {
            "integer", "number", "string", "boolean", "array", "object", "none",
            "row",
        }
        numeric_contract = expected.get("$type") in {"integer", "number"}
        non_empty_contract = expected.get("$type") in {"string", "array", "object"}
        if (
            expected.get("$type") not in supported_types
            or set(expected) - {"$type", "minimum", "maximum", "non_empty", "values"}
            or ("minimum" in expected and (
                not numeric_contract or not isinstance(expected["minimum"], (int, float))
            ))
            or ("maximum" in expected and (
                not numeric_contract or not isinstance(expected["maximum"], (int, float))
            ))
            or ("non_empty" in expected and (
                not non_empty_contract or not isinstance(expected["non_empty"], bool)
            ))
            or (
                "minimum" in expected and "maximum" in expected
                and expected["minimum"] > expected["maximum"]
            )
            or (
                "values" in expected and (
                    expected.get("$type") != "row"
                    or not isinstance(expected["values"], dict)
                )
            )
        ):
            raise ValidationError("Expected output type contract is invalid.")
    try:
        inputs_json = json.dumps(inputs, ensure_ascii=False, allow_nan=False)
        keyword_inputs_json = json.dumps(
            keyword_inputs, ensure_ascii=False, allow_nan=False
        )
        constructor_inputs_json = json.dumps(
            constructor_inputs, ensure_ascii=False, allow_nan=False
        )
        constructor_keyword_inputs_json = json.dumps(
            constructor_keyword_inputs, ensure_ascii=False, allow_nan=False
        )
        setup_json = json.dumps(setup, ensure_ascii=False, allow_nan=False)
        expected_json = json.dumps(case["expected"], ensure_ascii=False, allow_nan=False)
        database_assertions_json = json.dumps(
            database_assertions, ensure_ascii=False, allow_nan=False
        )
    except (TypeError, ValueError) as exc:
        raise ValidationError("Inputs and expected output must be JSON values.") from exc
    if any(len(value.encode("utf-8")) > 20_000 for value in (
        inputs_json, keyword_inputs_json, constructor_inputs_json,
        constructor_keyword_inputs_json, setup_json, expected_json,
        database_assertions_json
    )):
        raise ValidationError("Test inputs and expected output are limited to 20 KB each.")
    requirement_id = str(case.get("requirement_id") or "").strip() or None
    if requirement_id and requirement_id not in requirement_ids:
        raise ValidationError("Test case refers to a requirement that does not exist.")
    category = str(case.get("category", "normal")).strip().lower()
    if category not in ALLOWED_CATEGORIES:
        raise ValidationError("Test case category is not supported.")
    complexity = (
        "Intermediate" if category in INTERMEDIATE_CATEGORIES else "Basic"
    )
    supplied_complexity = case.get("complexity")
    if supplied_complexity and supplied_complexity != complexity:
        raise ValidationError(
            "Test complexity must match its actual test category and input objective."
        )
    testing_objective = str(
        case.get("testing_objective", case.get("description", ""))
    ).strip()
    if len(testing_objective) > 1000:
        raise ValidationError("Testing objective must be 1,000 characters or fewer.")
    if case.get("generation_source") == "ollama" and not testing_objective:
        raise ValidationError("Generated test cases must include a testing objective.")
    exception = str(case.get("expected_exception") or "").strip()
    if exception not in ALLOWED_EXCEPTIONS:
        raise ValidationError("Expected exception is not in the supported allowlist.")
    priority = str(case.get("priority", "Medium")).strip().title()
    if priority not in {"Low", "Medium", "High", "Critical"}:
        raise ValidationError("Priority must be Low, Medium, High, or Critical.")
    return {
        "name": name,
        "description": str(case.get("description", "")).strip()[:2000],
        "testing_objective": testing_objective,
        "complexity": complexity,
        "function_name": function_name,
        "target_class": target_class,
        "target_kind": target_kind,
        "inputs_json": inputs_json,
        "keyword_inputs_json": keyword_inputs_json,
        "constructor_inputs_json": constructor_inputs_json,
        "constructor_keyword_inputs_json": constructor_keyword_inputs_json,
        "setup_json": setup_json,
        "expected_json": expected_json,
        "database_assertions_json": database_assertions_json,
        "requirement_id": requirement_id,
        "category": category,
        "expected_exception": exception,
        "preconditions": str(case.get("preconditions", "")).strip()[:1000],
        "priority": priority,
        "generation_source": str(case.get("generation_source", "manual"))[:40],
    }


def add_test_case(project_id: int, case: dict[str, Any]) -> str:
    with connect() as connection:
        requirement_ids = {
            row[0] for row in connection.execute(
                "SELECT id FROM requirements WHERE project_id=?", (project_id,)
            ).fetchall()
        }
        validated = _validate_test_case(case, requirement_ids)
        approval_status = str(case.get("approval_status", "pending")).strip().lower()
        if approval_status not in {"pending", "approved", "rejected"}:
            approval_status = "pending"
        _validate_case_against_project_source(
            connection,
            project_id,
            case,
            required=approval_status == "approved",
        )
        count = connection.execute(
            "SELECT COUNT(*) FROM test_cases WHERE project_id=?", (project_id,)
        ).fetchone()[0]
        test_id = f"TC-{count + 1:03d}"
        while connection.execute(
            "SELECT 1 FROM test_cases WHERE project_id=? AND test_id=?",
            (project_id, test_id),
        ).fetchone():
            count += 1
            test_id = f"TC-{count + 1:03d}"
        connection.execute(
            """INSERT INTO test_cases
            (project_id, test_id, name, description, testing_objective, complexity,
             requirement_id, category,
             function_name, target_class, target_kind, inputs_json, keyword_inputs_json,
             constructor_inputs_json, constructor_keyword_inputs_json, setup_json,
             expected_json, database_assertions_json,
             expected_exception, preconditions, priority, generation_source, approval_status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                project_id, test_id, validated["name"], validated["description"],
                validated["testing_objective"], validated["complexity"],
                validated["requirement_id"], validated["category"],
                validated["function_name"], validated["target_class"],
                validated["target_kind"], validated["inputs_json"],
                validated["keyword_inputs_json"],
                validated["constructor_inputs_json"],
                validated["constructor_keyword_inputs_json"], validated["setup_json"],
                validated["expected_json"],
                validated["database_assertions_json"], validated["expected_exception"],
                validated["preconditions"], validated["priority"],
                validated["generation_source"], approval_status,
            ),
        )
        _touch_project(connection, project_id)
        return test_id


def _normalize_generation_assessments(
    category_assessments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not isinstance(category_assessments, list):
        raise ValidationError("Generation category coverage must be a list.")
    by_category: dict[str, dict[str, Any]] = {}
    for assessment in category_assessments:
        if (
            not isinstance(assessment, dict)
            or assessment.get("category") not in GENERATION_CATEGORIES
            or not isinstance(assessment.get("applicable"), bool)
            or not isinstance(assessment.get("reason"), str)
            or not assessment["reason"].strip()
            or assessment["category"] in by_category
        ):
            raise ValidationError("Generation category coverage is invalid.")
        by_category[assessment["category"]] = {
            "category": assessment["category"],
            "applicable": assessment["applicable"],
            "reason": assessment["reason"].strip()[:1000],
        }
    if set(by_category) != GENERATION_CATEGORIES:
        raise ValidationError("Generation category coverage is incomplete.")
    return list(by_category.values())


def add_generated_test_cases(
    project_id: int,
    cases: list[dict[str, Any]],
    *,
    source_code: str | None = None,
    category_assessments: list[dict[str, Any]] | None = None,
) -> list[str]:
    if not cases or len(cases) > 200:
        raise ValidationError("The generator must return between 1 and 200 test cases.")
    if (source_code is None) != (category_assessments is None):
        raise ValidationError(
            "Source and category assessments must be supplied together."
        )
    normalized_assessments = (
        _normalize_generation_assessments(category_assessments)
        if category_assessments is not None
        else None
    )
    with connect() as connection:
        requirement_ids = {
            row[0] for row in connection.execute(
                "SELECT id FROM requirements WHERE project_id=?", (project_id,)
            ).fetchall()
        }
        validated = []
        for case in cases:
            source_case = dict(case)
            source_case["generation_source"] = "ollama"
            _validate_case_against_project_source(
                connection, project_id, source_case, required=True
            )
            validated.append(_validate_test_case(source_case, requirement_ids))
        count = connection.execute(
            "SELECT COUNT(*) FROM test_cases WHERE project_id=?", (project_id,)
        ).fetchone()[0]
        identifiers = []
        for item in validated:
            count += 1
            test_id = f"TC-{count:03d}"
            while connection.execute(
                "SELECT 1 FROM test_cases WHERE project_id=? AND test_id=?",
                (project_id, test_id),
            ).fetchone():
                count += 1
                test_id = f"TC-{count:03d}"
            connection.execute(
                """INSERT INTO test_cases
                (project_id, test_id, name, description, testing_objective, complexity,
                 requirement_id, category,
                 function_name, target_class, target_kind, inputs_json, keyword_inputs_json,
                 constructor_inputs_json, constructor_keyword_inputs_json, setup_json,
                 expected_json, database_assertions_json,
                 expected_exception, preconditions, priority, generation_source, approval_status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'approved')""",
                (
                    project_id, test_id, item["name"], item["description"],
                    item["testing_objective"], item["complexity"],
                    item["requirement_id"], item["category"], item["function_name"],
                    item["target_class"], item["target_kind"], item["inputs_json"],
                    item["keyword_inputs_json"], item["constructor_inputs_json"],
                    item["constructor_keyword_inputs_json"], item["setup_json"],
                    item["expected_json"],
                    item["database_assertions_json"],
                    item["expected_exception"], item["preconditions"], item["priority"],
                    item["generation_source"],
                ),
            )
            identifiers.append(test_id)
        if normalized_assessments is not None and source_code is not None:
            connection.execute(
                """INSERT INTO generation_reports
                (project_id, source_sha256, coverage_json) VALUES (?, ?, ?)""",
                (
                    project_id,
                    hashlib.sha256(source_code.encode("utf-8")).hexdigest(),
                    json.dumps(
                        normalized_assessments,
                        ensure_ascii=False,
                        allow_nan=False,
                    ),
                ),
            )
        _touch_project(connection, project_id)
        return identifiers


def update_test_case(
    project_id: int, case_id: int, case: dict[str, Any], *, status: str | None = None
) -> None:
    with connect() as connection:
        requirement_ids = {
            row[0] for row in connection.execute(
                "SELECT id FROM requirements WHERE project_id=?", (project_id,)
            ).fetchall()
        }
        validated = _validate_test_case(case, requirement_ids)
        if status is not None and status not in {"pending", "approved", "rejected"}:
            raise ValidationError("Invalid approval status.")
        _validate_case_against_project_source(
            connection,
            project_id,
            case,
            required=status == "approved",
        )
        cursor = connection.execute(
            """UPDATE test_cases SET name=?, description=?, requirement_id=?,
            category=?, function_name=?, target_class=?, target_kind=?,
            inputs_json=?, keyword_inputs_json=?, constructor_inputs_json=?,
            constructor_keyword_inputs_json=?, setup_json=?, expected_json=?,
            database_assertions_json=?, expected_exception=?,
            preconditions=?, priority=?,
            approval_status=COALESCE(?, approval_status), updated_at=datetime('now')
            WHERE project_id=? AND id=?""",
            (
                validated["name"], validated["description"], validated["requirement_id"],
                validated["category"], validated["function_name"],
                validated["target_class"], validated["target_kind"],
                validated["inputs_json"], validated["keyword_inputs_json"],
                validated["constructor_inputs_json"],
                validated["constructor_keyword_inputs_json"],
                validated["setup_json"], validated["expected_json"],
                validated["database_assertions_json"],
                validated["expected_exception"], validated["preconditions"],
                validated["priority"], status, project_id, case_id,
            ),
        )
        if cursor.rowcount != 1:
            raise ValidationError("Test case no longer exists in this project.")
        _touch_project(connection, project_id)


def set_test_status(project_id: int, case_id: int, status: str) -> None:
    if status not in {"pending", "approved", "rejected"}:
        raise ValidationError("Invalid approval status.")
    with connect() as connection:
        if status == "approved":
            case = connection.execute(
                "SELECT * FROM test_cases WHERE project_id=? AND id=?",
                (project_id, case_id),
            ).fetchone()
            if case is None:
                raise ValidationError("Test case no longer exists.")
            _validate_case_against_project_source(
                connection, project_id, dict(case), required=True
            )
        cursor = connection.execute(
            """UPDATE test_cases SET approval_status=?, updated_at=datetime('now')
            WHERE project_id=? AND id=?""", (status, project_id, case_id)
        )
        if cursor.rowcount != 1:
            raise ValidationError("Test case no longer exists.")
        _touch_project(connection, project_id)


def delete_test_case(project_id: int, case_id: int) -> None:
    with connect() as connection:
        cursor = connection.execute(
            "DELETE FROM test_cases WHERE project_id=? AND id=?", (project_id, case_id)
        )
        if cursor.rowcount != 1:
            raise ValidationError("Test case no longer exists in this project.")
        _touch_project(connection, project_id)


def list_code_versions(project_id: int) -> list[dict]:
    with connect() as connection:
        return rows_dict(connection.execute(
            "SELECT * FROM code_versions WHERE project_id=? ORDER BY version_number DESC",
            (project_id,),
        ).fetchall())


def get_code_version(version_id: int) -> dict | None:
    with connect() as connection:
        return row_dict(connection.execute(
            "SELECT * FROM code_versions WHERE id=?", (version_id,)
        ).fetchone())


def save_code_version(project_id: int, content: str, kind: str) -> int:
    if len(content.encode("utf-8")) > 200_000:
        raise ValidationError("Python source must be 200 KB or smaller.")
    try:
        compile(content, "main.py", "exec")
    except SyntaxError as exc:
        raise ValidationError(
            f"Python syntax error on line {exc.lineno}: {exc.msg}"
        ) from exc
    with connect() as connection:
        source = connection.execute(
            "SELECT id FROM source_files WHERE project_id=? ORDER BY id LIMIT 1",
            (project_id,),
        ).fetchone()
        if source is None:
            raise ValidationError("Project source file is missing.")
        number = connection.execute(
            "SELECT COALESCE(MAX(version_number), 0) + 1 FROM code_versions WHERE project_id=?",
            (project_id,),
        ).fetchone()[0]
        cursor = connection.execute(
            """INSERT INTO code_versions(project_id, source_file_id, version_number, content, kind)
            VALUES (?, ?, ?, ?, ?)""",
            (project_id, source["id"], number, content, kind[:40]),
        )
        _touch_project(connection, project_id)
        return _inserted_id(cursor)


def get_original_code(project_id: int) -> dict | None:
    with connect() as connection:
        return row_dict(connection.execute(
            """SELECT s.filename, s.original_content FROM source_files s
            WHERE s.project_id=? ORDER BY s.id LIMIT 1""", (project_id,)
        ).fetchone())


def save_execution(project_id: int, version_id: int, result: dict[str, Any]) -> int:
    coverage = json.dumps(result.get("coverage") or {}, ensure_ascii=False)
    with connect() as connection:
        version = connection.execute(
            "SELECT 1 FROM code_versions WHERE id=? AND project_id=?",
            (version_id, project_id),
        ).fetchone()
        if version is None:
            raise ValidationError("Execution code version does not belong to this project.")
        case_ids = {
            item.get("test_case_id")
            for item in result.get("results", [])
            if item.get("test_case_id") is not None
        }
        if case_ids:
            placeholders = ",".join("?" for _ in case_ids)
            owned_case_ids = {
                row["id"]
                for row in connection.execute(
                    f"""SELECT id FROM test_cases
                    WHERE project_id=? AND id IN ({placeholders})""",
                    (project_id, *case_ids),
                ).fetchall()
            }
            if owned_case_ids != case_ids:
                raise ValidationError(
                    "Execution results contain a test case from another project."
                )
        cursor = connection.execute(
            """INSERT INTO test_executions
            (project_id, code_version_id, status, total, passed, failed, skipped,
             errors, duration, coverage_json, output, error_kind)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                project_id, version_id, result.get("status", "infrastructure_error"),
                int(result.get("total", 0)), int(result.get("passed", 0)),
                int(result.get("failed", 0)), int(result.get("skipped", 0)),
                int(result.get("errors", 0)), float(result.get("duration", 0)),
                coverage, str(result.get("output", ""))[:40000],
                str(result.get("error_kind", ""))[:80],
            ),
        )
        execution_id = _inserted_id(cursor)
        for item in result.get("results", []):
            message = str(item.get("message", ""))[:4000]
            output_details = []
            if item.get("expected_output") is not None:
                output_details.append(
                    f"Expected output: {item['expected_output']}"
                )
            if item.get("actual_output") is not None:
                output_details.append(f"Actual output: {item['actual_output']}")
            if output_details:
                details = "\n".join(output_details)
                message = f"{message}\n{details}" if message else details
            connection.execute(
                """INSERT INTO test_results
                (execution_id, test_case_id, test_id, name, status, message,
                 traceback, duration, source_location)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    execution_id, item.get("test_case_id"), item.get("test_id", ""),
                    item.get("name", ""), item.get("status", "error"),
                    message[:4000], item.get("traceback", "")[:10000],
                    float(item.get("duration", 0)), item.get("source_location", "")[:500],
                ),
            )
        _touch_project(connection, project_id)
        return execution_id


def list_executions(project_id: int) -> list[dict]:
    with connect() as connection:
        rows = connection.execute(
            """SELECT e.*, v.version_number, v.kind AS version_kind
            FROM test_executions e JOIN code_versions v ON v.id=e.code_version_id
            WHERE e.project_id=? ORDER BY e.id DESC""", (project_id,)
        ).fetchall()
        return rows_dict(rows)


def get_execution(execution_id: int) -> dict | None:
    with connect() as connection:
        execution = row_dict(connection.execute(
            """SELECT e.*, v.version_number, v.content AS code_content
            FROM test_executions e JOIN code_versions v ON v.id=e.code_version_id
            WHERE e.id=?""", (execution_id,)
        ).fetchone())
        if execution:
            execution["coverage"] = json.loads(execution.pop("coverage_json") or "{}")
            execution["results"] = rows_dict(connection.execute(
                "SELECT * FROM test_results WHERE execution_id=? ORDER BY id",
                (execution_id,),
            ).fetchall())
        return execution


def save_analysis(execution_id: int, model: str, content: str) -> None:
    with connect() as connection:
        execution = connection.execute(
            "SELECT project_id FROM test_executions WHERE id=?", (execution_id,)
        ).fetchone()
        if execution is None:
            raise ValidationError("Analysis execution no longer exists.")
        connection.execute(
            "INSERT INTO ai_analyses(execution_id, model, content) VALUES (?, ?, ?)",
            (execution_id, model, content),
        )
        _touch_project(connection, execution["project_id"])


def list_analyses(project_id: int) -> list[dict]:
    with connect() as connection:
        return rows_dict(connection.execute(
            """SELECT a.*, e.created_at AS execution_created_at
            FROM ai_analyses a JOIN test_executions e ON e.id=a.execution_id
            WHERE e.project_id=? ORDER BY a.id DESC""", (project_id,)
        ).fetchall())


def create_correction(
    project_id: int, execution_id: int, version_id: int,
    proposed_code: str, explanation: str,
) -> int:
    if len(proposed_code.encode("utf-8")) > 200_000:
        raise ValidationError("Suggested source must be 200 KB or smaller.")
    try:
        compile(proposed_code, "main.py", "exec")
    except SyntaxError as exc:
        raise ValidationError(
            f"AI suggestion has invalid Python syntax on line {exc.lineno}: {exc.msg}"
        ) from exc
    with connect() as connection:
        execution = connection.execute(
            """SELECT project_id, code_version_id FROM test_executions
            WHERE id=?""",
            (execution_id,),
        ).fetchone()
        version = connection.execute(
            "SELECT project_id FROM code_versions WHERE id=?", (version_id,)
        ).fetchone()
        if execution is None or execution["project_id"] != project_id:
            raise ValidationError("Correction execution and source must belong to the same project.")
        if version is None or version["project_id"] != project_id:
            raise ValidationError("Correction source version does not belong to this project.")
        if execution["code_version_id"] != version_id:
            raise ValidationError(
                "Correction source version must match the execution source version."
            )
        cursor = connection.execute(
            """INSERT INTO corrections
            (project_id, execution_id, original_version_id, proposed_code, explanation)
            VALUES (?, ?, ?, ?, ?)""",
            (project_id, execution_id, version_id, proposed_code, explanation[:4000]),
        )
        _touch_project(connection, project_id)
        return _inserted_id(cursor)


def list_corrections(project_id: int) -> list[dict]:
    with connect() as connection:
        return rows_dict(connection.execute(
            """SELECT c.*, v.version_number AS original_version_number
            FROM corrections c JOIN code_versions v ON v.id=c.original_version_id
            WHERE c.project_id=? ORDER BY c.id DESC""", (project_id,)
        ).fetchall())


def decide_correction(correction_id: int, approve: bool) -> int | None:
    with connect() as connection:
        correction = connection.execute(
            "SELECT * FROM corrections WHERE id=?", (correction_id,)
        ).fetchone()
        if correction is None or correction["status"] != "pending":
            raise ValidationError("This correction is no longer pending review.")
        if not approve:
            connection.execute(
                """UPDATE corrections SET status='rejected', decided_at=datetime('now')
                WHERE id=?""", (correction_id,)
            )
            _touch_project(connection, correction["project_id"])
            return None
        latest_version = connection.execute(
            """SELECT id FROM code_versions WHERE project_id=?
            ORDER BY version_number DESC LIMIT 1""",
            (correction["project_id"],),
        ).fetchone()
        if (
            latest_version is None
            or latest_version["id"] != correction["original_version_id"]
        ):
            raise ValidationError(
                "This correction was based on an older code version than the newer "
                "code version now saved. Generate a new "
                "suggestion for the current version before approving it."
            )
        content = correction["proposed_code"]
        try:
            compile(content, "main.py", "exec")
        except SyntaxError as exc:
            raise ValidationError(
                f"Suggested code is invalid on line {exc.lineno}: {exc.msg}"
            ) from exc
        source_id = connection.execute(
            "SELECT id FROM source_files WHERE project_id=? ORDER BY id LIMIT 1",
            (correction["project_id"],),
        ).fetchone()["id"]
        number = connection.execute(
            "SELECT COALESCE(MAX(version_number), 0) + 1 FROM code_versions WHERE project_id=?",
            (correction["project_id"],),
        ).fetchone()[0]
        version_cursor = connection.execute(
            """INSERT INTO code_versions(project_id, source_file_id, version_number, content, kind)
            VALUES (?, ?, ?, ?, 'ai-correction')""",
            (correction["project_id"], source_id, number, content),
        )
        version_id = _inserted_id(version_cursor)
        connection.execute(
            """UPDATE corrections SET status='approved', applied_version_id=?,
            decided_at=datetime('now') WHERE id=?""", (version_id, correction_id)
        )
        _touch_project(connection, correction["project_id"])
        return version_id


def add_report_metadata(project_id: int, report_type: str) -> None:
    with connect() as connection:
        connection.execute(
            "INSERT INTO report_metadata(project_id, report_type) VALUES (?, ?)",
            (project_id, report_type),
        )
        _touch_project(connection, project_id)


def list_report_metadata(project_id: int) -> list[dict]:
    with connect() as connection:
        return rows_dict(connection.execute(
            "SELECT * FROM report_metadata WHERE project_id=? ORDER BY id DESC",
            (project_id,),
        ).fetchall())


def project_report_data(project_id: int) -> dict[str, Any]:
    project = get_project(project_id)
    if project is None:
        raise ValidationError("Project no longer exists.")
    tests = list_test_cases(project_id)
    executions = list_executions(project_id)
    for execution in executions:
        full = get_execution(execution["id"])
        execution["coverage"] = full["coverage"] if full else {}
        execution["results"] = full["results"] if full else []
    requirements = list_requirements(project_id)
    versions = list_code_versions(project_id)
    latest_source = versions[0]["content"] if versions else ""
    return {
        "project": project,
        "requirements": requirements,
        "requirement_analysis": analyze_requirements(requirements),
        "generation_report": get_generation_report(project_id, latest_source),
        "test_cases": tests,
        "versions": versions,
        "executions": executions,
        "analyses": list_analyses(project_id),
        "corrections": list_corrections(project_id),
        "reports": list_report_metadata(project_id),
    }


def save_generation_report(
    project_id: int,
    source_code: str,
    category_assessments: list[dict[str, Any]],
) -> int:
    normalized_assessments = _normalize_generation_assessments(
        category_assessments
    )
    coverage_json = json.dumps(
        normalized_assessments, ensure_ascii=False, allow_nan=False
    )
    source_sha256 = hashlib.sha256(source_code.encode("utf-8")).hexdigest()
    with connect() as connection:
        if not connection.execute(
            "SELECT 1 FROM projects WHERE id=?", (project_id,)
        ).fetchone():
            raise ValidationError("Project no longer exists.")
        cursor = connection.execute(
            """INSERT INTO generation_reports
            (project_id, source_sha256, coverage_json) VALUES (?, ?, ?)""",
            (project_id, source_sha256, coverage_json),
        )
        return _inserted_id(cursor)


def get_generation_report(
    project_id: int, source_code: str | None = None
) -> dict[str, Any] | None:
    with connect() as connection:
        row = connection.execute(
            """SELECT * FROM generation_reports
            WHERE project_id=? ORDER BY id DESC LIMIT 1""",
            (project_id,),
        ).fetchone()
    report = row_dict(row)
    if report is None:
        return None
    if source_code is not None:
        source_sha256 = hashlib.sha256(source_code.encode("utf-8")).hexdigest()
        if report["source_sha256"] != source_sha256:
            return None
    try:
        report["category_assessments"] = json.loads(report.pop("coverage_json"))
    except json.JSONDecodeError as exc:
        LOGGER.exception("Stored generation report %s has invalid JSON", report["id"])
        raise ValidationError("Saved generation category coverage is corrupted.") from exc
    return report


def requirement_coverage(project_id: int) -> dict[str, Any]:
    requirements = list_requirements(project_id)
    tests = list_test_cases(project_id)
    executions = list_executions(project_id)
    latest = get_execution(executions[0]["id"]) if executions else None
    result_by_test = {
        result["test_id"]: result for result in (latest or {}).get("results", [])
    }
    mapping = []
    for requirement in requirements:
        linked_tests = [
            test for test in tests if test["requirement_id"] == requirement["id"]
        ]
        mapping.append({
            "requirement_id": requirement["id"],
            "description": requirement["description"],
            "tests": [
                {
                    "test_id": test["test_id"],
                    "name": test["name"],
                    "approval_status": test["approval_status"],
                    "execution_status": result_by_test.get(test["test_id"], {}).get(
                        "status", "not_run"
                    ),
                }
                for test in linked_tests
            ],
        })
    linked = sum(bool(item["tests"]) for item in mapping)
    return {
        "total": len(requirements),
        "linked": linked,
        "uncovered": len(requirements) - linked,
        "mapping": mapping,
    }


def compare_executions(before_id: int, after_id: int) -> dict[str, Any]:
    before = get_execution(before_id)
    after = get_execution(after_id)
    if before is None or after is None:
        raise ValidationError("Both executions must exist to compare them.")
    before_results = {item["test_id"]: item for item in before["results"]}
    after_results = {item["test_id"]: item for item in after["results"]}
    newly_added = set(after_results) - set(before_results)
    return {
        "fixed": [
            key for key in before_results.keys() & after_results.keys()
            if before_results[key]["status"] == "failed"
            and after_results[key]["status"] == "passed"
        ],
        "still_failing": [
            key for key in before_results.keys() & after_results.keys()
            if after_results[key]["status"] in {"failed", "error"}
        ],
        "regressions": [
            key for key in before_results.keys() & after_results.keys()
            if before_results[key]["status"] == "passed"
            and after_results[key]["status"] in {"failed", "error"}
        ],
        "new_failures": [
            key for key in newly_added
            if after_results[key]["status"] in {"failed", "error"}
        ],
        "before": before,
        "after": after,
    }
