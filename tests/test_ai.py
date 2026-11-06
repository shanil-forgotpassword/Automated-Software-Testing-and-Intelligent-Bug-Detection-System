from __future__ import annotations

import json

import pytest

from codeguard import ai
from codeguard.source_analysis import analyze_source


def test_case_targets_scale_with_inspected_source_complexity() -> None:
    simple = analyze_source(
        "def grade(mark):\n"
        "    return mark\n"
    )
    complex_source = analyze_source(
        "def grade(mark):\n"
        "    if mark < 0:\n"
        "        raise ValueError('too low')\n"
        "    if mark > 100:\n"
        "        raise ValueError('too high')\n"
        "    if mark < 50:\n"
        "        return 'F'\n"
        "    if mark < 75:\n"
        "        return 'C'\n"
        "    if mark < 90:\n"
        "        return 'B'\n"
        "    return 'A'\n"
    )

    simple_targets = ai._category_case_targets(
        ["boundary", "conditional", "exception"], simple
    )
    complex_targets = ai._category_case_targets(
        ["boundary", "conditional", "exception"], complex_source
    )

    assert complex_targets["boundary"] > simple_targets["boundary"]
    assert complex_targets["conditional"] > simple_targets["conditional"]
    assert complex_targets["exception"] > simple_targets["exception"]
    assert all(
        target <= ai.MAX_STANDARD_CASES_PER_CATEGORY
        for category, target in complex_targets.items()
        if category != "boundary"
    )
    assert complex_targets["boundary"] <= ai.MAX_CASES_PER_CATEGORY


def test_generation_batches_pack_intermediate_categories() -> None:
    assert len(ai.CATEGORY_BATCHES) == 5
    assert ai.CATEGORY_BATCHES[3] == (
        "exception", "conditional", "combination", "failure"
    )


def test_source_analysis_promotes_supported_categories_and_boundary_scenarios() -> None:
    inspection = analyze_source(
        "def grade(name, marks):\n"
        "    if not name or not name.strip():\n"
        "        raise ValueError('name required')\n"
        "    for mark in marks:\n"
        "        if mark < 0 or mark > 100:\n"
        "            raise ValueError('mark out of range')\n"
        "    if sum(marks) / len(marks) < 50:\n"
        "        return 'F'\n"
        "    if sum(marks) / len(marks) < 75:\n"
        "        return 'C'\n"
        "    if sum(marks) / len(marks) < 90:\n"
        "        return 'B'\n"
        "    return 'A'\n"
    )
    assessments = {
        category: {
            "category": category,
            "applicable": False,
            "reason": "model assessment",
        }
        for category in ai.GENERATION_CATEGORIES
    }

    planned = ai._apply_source_applicability(assessments, inspection)
    targets = ai._category_case_targets(["boundary"], inspection)

    assert planned["combination"]["applicable"]
    assert planned["conditional"]["applicable"]
    assert planned["boundary"]["applicable"]
    assert "multiple parameters" in planned["combination"]["reason"]
    boundary_scenarios = ai._source_boundary_scenarios(inspection)
    assert targets["boundary"] == len(boundary_scenarios)
    assert boundary_scenarios == [
        49, 50, 74, 75, 89, 90
    ]


def test_boundary_coverage_accumulates_exact_scenarios_across_cases() -> None:
    scenarios: list[int | float] = [49, 50, 74, 75, 89, 90]
    cases = [
        {"category": "boundary", "inputs": ["Avery", [49]], "keyword_inputs": {}},
        {"category": "boundary", "inputs": ["Jordan", [49]], "keyword_inputs": {}},
        {"category": "boundary", "inputs": ["Avery", [50]], "keyword_inputs": {}},
        {"category": "normal", "inputs": ["Avery", [74]], "keyword_inputs": {}},
    ]

    assert ai._covered_boundary_scenarios(cases, scenarios) == {49, 50}


def test_generation_passes_source_boundary_scenarios_to_boundary_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = (
        "def grade(mark):\n"
        "    if mark < 50:\n"
        "        return 'F'\n"
        "    if mark < 75:\n"
        "        return 'C'\n"
        "    return 'A'\n"
    )
    planned_categories = {
        category: {
            "category": category,
            "applicable": category == "boundary",
            "reason": "test",
        }
        for category in ai.GENERATION_CATEGORIES
    }
    captured_scenarios: list[list[int | float] | None] = []

    monkeypatch.setattr(ai, "analyze_requirements", lambda _: {})
    monkeypatch.setattr(
        ai,
        "supported_requirements",
        lambda _: [{"id": "REQ-001", "description": "Grade marks"}],
    )
    monkeypatch.setattr(ai, "_build_context", lambda *args: {})
    monkeypatch.setattr(ai, "_plan_applicable_categories", lambda _: {})
    monkeypatch.setattr(
        ai, "_apply_source_applicability", lambda assessments, _: planned_categories
    )

    def generate_batch(
        batch,
        context,
        applicable_categories,
        inspection,
        requirement_ids,
        *,
        boundary_scenarios=None,
    ):
        if "boundary" in batch:
            captured_scenarios.append(boundary_scenarios)
        return [{"category": "boundary"}], None, 0, 0

    monkeypatch.setattr(ai, "_generate_category_batch", generate_batch)

    ai.generate_test_cases(
        {"title": "Grade calculator"},
        [{"id": "REQ-001", "description": "Grade marks"}],
        source_code=source,
    )

    assert captured_scenarios == [
        ai._source_boundary_scenarios(analyze_source(source))
    ]


def test_case_schema_supports_class_targets_and_requested_batch_size() -> None:
    inspection = analyze_source(
        "class Validator:\n"
        "    def check(self, value):\n"
        "        return value is not None\n"
    )

    schema = ai._make_case_schema(["REQ-001"], inspection, max_items=12)
    test_cases = schema["properties"]["test_cases"]
    item_properties = test_cases["items"]["properties"]

    assert test_cases["maxItems"] == 12
    assert item_properties["target_class"]["enum"] == ["", "Validator"]
    assert item_properties["target_kind"]["enum"] == [
        "function", "instance_method", "class_method", "static_method"
    ]


def test_invalid_type_detection_covers_values_inside_iterated_parameters() -> None:
    inspection = analyze_source(
        "def calculate_grade(name, marks):\n"
        "    if not isinstance(marks, list):\n"
        "        raise TypeError('marks must be a list')\n"
        "    for mark in marks:\n"
        "        if not isinstance(mark, (int, float)):\n"
        "            raise TypeError('marks must be numbers')\n"
    )
    behavior = inspection["functions"][0]
    case = {
        "category": "invalid_type",
        "inputs": ["Avery", ["not-a-number"]],
        "keyword_inputs": {},
    }

    assert behavior["iterable_items"] == {"marks": ["mark"]}
    assert ai._case_matches_intermediate_category(case, behavior)


def test_module_function_validation_normalizes_model_shape_errors() -> None:
    inspection = analyze_source(
        "def increment(value):\n"
        "    return value + 1\n"
    )
    item = {
        "name": "increments value",
        "description": "Checks a module-level function.",
        "testing_objective": "Verify the return value.",
        "requirement_id": "REQ-001",
        "category": "normal",
        "function_name": "increment",
        "target_class": "",
        "target_kind": "function",
        "inputs": [{"name": "value", "value": 1}],
        "keyword_inputs": {},
        "constructor_inputs": [],
        "constructor_keyword_inputs": {},
        "setup": [{"method": "unused", "inputs": [], "keyword_inputs": {}, "store_as": ""}],
        "expected": 2,
        "database_assertions": [],
        "expected_exception": "",
        "preconditions": "",
        "priority": "Medium",
    }

    validated = ai._validate_single_case(
        item,
        ["REQ-001"],
        inspection,
        {
            category: {"applicable": category == "normal"}
            for category in ai.GENERATION_CATEGORIES
        },
    )
    assert validated["inputs"] == [1]
    assert validated["setup"] == []


def test_json_parser_rejects_truncated_responses() -> None:
    assert ai._extract_json_from_text('{"complete": true}') == {"complete": True}

    with pytest.raises(ai.OllamaError, match="incomplete or malformed JSON"):
        ai._extract_json_from_text('{"test_cases":[{"name":"partial"}')


def test_ollama_streaming_request_reports_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeResponse:
        status = 200
        reason = "OK"

        def __init__(self) -> None:
            self.lines = iter([
                b'{"response":"{\\"result\\":","done":false}\n',
                b'{"response":"true}","done":false}\n',
                b'{"response":"","done":true,"done_reason":"stop"}\n',
            ])

        def readline(self, limit: int) -> bytes:
            return next(self.lines, b"")

    class FakeConnection:
        def __init__(self, *args, **kwargs) -> None:
            self.payload = None

        def request(self, method: str, path: str, *, body: bytes, headers: dict) -> None:
            self.payload = json.loads(body)

        def getresponse(self) -> FakeResponse:
            return FakeResponse()

        def close(self) -> None:
            pass

    connection = FakeConnection()
    monkeypatch.setattr(
        ai.http.client, "HTTPConnection", lambda *args, **kwargs: connection
    )
    progress: list[str] = []

    response = ai._request(
        "Return JSON.",
        json_output=True,
        progress_callback=progress.append,
        progress_label="Test generation",
    )

    assert response == '{"result":true}'
    assert connection.payload is not None
    assert connection.payload["stream"] is True
    assert progress[0].endswith("waiting for the local model.")
    assert progress[-1].startswith("Test generation:")


def test_category_batch_keeps_partial_cases_without_a_slow_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inspection = analyze_source("def add(a, b):\n    return a + b\n")
    case = {
        "name": "adds two values",
        "description": "Adds two valid numbers.",
        "testing_objective": "Verify a normal addition.",
        "requirement_id": "REQ-001",
        "category": "normal",
        "function_name": "add",
        "target_class": "",
        "target_kind": "function",
        "inputs": [1, 2],
        "keyword_inputs": {},
        "constructor_inputs": [],
        "constructor_keyword_inputs": {},
        "setup": [],
        "expected": 3,
        "database_assertions": [],
        "expected_exception": "",
        "preconditions": "",
        "priority": "Medium",
    }
    request_count = 0

    def request(*args, **kwargs):
        nonlocal request_count
        request_count += 1
        return json.dumps({"test_cases": [case]})

    monkeypatch.setattr(ai, "_request", request)
    cases, error, rejected, missing = ai._generate_category_batch(
        ("normal",),
        {},
        {"normal": {"applicable": True, "reason": "callable exists"}},
        inspection,
        ["REQ-001"],
    )

    assert request_count == 1
    assert len(cases) == 1
    assert error is not None
    assert rejected == 0
    assert missing == 1
