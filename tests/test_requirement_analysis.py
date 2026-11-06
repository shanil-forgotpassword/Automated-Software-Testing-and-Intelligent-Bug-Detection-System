from __future__ import annotations

from codeguard.requirement_analysis import (
    analyze_requirements,
    remove_duplicate_cases,
    supported_requirements,
)


def test_requirement_complexity_is_explainable_and_splits_supported_scope():
    analysis = analyze_requirements([
        {"id": "REQ-001", "description": "Add two integer values."},
        {"id": "REQ-002", "description": "Reject empty input and raise a clear error."},
        {"id": "REQ-003", "description": "Prevent race conditions under concurrent requests."},
    ])

    assert (analysis["basic"], analysis["intermediate"], analysis["advanced"]) == (
        1, 1, 1
    )
    assert analysis["advanced_requirements"][0]["id"] == "REQ-003"
    assert analysis["advanced_requirements"][0]["reason"]
    assert [item["id"] for item in supported_requirements(analysis)] == [
        "REQ-001", "REQ-002"
    ]


def test_duplicate_case_removal_preserves_distinct_inputs():
    first = {
        "requirement_id": "REQ-001",
        "function_name": "add",
        "inputs": [2, 3],
        "expected": 5,
    }
    duplicate = {**first, "name": "Duplicate description"}
    different = {**first, "inputs": [0, 0], "expected": 0}

    unique, removed_count = remove_duplicate_cases([first, duplicate, different])

    assert unique == [first, different]
    assert removed_count == 1
