from __future__ import annotations

import pytest

from codeguard.source_analysis import (
    SourceAnalysisError,
    analyze_source,
    validate_case_target,
)


def test_ast_inspection_finds_functions_classes_signatures_imports_and_returns():
    inspection = analyze_source(
        "import sqlite3\n"
        "def calculate_total(price, quantity=1):\n"
        "    return price * quantity\n"
        "class InventorySystem:\n"
        "    def __init__(self, database='inventory.db'):\n"
        "        self.database = database\n"
        "    def add_product(self, name, price, stock):\n"
        "        return self.cursor.lastrowid\n"
    )

    assert inspection["imports"] == ["sqlite3"]
    calculate = inspection["functions"][0]
    assert [item["name"] for item in calculate["parameters"]] == [
        "price", "quantity"
    ]
    assert [item["required"] for item in calculate["parameters"]] == [True, False]
    inventory = inspection["classes"][0]
    assert inventory["name"] == "InventorySystem"
    assert inventory["constructor"]["parameters"][0]["name"] == "database"
    add_product = inventory["methods"][0]
    assert add_product["kind"] == "instance_method"
    assert [item["name"] for item in add_product["parameters"]] == [
        "name", "price", "stock"
    ]
    assert add_product["return"]["kind"] == "integer"


def test_ast_inspection_exposes_branches_thresholds_raises_and_dependencies():
    inspection = analyze_source(
        "def grade(name: str, marks: list[int]):\n"
        "    if not name:\n"
        "        raise ValueError('name required')\n"
        "    if not marks:\n"
        "        raise ValueError('marks required')\n"
        "    percentage = sum(marks) / len(marks)\n"
        "    if percentage >= 90:\n"
        "        return 'A'\n"
        "    elif percentage >= 75:\n"
        "        return 'B'\n"
        "    return 'C'\n"
    )
    grade = inspection["functions"][0]

    assert [item["name"] for item in grade["parameters"]] == ["name", "marks"]
    assert len(grade["branches"]) == 4
    assert grade["boundary_values"] == [75, 90]
    assert grade["raises"] == ["ValueError('name required')", "ValueError('marks required')"]
    assert "sum" in grade["calls"]


def test_ast_inspection_includes_locally_inherited_methods():
    inspection = analyze_source(
        "class Base:\n"
        "    def close(self):\n"
        "        return None\n"
        "class Inventory(Base):\n"
        "    pass\n"
    )

    assert inspection["classes"][1]["bases"] == ["Base"]
    assert inspection["classes"][1]["methods"][0]["name"] == "close"
    assert inspection["classes"][1]["methods"][0]["declared_in"] == "Base"


def test_ast_inspection_does_not_execute_source():
    inspection = analyze_source(
        "raise RuntimeError('must never execute')\n"
        "def harmless(value):\n"
        "    return value\n"
    )

    assert inspection["functions"][0]["name"] == "harmless"


def test_ast_inspection_reports_syntax_errors():
    with pytest.raises(SourceAnalysisError, match="line 1"):
        analyze_source("def broken(:\n    pass\n")


def test_target_validation_rejects_mismatched_method_signature():
    inspection = analyze_source(
        "class Inventory:\n"
        "    def add_product(self, name, price, stock):\n"
        "        return 1\n"
    )
    with pytest.raises(SourceAnalysisError, match="missing required argument"):
        validate_case_target({
            "target_class": "Inventory",
            "function_name": "add_product",
            "inputs": [[1, 50.0, 100]],
            "constructor_inputs": [],
        }, inspection)


def test_target_validation_accepts_exact_inventory_method_and_rejects_unknown_name():
    inspection = analyze_source(
        "class InventorySystem:\n"
        "    def add_customer(self, name, email):\n"
        "        return 1\n"
    )
    validate_case_target({
        "target_class": "InventorySystem",
        "target_kind": "instance_method",
        "function_name": "add_customer",
        "constructor_inputs": [],
        "inputs": ["Rahul", "rahul@example.com"],
        "expected": {"$type": "integer"},
    }, inspection)
    with pytest.raises(SourceAnalysisError, match="register_customer"):
        validate_case_target({
            "target_class": "InventorySystem",
            "target_kind": "instance_method",
            "function_name": "register_customer",
            "constructor_inputs": [],
            "inputs": ["Rahul", "rahul@example.com"],
            "expected": {"$type": "integer"},
        }, inspection)


def test_target_validation_checks_constructor_and_keyword_parameter_names():
    inspection = analyze_source(
        "class InventorySystem:\n"
        "    def __init__(self, database=':memory:'):\n"
        "        pass\n"
        "    def create_order(self, customer_id, products, discount=0):\n"
        "        return 1\n"
    )
    base = {
        "target_class": "InventorySystem",
        "target_kind": "instance_method",
        "function_name": "create_order",
        "inputs": [1, [(1, 1)]],
        "keyword_inputs": {"discount": 0},
        "expected": 1,
    }
    validate_case_target({
        **base,
        "constructor_inputs": [],
    }, inspection)
    with pytest.raises(SourceAnalysisError, match="constructor"):
        validate_case_target({
            **base,
            "constructor_inputs": [1, 2, 3],
        }, inspection)
    with pytest.raises(SourceAnalysisError, match="does not accept"):
        validate_case_target({
            **base,
            "constructor_inputs": [],
            "keyword_inputs": {"discount_rate": 0},
        }, inspection)


def test_target_validation_rejects_named_argument_objects_in_positional_values():
    inspection = analyze_source(
        "def add(left, right):\n"
        "    return left + right\n"
    )
    case = {
        "function_name": "add",
        "target_class": "",
        "target_kind": "function",
        "inputs": [
            {"name": "left", "value": 2},
            {"name": "right", "value": 3},
        ],
        "keyword_inputs": {},
        "expected": 5,
    }

    with pytest.raises(SourceAnalysisError, match="named argument object"):
        validate_case_target(case, inspection)


def test_target_validation_allows_omitted_required_args_only_for_expected_type_error():
    inspection = analyze_source(
        "def generate_student_report(name, marks):\n"
        "    return name\n"
    )
    base = {
        "function_name": "generate_student_report",
        "target_kind": "function",
        "inputs": [],
        "keyword_inputs": {},
        "expected": None,
    }

    with pytest.raises(SourceAnalysisError, match="missing required argument"):
        validate_case_target(base, inspection)
    validate_case_target(
        {**base, "expected_exception": "TypeError"},
        inspection,
    )
