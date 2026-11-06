from __future__ import annotations

import sqlite3
import json

import pytest

from codeguard import database, services


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DATA_DIR", tmp_path)
    monkeypatch.setattr(database, "DATABASE_PATH", tmp_path / "test.sqlite3")
    database.initialize_database()


def _project() -> int:
    return services.create_project(
        title="Calculator",
        requirements=["Adds two values", "Rejects invalid input"],
        code="def add(a, b):\n    return a + b\n",
    )


def _case(requirement_id: str = "REQ-001") -> dict:
    return {
        "name": "Add integers",
        "description": "Returns the sum.",
        "testing_objective": "Verify two ordinary integer operands are summed.",
        "requirement_id": requirement_id,
        "category": "normal",
        "function_name": "add",
        "inputs": [2, 3],
        "expected": 5,
        "priority": "High",
    }


def test_project_persists_requirements_and_original_source(db):
    project_id = _project()
    project = services.get_project(project_id)
    assert project is not None
    assert project["title"] == "Calculator"
    assert [item["id"] for item in services.list_requirements(project_id)] == [
        "REQ-001", "REQ-002"
    ]
    original = services.get_original_code(project_id)
    assert original is not None
    assert original["original_content"].startswith("def add")
    assert services.list_code_versions(project_id)[0]["version_number"] == 1


def test_project_title_duplicate_is_case_insensitive(db):
    _project()
    with pytest.raises(services.ValidationError, match="already exists"):
        services.create_project(title="calculator")


def test_requirement_ids_are_scoped_to_each_project(db):
    first = _project()
    second = services.create_project(title="Second calculator", requirements=["Different feature"])
    assert services.list_requirements(first)[0]["id"] == "REQ-001"
    assert services.list_requirements(second)[0]["id"] == "REQ-001"


def test_legacy_global_requirement_ids_are_migrated(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DATA_DIR", tmp_path)
    monkeypatch.setattr(database, "DATABASE_PATH", tmp_path / "legacy.sqlite3")
    legacy_requirements = (
        "id TEXT NOT NULL,\n"
        "    project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,\n"
        "    description TEXT NOT NULL,\n"
        "    created_at TEXT NOT NULL DEFAULT (datetime('now')),\n"
        "    PRIMARY KEY(project_id, id)"
    )
    old_requirements = (
        "id TEXT PRIMARY KEY,\n"
        "    project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,\n"
        "    description TEXT NOT NULL,\n"
        "    created_at TEXT NOT NULL DEFAULT (datetime('now')),\n"
        "    UNIQUE(project_id, id)"
    )
    old_schema = database.SCHEMA.replace(legacy_requirements, old_requirements)
    assert old_schema != database.SCHEMA
    with sqlite3.connect(database.DATABASE_PATH) as connection:
        connection.executescript(old_schema)
        connection.execute("INSERT INTO schema_migrations(version) VALUES (1)")
        first_id = connection.execute(
            "INSERT INTO projects(title) VALUES ('Legacy one')"
        ).lastrowid
        second_id = connection.execute(
            "INSERT INTO projects(title) VALUES ('Legacy two')"
        ).lastrowid
        connection.execute(
            "INSERT INTO requirements(id, project_id, description) VALUES (?, ?, ?)",
            ("REQ-001", first_id, "First requirement"),
        )
    database.initialize_database()
    with database.connect() as connection:
        connection.execute(
            "INSERT INTO requirements(id, project_id, description) VALUES (?, ?, ?)",
            ("REQ-001", second_id, "Second requirement"),
        )
        versions = connection.execute(
            "SELECT version FROM schema_migrations ORDER BY version"
        ).fetchall()
        project_columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(projects)")
        }
    assert [row["version"] for row in versions] == [1, 2, 3, 4, 5, 6, 7]
    assert "is_pinned" in project_columns
    with database.connect() as connection:
        test_case_columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(test_cases)")
        }
    assert {
        "target_class", "target_kind", "keyword_inputs_json",
        "testing_objective", "complexity",
        "constructor_inputs_json", "constructor_keyword_inputs_json",
        "setup_json", "database_assertions_json",
    }.issubset(test_case_columns)


def test_test_case_metadata_migration_preserves_existing_cases(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DATA_DIR", tmp_path)
    monkeypatch.setattr(database, "DATABASE_PATH", tmp_path / "legacy-cases.sqlite3")
    old_schema = database.SCHEMA
    for column in (
        "    target_class TEXT NOT NULL DEFAULT '',\n",
        "    target_kind TEXT NOT NULL DEFAULT 'function',\n",
        "    keyword_inputs_json TEXT NOT NULL DEFAULT '{}',\n",
        "    constructor_inputs_json TEXT NOT NULL DEFAULT '[]',\n",
        "    constructor_keyword_inputs_json TEXT NOT NULL DEFAULT '{}',\n",
        "    setup_json TEXT NOT NULL DEFAULT '[]',\n",
        "    database_assertions_json TEXT NOT NULL DEFAULT '[]',\n",
    ):
        old_schema = old_schema.replace(column, "")
    with sqlite3.connect(database.DATABASE_PATH) as connection:
        connection.executescript(old_schema)
        project_id = connection.execute(
            "INSERT INTO projects(title) VALUES ('Legacy tests')"
        ).lastrowid
        assert project_id is not None
        connection.execute(
            """INSERT INTO test_cases
            (project_id, test_id, name, category, function_name, inputs_json,
             expected_json, approval_status)
            VALUES (?, 'TC-001', 'Legacy function', 'normal', 'solve', '[]', '1',
                    'approved')""",
            (project_id,),
        )

    database.initialize_database()

    case = services.list_test_cases(int(project_id))[0]
    assert case["test_id"] == "TC-001"
    assert case["target_class"] == ""
    assert case["target_kind"] == "function"
    assert case["constructor_inputs_json"] == "[]"
    assert case["setup_json"] == "[]"
    assert case["database_assertions_json"] == "[]"
    assert case["approval_status"] == "pending"


def test_class_test_case_metadata_is_persisted_and_editable(db):
    project_id = services.create_project(
        title="Inventory",
        requirements=["Stores products"],
        code=(
            "class Inventory:\n"
            "    def __init__(self, database='inventory.db'):\n"
            "        pass\n"
            "    def add_product(self, name, price, stock):\n"
            "        return 1\n"
        ),
    )
    test_id = services.add_test_case(project_id, {
        "name": "Adds product",
        "requirement_id": "REQ-001",
        "function_name": "add_product",
        "target_class": "Inventory",
        "target_kind": "instance_method",
        "constructor_inputs": ["__CODEGUARD_TEMP_DB__"],
        "constructor_keyword_inputs": {},
        "inputs": ["Laptop", 50.0, 100],
        "keyword_inputs": {},
        "expected": {"$type": "integer", "minimum": 1},
        "database_assertions": [{
            "table": "products",
            "where": {"product_id": {"$actual_return": True}},
            "values": {"name": "Laptop", "price": 50.0, "stock": 100},
        }],
    })
    case = services.list_test_cases(project_id)[0]
    assert case["test_id"] == test_id
    assert case["target_class"] == "Inventory"
    assert case["target_kind"] == "instance_method"
    assert json.loads(case["constructor_inputs_json"]) == ["__CODEGUARD_TEMP_DB__"]
    assert json.loads(case["database_assertions_json"])[0]["table"] == "products"

    services.set_test_status(project_id, case["id"], "approved")
    services.update_test_case(project_id, case["id"], {
        "name": "Adds a product",
        "function_name": "add_product",
        "target_class": "Inventory",
        "target_kind": "instance_method",
        "constructor_inputs": [],
        "constructor_keyword_inputs": {},
        "inputs": ["Laptop", 50.0, 100],
        "keyword_inputs": {},
        "expected": {"$type": "integer", "minimum": 1},
        "database_assertions": [],
    })
    updated = services.list_test_cases(project_id)[0]
    assert updated["approval_status"] == "approved"
    assert updated["target_class"] == "Inventory"


@pytest.mark.parametrize("assertion", [
    {"table": "products; DROP TABLE projects", "where": {"id": 1}, "values": {"name": "x"}},
    {"table": "products", "where": {"id": {"$actual_return": False}}, "values": {"name": "x"}},
])
def test_database_assertions_reject_unsafe_metadata(db, assertion):
    project_id = _project()
    with pytest.raises(services.ValidationError, match="Database assertion"):
        services.add_test_case(project_id, {
            **_case(),
            "database_assertions": [assertion],
        })


def test_project_rename_and_pin_are_persisted_and_sorted(db):
    first = services.create_project(title="First")
    second = services.create_project(title="Second")

    services.rename_project(first, "Renamed")
    services.set_project_pinned(first, True)

    project = services.get_project(first)
    assert project is not None
    assert project["title"] == "Renamed"
    assert project["is_pinned"] == 1
    assert [project["id"] for project in services.list_projects()] == [first, second]
    services.set_project_pinned(first, False)
    assert [project["id"] for project in services.list_projects()] == [second, first]


def test_project_rename_validates_title_and_protects_demo(db):
    project_id = services.create_project(title="Existing")
    other_project_id = services.create_project(title="Other")
    demo_id = services.create_project(title="Demo", is_demo=True)

    with pytest.raises(services.ValidationError, match="already exists"):
        services.rename_project(other_project_id, "Existing")
    with pytest.raises(services.ValidationError, match="1–120"):
        services.rename_project(project_id, " ")
    with pytest.raises(services.ValidationError, match="Demo projects"):
        services.rename_project(demo_id, "Renamed demo")
    with pytest.raises(services.ValidationError, match="Demo projects"):
        services.set_project_pinned(demo_id, True)


def test_invalid_generated_cases_are_saved_atomically(db):
    project_id = _project()
    with pytest.raises(services.ValidationError, match="requirement"):
        services.add_generated_test_cases(
            project_id, [_case(), _case("REQ-999")]
        )
    assert services.list_test_cases(project_id) == []


def test_invalid_class_method_cannot_be_saved_or_approved(db):
    project_id = services.create_project(
        title="Inventory validation",
        requirements=["Adds customers"],
        code=(
            "class InventorySystem:\n"
            "    def add_customer(self, name, email):\n"
            "        return 1\n"
        ),
    )
    with pytest.raises(services.ValidationError, match="register_customer"):
        services.add_test_case(project_id, {
            "name": "Invalid method",
            "requirement_id": "REQ-001",
            "function_name": "register_customer",
            "target_class": "InventorySystem",
            "target_kind": "instance_method",
            "inputs": ["Rahul", "rahul@example.com"],
            "expected": 1,
        })
    assert services.list_test_cases(project_id) == []


def test_generated_tests_are_approved_after_source_validation(db):
    project_id = _project()
    identifiers = services.add_generated_test_cases(project_id, [_case()])
    case = services.list_test_cases(project_id)[0]
    assert identifiers == [case["test_id"]]
    assert case["approval_status"] == "approved"


def test_generated_tests_cannot_be_approved_without_source_code(db):
    project_id = services.create_project(
        title="Project without source",
        requirements=["Calculates a result"],
    )

    with pytest.raises(services.ValidationError, match="source code"):
        services.add_generated_test_cases(project_id, [_case()])

    assert services.list_test_cases(project_id) == []


def test_generated_tests_persist_objective_complexity_and_category_coverage(db):
    project_id = _project()
    assessments = [
        {
            "category": category,
            "applicable": category == "normal",
            "reason": (
                "The source supports this case."
                if category == "normal"
                else "This category is not supported by the current source behavior."
            ),
        }
        for category in sorted(services.GENERATION_CATEGORIES)
    ]

    services.add_generated_test_cases(
        project_id,
        [_case()],
        source_code="def add(a, b):\n    return a + b\n",
        category_assessments=assessments,
    )

    case = services.list_test_cases(project_id)[0]
    assert case["testing_objective"] == "Verify two ordinary integer operands are summed."
    assert case["complexity"] == "Basic"
    report = services.get_generation_report(
        project_id, "def add(a, b):\n    return a + b\n"
    )
    assert report is not None
    assert len(report["category_assessments"]) == len(
        services.GENERATION_CATEGORIES
    )
    assert services.get_generation_report(project_id, "def add(a, b):\n    return a\n") is None


def test_requirement_linkage_is_not_execution_success(db):
    project_id = _project()
    services.add_test_case(project_id, _case())
    coverage = services.requirement_coverage(project_id)
    assert coverage["linked"] == 1
    assert coverage["uncovered"] == 1
    assert coverage["mapping"][0]["tests"][0]["execution_status"] == "not_run"


def test_execution_snapshots_and_comparison(db):
    project_id = _project()
    test_id = services.add_test_case(project_id, _case())
    services.set_test_status(
        project_id, services.list_test_cases(project_id)[0]["id"], "approved"
    )
    version_id = services.list_code_versions(project_id)[0]["id"]
    before_id = services.save_execution(project_id, version_id, {
        "status": "completed", "total": 1, "failed": 1, "duration": 0.3,
        "results": [{
            "test_case_id": services.list_test_cases(project_id)[0]["id"],
            "test_id": test_id, "name": "Add integers", "status": "failed",
            "message": "expected 5, got 4",
        }],
    })
    second_version = services.save_code_version(
        project_id, "def add(a, b):\n    return a + b\n", "user-edit"
    )
    after_id = services.save_execution(project_id, second_version, {
        "status": "completed", "total": 1, "passed": 1, "duration": 0.2,
        "results": [{
            "test_case_id": services.list_test_cases(project_id)[0]["id"],
            "test_id": test_id, "name": "Add integers", "status": "passed",
        }],
    })
    comparison = services.compare_executions(before_id, after_id)
    assert comparison["fixed"] == [test_id]
    assert comparison["regressions"] == []
    execution = services.get_execution(after_id)
    assert execution is not None
    assert execution["version_number"] == 2
    assert execution["coverage"] == {}


def test_execution_persists_expected_and_actual_outputs(db):
    project_id = _project()
    test_id = services.add_test_case(project_id, _case())
    case = services.list_test_cases(project_id)[0]
    version_id = services.list_code_versions(project_id)[0]["id"]
    execution_id = services.save_execution(project_id, version_id, {
        "status": "completed",
        "total": 1,
        "failed": 1,
        "results": [{
            "test_case_id": case["id"],
            "test_id": test_id,
            "name": case["name"],
            "status": "failed",
            "message": "AssertionError",
            "expected_output": "5",
            "actual_output": "4",
        }],
    })

    execution = services.get_execution(execution_id)
    assert execution is not None
    result = execution["results"][0]
    assert result["status"] == "failed"
    assert "Expected output: 5" in result["message"]
    assert "Actual output: 4" in result["message"]


def test_delete_test_case_is_project_scoped(db):
    project_id = _project()
    other_project_id = services.create_project(
        title="Other project", requirements=["Returns a value"]
    )
    test_id = services.add_test_case(project_id, _case())
    other_test_id = services.add_test_case(other_project_id, _case())
    case = next(
        item for item in services.list_test_cases(project_id)
        if item["test_id"] == test_id
    )

    with pytest.raises(services.ValidationError, match="this project"):
        services.delete_test_case(other_project_id, case["id"])
    services.delete_test_case(project_id, case["id"])

    assert services.list_test_cases(project_id) == []
    assert [item["test_id"] for item in services.list_test_cases(other_project_id)] == [
        other_test_id
    ]


def test_correction_approval_creates_new_version(db):
    project_id = _project()
    version_id = services.list_code_versions(project_id)[0]["id"]
    execution_id = services.save_execution(
        project_id, version_id, {"status": "completed"}
    )
    correction_id = services.create_correction(
        project_id, execution_id, version_id,
        "def add(a, b):\n    return a + b\n", "Return the sum.",
    )
    new_version_id = services.decide_correction(correction_id, True)
    assert new_version_id is not None
    correction = services.list_corrections(project_id)[0]
    assert correction["status"] == "approved"
    assert correction["applied_version_id"] == new_version_id
    version = services.get_code_version(new_version_id)
    assert version is not None
    assert version["kind"] == "ai-correction"


def test_update_missing_test_case_reports_error(db):
    project_id = _project()

    with pytest.raises(services.ValidationError, match="no longer exists"):
        services.update_test_case(project_id, 999, _case())


def test_execution_cannot_reference_another_projects_code_version(db):
    first_project_id = _project()
    second_project_id = services.create_project(
        title="Separate project",
        code="def solve():\n    return 42\n",
    )
    first_version_id = services.list_code_versions(first_project_id)[0]["id"]

    with pytest.raises(services.ValidationError, match="belong to this project"):
        services.save_execution(second_project_id, first_version_id, {})


def test_execution_cannot_attach_another_projects_test_result(db):
    first_project_id = _project()
    second_project_id = services.create_project(
        title="Separate project",
        code="def solve():\n    return 42\n",
    )
    services.add_test_case(first_project_id, _case())
    first_case_id = services.list_test_cases(first_project_id)[0]["id"]
    second_version_id = services.list_code_versions(second_project_id)[0]["id"]

    with pytest.raises(services.ValidationError, match="another project"):
        services.save_execution(second_project_id, second_version_id, {
            "results": [{"test_case_id": first_case_id}],
        })


def test_analysis_for_missing_execution_reports_error(db):
    with pytest.raises(services.ValidationError, match="no longer exists"):
        services.save_analysis(999, "test-model", "Analysis")


def test_correction_must_match_its_project_execution_and_source_version(db):
    first_project_id = _project()
    second_project_id = services.create_project(
        title="Separate project",
        code="def solve():\n    return 42\n",
    )
    second_version_id = services.list_code_versions(second_project_id)[0]["id"]
    first_version_id = services.list_code_versions(first_project_id)[0]["id"]
    services.save_code_version(
        first_project_id, "def add(a, b):\n    return a - b\n", "user-edit"
    )
    second_first_project_version_id = services.list_code_versions(first_project_id)[0]["id"]
    first_execution_id = services.save_execution(
        first_project_id, first_version_id, {"status": "completed"}
    )

    with pytest.raises(services.ValidationError, match="same project"):
        services.create_correction(
            second_project_id,
            first_execution_id,
            second_version_id,
            "def solve():\n    return 0\n",
            "Test correction",
        )

    with pytest.raises(services.ValidationError, match="execution source version"):
        services.create_correction(
            first_project_id,
            first_execution_id,
            second_first_project_version_id,
            "def add(a, b):\n    return 0\n",
            "Test correction",
        )


def test_stale_correction_cannot_replace_a_newer_code_version(db):
    project_id = _project()
    original_version_id = services.list_code_versions(project_id)[0]["id"]
    execution_id = services.save_execution(
        project_id, original_version_id, {"status": "completed"}
    )
    correction_id = services.create_correction(
        project_id,
        execution_id,
        original_version_id,
        "def add(a, b):\n    return a + b + 1\n",
        "Apply an old suggestion.",
    )
    services.save_code_version(
        project_id, "def add(a, b):\n    return a * b\n", "user-edit"
    )

    with pytest.raises(services.ValidationError, match="newer code version"):
        services.decide_correction(correction_id, True)

    assert services.list_corrections(project_id)[0]["status"] == "pending"


def test_delete_project_removes_related_records_but_preserves_demo_projects(db):
    project_id = _project()
    case_id = services.add_test_case(project_id, _case())
    case = services.list_test_cases(project_id)[0]
    version_id = services.list_code_versions(project_id)[0]["id"]
    execution_id = services.save_execution(project_id, version_id, {
        "status": "completed",
        "total": 1,
        "failed": 1,
        "results": [{
            "test_case_id": case["id"],
            "test_id": case_id,
            "name": case["name"],
            "status": "failed",
        }],
    })
    services.save_analysis(execution_id, "test-model", "Failure analysis")
    services.create_correction(
        project_id,
        execution_id,
        version_id,
        "def add(a, b):\n    return a + b\n",
        "Return the sum.",
    )
    demo_id = services.create_project(title="Protected demo", is_demo=True)

    services.delete_project(project_id)

    assert services.get_project(project_id) is None
    assert services.list_test_cases(project_id) == []
    assert services.list_code_versions(project_id) == []
    assert services.list_executions(project_id) == []
    assert services.list_analyses(project_id) == []
    assert services.list_corrections(project_id) == []
    demo = services.get_project(demo_id)
    assert demo is not None
    assert demo["title"] == "Protected demo"
    with pytest.raises(services.ValidationError, match="Demo projects"):
        services.delete_project(demo_id)
