from __future__ import annotations

from streamlit.testing.v1 import AppTest

from codeguard import database, services, ui


def _app_with_project(tmp_path, monkeypatch, *, filename="main.py", code=""):
    monkeypatch.setattr(database, "DATA_DIR", tmp_path)
    monkeypatch.setattr(database, "DATABASE_PATH", tmp_path / "workspace.sqlite3")
    database.initialize_database()
    project_id = services.create_project(
        title="Workspace sample",
        description="A project for workspace tests.",
        requirements=["The project should greet a user."],
        filename=filename,
        code=code,
    )
    app = AppTest.from_file(str(database.PROJECT_ROOT / "app.py"))
    app.session_state["active_project_id"] = project_id
    app.session_state["project_open"] = True
    app.session_state["cg_navigation"] = "New Project"
    app.session_state["cg_home_open"] = False
    app.run()
    return app, project_id


def _button(app, key):
    return next(button for button in app.button if button.key == key)


def _generated_case():
    return {
        "name": "Greets a named user",
        "description": "Returns a greeting for the supplied name.",
        "testing_objective": "Verify a supplied name is included in the returned greeting.",
        "requirement_id": "REQ-001",
        "category": "normal",
        "function_name": "greet",
        "inputs": ["Ari"],
        "expected": "Hello, Ari!",
        "expected_exception": "",
        "preconditions": "",
        "priority": "High",
    }


def test_project_overview_is_separate_and_workspace_generates_with_source(
    tmp_path, monkeypatch
):
    source = "def greet(name):\n    return f'Hello, {name}!'\n"
    app, project_id = _app_with_project(
        tmp_path, monkeypatch, filename="uploaded_greeting.py", code=source
    )
    assert any(item.value == "Project overview" for item in app.subheader)
    assert any(
        button.key == f"overview-open-workspace-{project_id}"
        for button in app.button
    )
    assert not any(button.key == "cg-nav-project-overview" for button in app.button)

    calls = []

    def generate(project, requirements, *, source_code="", **_kwargs):
        calls.append((project["id"], source_code))
        return [_generated_case()]

    monkeypatch.setattr(ui, "generate_test_cases", generate)
    _button(app, f"overview-open-workspace-{project_id}").click().run()

    assert app.session_state["cg_project_view"] == "workspace"
    assert calls == [(project_id, source)]
    saved_cases = services.list_test_cases(project_id)
    assert len(saved_cases) == 1
    assert saved_cases[0]["approval_status"] == "approved"
    assert any(item.value == "uploaded_greeting.py" for item in app.subheader)
    assert app.exception == []

    app.run()
    assert len(calls) == 1


def test_initial_generation_error_is_visible_and_retryable(tmp_path, monkeypatch):
    app, project_id = _app_with_project(
        tmp_path,
        monkeypatch,
        code="def greet(name):\n    return f'Hello, {name}!'\n",
    )
    attempts = 0

    def generate(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ui.OllamaError("Ollama is unavailable.")
        return [_generated_case()]

    monkeypatch.setattr(ui, "generate_test_cases", generate)
    _button(app, f"overview-open-workspace-{project_id}").click().run()
    assert any("Ollama is unavailable." in item.value for item in app.error)
    assert any(
        button.key == f"retry-generation-{project_id}" for button in app.button
    )

    _button(app, f"retry-generation-{project_id}").click().run()
    assert attempts == 2
    assert len(services.list_test_cases(project_id)) == 1
    assert services.list_test_cases(project_id)[0]["approval_status"] == "approved"
    assert app.exception == []


def test_manual_project_creation_waits_for_code_before_generating_tests(
    tmp_path, monkeypatch
):
    app, existing_id = _app_with_project(tmp_path, monkeypatch)
    _button(app, f"back-{existing_id}").click().run()

    next(item for item in app.text_input if item.label == "Project title").set_value(
        "Created through the form"
    )
    next(
        item for item in app.text_area if item.label == "Functional requirements"
    ).set_value("The function should greet a user.")

    calls = []

    def generate(project, requirements, *, source_code="", **_kwargs):
        calls.append((project["id"], source_code))
        return [_generated_case()]

    monkeypatch.setattr(ui, "generate_test_cases", generate)
    next(item for item in app.button if item.label == "Submit Requirements").click().run()

    project_id = app.session_state["active_project_id"]
    project = services.get_project(project_id)
    assert project is not None
    assert project["title"] == "Created through the form"
    assert app.session_state["cg_project_view"] == "workspace"
    assert app.tabs[-1].label == "Project Overview"
    assert calls == []
    assert services.list_test_cases(project_id) == []
    assert any(
        "automatically generate and validate tests" in item.value
        for item in app.info
    )
    assert app.exception == []


def test_adding_source_after_submission_starts_generation_automatically(
    tmp_path, monkeypatch
):
    app, project_id = _app_with_project(tmp_path, monkeypatch)
    _button(app, f"overview-open-workspace-{project_id}").click().run()
    assert app.session_state[f"cg_initial_generation_status_{project_id}"] == "no_source"
    source = "def greet(name):\n    return f'Hello, {name}!'\n"
    captured = []
    monkeypatch.setattr(
        ui,
        "st_ace",
        lambda **kwargs: source,
    )
    monkeypatch.setattr(
        ui,
        "generate_test_cases",
        lambda project, requirements, *, source_code="", **_kwargs: (
            captured.append((project["id"], source_code)) or [_generated_case()]
        ),
    )

    app.run()

    assert captured == [(project_id, source)]
    assert services.list_code_versions(project_id)[0]["content"] == source
    assert len(services.list_test_cases(project_id)) == 1
    assert app.exception == []


def test_manual_code_is_saved_before_generation_and_used_for_target_validation(
    tmp_path, monkeypatch
):
    app, project_id = _app_with_project(tmp_path, monkeypatch)
    original = services.list_code_versions(project_id)[0]
    edited_source = "def greet(name):\n    return f'Hello, {name}!'\n"
    project = services.get_project(project_id)
    assert project is not None
    observed = {}

    def generate(_project, _requirements, *, source_code, **_kwargs):
        observed["source"] = source_code
        observed["saved_source"] = services.list_code_versions(project_id)[0]["content"]
        return [_generated_case()]

    monkeypatch.setattr(ui, "generate_test_cases", generate)
    class Progress:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def write(self, *_args):
            pass

        def update(self, **_kwargs):
            pass

    monkeypatch.setattr(ui.st, "status", lambda *_args, **_kwargs: Progress())
    identifiers = ui._generate_workspace_tests(
        project,
        services.list_requirements(project_id),
        edited_source,
        original,
    )

    assert len(identifiers) == 1
    assert observed == {
        "source": edited_source,
        "saved_source": edited_source,
    }
    assert services.list_test_cases(project_id)[0]["approval_status"] == "approved"
    assert app.exception == []


def test_upload_mode_reveals_file_picker_without_submitting_form(
    tmp_path, monkeypatch
):
    app, project_id = _app_with_project(tmp_path, monkeypatch)
    _button(app, f"back-{project_id}").click().run()

    next(item for item in app.radio if item.label == "Starting code").set_value(
        "Upload Existing Code"
    ).run()

    assert len(app.file_uploader) == 1
    assert app.exception == []


def test_uploaded_project_preserves_source_and_generates_against_it(
    tmp_path, monkeypatch
):
    app, previous_id = _app_with_project(tmp_path, monkeypatch)
    _button(app, f"back-{previous_id}").click().run()
    next(item for item in app.radio if item.label == "Starting code").set_value(
        "Upload Existing Code"
    ).run()
    source = "def greet(name):\n    return 'Hello, ' + name\n"
    app.file_uploader[0].set_value(
        ("uploaded_greeting.py", source.encode("utf-8"), "text/x-python")
    ).run()
    next(item for item in app.text_input if item.label == "Project title").set_value(
        "Uploaded project"
    )
    next(
        item for item in app.text_area if item.label == "Functional requirements"
    ).set_value("The greet function returns a greeting for a name.")

    captured_source = []
    monkeypatch.setattr(
        ui,
        "generate_test_cases",
        lambda project, requirements, *, source_code="", **_kwargs: (
            captured_source.append(source_code) or [_generated_case()]
        ),
    )
    next(item for item in app.button if item.label == "Submit Requirements").click().run()

    project_id = app.session_state["active_project_id"]
    original = services.get_original_code(project_id)
    assert original is not None
    assert original["filename"] == "uploaded_greeting.py"
    assert original["original_content"] == source
    assert app.session_state["cg_project_view"] == "workspace"
    assert captured_source == [source]
    assert services.list_test_cases(project_id)[0]["approval_status"] == "approved"
    assert app.exception == []


def test_submitting_requirements_and_manual_source_starts_generation_automatically(
    tmp_path, monkeypatch
):
    app, previous_id = _app_with_project(tmp_path, monkeypatch)
    _button(app, f"back-{previous_id}").click().run()
    next(item for item in app.text_input if item.label == "Project title").set_value(
        "Requirements submitted with source"
    )
    next(
        item for item in app.text_area if item.label == "Functional requirements"
    ).set_value("The function should greet a user.")
    source = "def greet(name):\n    return f'Hello, {name}!'\n"
    next(item for item in app.text_area if item.label == "Python source code").set_value(
        source
    )
    captured = []

    def generate(project, requirements, *, source_code="", **_kwargs):
        captured.append((project["id"], source_code))
        return [_generated_case()]

    monkeypatch.setattr(ui, "generate_test_cases", generate)
    next(
        item for item in app.button if item.label == "Submit Requirements"
    ).click().run()

    project_id = app.session_state["active_project_id"]
    assert captured == [(project_id, source)]
    assert len(services.list_test_cases(project_id)) == 1
    assert services.list_test_cases(project_id)[0]["approval_status"] == "approved"
    assert app.exception == []


def test_project_overview_shows_source_test_details_and_execution_results(
    tmp_path, monkeypatch
):
    source = "def greet(name):\n    return f'Hello, {name}!'\n"
    app, project_id = _app_with_project(
        tmp_path, monkeypatch, filename="greeting.py", code=source
    )
    services.add_test_case(project_id, _generated_case())
    case = services.list_test_cases(project_id)[0]
    services.set_test_status(project_id, case["id"], "approved")
    version_id = services.list_code_versions(project_id)[0]["id"]
    services.save_execution(project_id, version_id, {
        "status": "completed",
        "total": 1,
        "passed": 1,
        "output": "1 passed",
        "coverage": {
            "available": True,
            "percentage": 100.0,
            "covered_lines": 2,
            "missed_lines": 0,
        },
        "results": [{
            "test_case_id": case["id"],
            "test_id": case["test_id"],
            "name": case["name"],
            "status": "passed",
            "message": "Test passed.",
            "traceback": "",
            "duration": 0.01,
        }],
    })
    execution_id = services.list_executions(project_id)[0]["id"]
    services.save_analysis(execution_id, "qwen2.5:3b", "The test passed.")
    services.create_correction(
        project_id,
        execution_id,
        version_id,
        "def greet(name):\n    return 'Hello, ' + name\n",
        "Use simple string concatenation.",
    )

    app.run()

    assert any(item.value == "Source code" for item in app.subheader)
    assert any(item.value == "Test cases" for item in app.subheader)
    assert any(item.value == "Latest test execution" for item in app.subheader)
    assert any(item.value == "Execution history" for item in app.subheader)
    assert any(item.value == "Code version history" for item in app.subheader)
    assert any(item.value == "AI failure analyses" for item in app.subheader)
    assert any(item.value == "Suggested correction history" for item in app.subheader)
    assert any("greeting.py" in item.value for item in app.markdown)
    assert any(
        item.label == "View current Python source" for item in app.expander
    )
    assert any(item.label == "View original Python source" for item in app.expander)
    assert any(item.label == "View latest Pytest output" for item in app.expander)
    original = services.get_original_code(project_id)
    assert original is not None
    assert original["original_content"] == source
    assert any(
        "Measured Coverage.py line coverage: 100.0%" in item.value
        for item in app.caption
    )
    assert app.exception == []


def test_workspace_runs_only_current_project_cases_and_deletes_without_confirmation(
    tmp_path, monkeypatch
):
    source = "def greet(name):\n    return f'Hello, {name}!'\n"
    app, project_id = _app_with_project(
        tmp_path, monkeypatch, code=source
    )
    services.add_test_case(project_id, _generated_case())
    case = services.list_test_cases(project_id)[0]
    services.set_test_status(project_id, case["id"], "approved")
    case = services.list_test_cases(project_id)[0]
    case_id = case["id"]
    other_project_id = services.create_project(
        title="Separate project",
        requirements=["Calculates a result."],
        code="def solve():\n    return 42\n",
    )
    services.add_test_case(
        other_project_id,
        {
            **_generated_case(),
            "requirement_id": "REQ-001",
            "function_name": "solve",
            "inputs": [],
            "expected": 42,
        },
    )
    other_case_id = services.list_test_cases(other_project_id)[0]["id"]
    app.run()
    _button(app, f"overview-open-workspace-{project_id}").click().run()
    assert app.tabs[-1].label == "Project Overview"
    test_expander = next(
        item for item in app.expander
        if case["test_id"] in item.label and "Greets a named user" in item.label
    )
    calls = []

    def run_cases(code, cases):
        calls.append((code, [item["id"] for item in cases]))
        return {
            "status": "completed",
            "total": 1,
            "passed": 1,
            "failed": 0,
            "skipped": 0,
            "errors": 0,
            "duration": 0.1,
            "coverage": {"available": False},
            "output": "1 passed",
            "results": [{
                "test_case_id": case_id,
                "test_id": "TC-001",
                "name": "Greets a named user",
                "status": "passed",
                "message": "",
                "expected_output": "'Hello, Ari!'",
                "actual_output": "'Hello, Ari!'",
            }],
        }

    monkeypatch.setattr(ui, "run_approved_tests", run_cases)
    assert _button(app, f"run-tests-{project_id}").label == "Run Tests"
    delete_key = f"delete-test-{project_id}-{case_id}"
    assert _button(app, delete_key).help == "Delete this test case immediately"

    _button(app, f"run-tests-{project_id}").click().run()

    assert calls == [(source, [case_id])]
    execution_id = services.list_executions(project_id)[0]["id"]
    execution = services.get_execution(execution_id)
    assert execution is not None
    result = execution["results"][0]
    assert result["status"] == "passed"
    assert "Expected output: 'Hello, Ari!'" in result["message"]
    assert "Actual output: 'Hello, Ari!'" in result["message"]
    assert any("PASSED" in item.label for item in app.expander)

    _button(app, delete_key).click().run()
    assert services.list_test_cases(project_id) == []
    assert [item["id"] for item in services.list_test_cases(other_project_id)] == [
        other_case_id
    ]
