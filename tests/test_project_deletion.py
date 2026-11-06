from __future__ import annotations

from streamlit.testing.v1 import AppTest

from codeguard import database, services


def _app_with_project(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DATA_DIR", tmp_path)
    monkeypatch.setattr(database, "DATABASE_PATH", tmp_path / "deletion.sqlite3")
    database.initialize_database()
    project_id = services.create_project(
        title="Project to delete",
        requirements=["Project data should be scoped."],
    )
    app = AppTest.from_file(str(database.PROJECT_ROOT / "app.py"))
    app.run()
    return app, project_id


def _button(app, key):
    return next(button for button in app.button if button.key == key)


def test_project_actions_confirm_and_delete_only_selected_project(
    tmp_path, monkeypatch
):
    app, project_id = _app_with_project(tmp_path, monkeypatch)

    _button(app, f"cg-project-menu-{project_id}").click().run()
    assert any(
        button.key == f"cg-request-delete-{project_id}" for button in app.button
    )
    assert services.get_project(project_id) is not None

    _button(app, f"cg-request-delete-{project_id}").click().run()
    assert any(
        button.key == f"cg-confirm-delete-{project_id}" for button in app.button
    )
    assert services.get_project(project_id) is not None

    _button(app, f"cg-confirm-delete-{project_id}").click().run()
    assert services.get_project(project_id) is None
    assert any(
        item.value == f"Deleted project “Project to delete”."
        for item in app.success
    )
    assert app.exception == []


def test_cancel_delete_confirmation_keeps_project(tmp_path, monkeypatch):
    app, project_id = _app_with_project(tmp_path, monkeypatch)
    _button(app, f"cg-project-menu-{project_id}").click().run()
    _button(app, f"cg-request-delete-{project_id}").click().run()

    _button(app, f"cg-cancel-delete-{project_id}").click().run()

    assert services.get_project(project_id) is not None
    assert not any(
        button.key == f"cg-confirm-delete-{project_id}" for button in app.button
    )
