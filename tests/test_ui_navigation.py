from __future__ import annotations

from streamlit.testing.v1 import AppTest

from codeguard import database, services


def test_demo_workspace_back_button_returns_to_demo_landing(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DATA_DIR", tmp_path)
    monkeypatch.setattr(database, "DATABASE_PATH", tmp_path / "navigation.sqlite3")
    database.initialize_database()
    project_id = services.create_project(
        title="Navigation demo",
        is_demo=True,
        requirements=["The sample remains isolated."],
    )

    app = AppTest.from_file(str(database.PROJECT_ROOT / "app.py"))
    app.session_state["active_project_id"] = project_id
    app.session_state["project_open"] = True
    app.session_state["navigation"] = "Demo"
    app.run()
    assert [item.value for item in app.title] == ["Navigation demo"]

    next(item for item in app.button if item.label == "Back to Demo").click().run()

    assert app.session_state["cg_navigation"] == "Demo"
    assert any(item.key == "cg-nav-demo" for item in app.button)
    assert [item.value for item in app.title] == ["Demo"]
    assert any(item.label == "Open calculator demo" for item in app.button)
    project = services.get_project(project_id)
    assert project is not None
    assert project["is_demo"] == 1
