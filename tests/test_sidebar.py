from __future__ import annotations

from streamlit.testing.v1 import AppTest

from codeguard import database, services


def _app_with_project(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DATA_DIR", tmp_path)
    monkeypatch.setattr(database, "DATABASE_PATH", tmp_path / "sidebar.sqlite3")
    database.initialize_database()
    project_id = services.create_project(
        title="Sidebar sample",
        project_type="Library",
        requirements=["The sidebar can open this project."],
    )
    app = AppTest.from_file(str(database.PROJECT_ROOT / "app.py"))
    app.run()
    return app, project_id


def _button(app, key: str):
    return next(button for button in app.button if button.key == key)


def _project_info(project_id: int) -> dict:
    project = services.get_project(project_id)
    assert project is not None
    return project


def test_recents_expands_in_sidebar_without_navigating_away_from_project(
    tmp_path, monkeypatch
):
    app, project_id = _app_with_project(tmp_path, monkeypatch)
    app.session_state["active_project_id"] = project_id
    app.session_state["project_open"] = True
    app.session_state["cg_navigation"] = "New Project"
    app.session_state["cg_home_open"] = False
    app.run()
    assert [item.value for item in app.title] == ["Sidebar sample"]

    _button(app, "cg-nav-recents").click().run()
    assert app.session_state["recents_expanded"] is False
    assert app.session_state["cg_navigation"] == "New Project"
    assert [item.value for item in app.title] == ["Sidebar sample"]
    assert not any(item.value == "Recents" for item in app.title)

    _button(app, "cg-recents-toggle").click().run()
    assert app.session_state["recents_expanded"] is True
    assert [item.value for item in app.title] == ["Sidebar sample"]


def test_recent_project_button_opens_project_without_recents_page(
    tmp_path, monkeypatch
):
    app, project_id = _app_with_project(tmp_path, monkeypatch)
    _button(app, f"cg-recent-project-{project_id}").click().run()
    assert app.session_state["cg_navigation"] == "New Project"
    assert app.session_state["active_project_id"] == project_id
    assert app.session_state["project_open"] is True
    assert [item.value for item in app.title] == ["Sidebar sample"]
    assert not any(item.value == "Recents" for item in app.title)

    _button(app, f"back-{project_id}").click().run()
    assert app.session_state["project_open"] is False
    assert any(item.value == "Start with your requirements" for item in app.title)


def test_recent_project_menu_renames_and_pins_project(tmp_path, monkeypatch):
    app, project_id = _app_with_project(tmp_path, monkeypatch)

    _button(app, f"cg-project-menu-{project_id}").click().run()
    assert any(button.key == f"cg-rename-project-{project_id}" for button in app.button)
    assert any(button.key == f"cg-toggle-pin-{project_id}" for button in app.button)
    assert any(button.key == f"cg-request-delete-{project_id}" for button in app.button)

    _button(app, f"cg-toggle-pin-{project_id}").click().run()
    assert _project_info(project_id)["is_pinned"] == 1

    _button(app, f"cg-project-menu-{project_id}").click().run()
    _button(app, f"cg-rename-project-{project_id}").click().run()
    next(
        item for item in app.text_input
        if item.key == f"cg-rename-input-{project_id}"
    ).set_value("Renamed sidebar sample")
    next(item for item in app.button if item.label == "Save rename").click().run()

    project = _project_info(project_id)
    assert project["title"] == "Renamed sidebar sample"
    assert project["is_pinned"] == 1
    assert any(
        button.key == f"cg-recent-project-{project_id}"
        and button.label == "Renamed sidebar sample"
        for button in app.button
    )
    assert app.exception == []


def test_navigation_buttons_keep_their_destination_and_project_data(
    tmp_path, monkeypatch
):
    app, project_id = _app_with_project(tmp_path, monkeypatch)
    _button(app, f"cg-recent-project-{project_id}").click().run()
    _button(app, "cg-nav-demo").click().run()
    assert app.session_state["cg_navigation"] == "Demo"
    assert app.session_state["project_open"] is False
    assert [item.value for item in app.title] == ["Demo"]
    assert _project_info(project_id)["title"] == "Sidebar sample"


def test_brand_returns_home_and_about_is_not_in_sidebar(tmp_path, monkeypatch):
    app, _ = _app_with_project(tmp_path, monkeypatch)
    _button(app, "cg-nav-new-project").click().run()
    assert any(item.value == "Start with your requirements" for item in app.title)

    _button(app, "cg-home-brand").click().run()
    assert app.session_state["cg_home_open"] is True
    assert app.session_state["cg_navigation"] == "Home"
    assert any(button.key == "cg-home-create-project" for button in app.button)
    assert any(button.key == "cg-home-explore-demo" for button in app.button)
    assert not any(button.key == "cg-nav-about" for button in app.button)
    assert not any(button.key == "cg-sidebar-toggle" for button in app.button)


def test_homepage_actions_open_existing_destinations(tmp_path, monkeypatch):
    app, _ = _app_with_project(tmp_path, monkeypatch)

    _button(app, "cg-home-create-project").click().run()
    assert app.session_state["cg_home_open"] is False
    assert app.session_state["cg_navigation"] == "New Project"
    assert any(item.value == "Start with your requirements" for item in app.title)

    _button(app, "cg-home-brand").click().run()
    _button(app, "cg-home-explore-demo").click().run()
    assert app.session_state["cg_home_open"] is False
    assert app.session_state["cg_navigation"] == "Demo"
    assert any(item.value == "Demo" for item in app.title)


def test_start_and_closing_actions_use_existing_new_project_route(
    tmp_path, monkeypatch
):
    app, _ = _app_with_project(tmp_path, monkeypatch)

    _button(app, "cg-home-start-project").click().run()
    assert app.session_state["cg_navigation"] == "New Project"
    assert any(item.value == "Start with your requirements" for item in app.title)

    _button(app, "cg-home-brand").click().run()
    _button(app, "cg-home-closing-project").click().run()
    assert app.session_state["cg_navigation"] == "New Project"
    assert any(item.value == "Start with your requirements" for item in app.title)
