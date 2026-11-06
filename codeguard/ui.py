from __future__ import annotations

import base64
import difflib
import json
import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_ace import st_ace

from codeguard.ai import (
    MODEL,
    OllamaError,
    analyze_failures,
    generate_test_cases,
    suggest_correction,
)
from codeguard.database import PROJECT_ROOT, initialize_database
from codeguard.execution import SandboxError, run_approved_tests
from codeguard.reports import executions_csv, json_export, pdf_report, tests_csv
from codeguard.requirement_analysis import (
    analyze_requirements,
    remove_duplicate_cases,
    supported_requirements,
)
from codeguard import services
from codeguard.services import ValidationError

LOGGER = logging.getLogger(__name__)
NAVIGATION = ["New Project", "Demo"]
NAVIGATION_ICONS = {
    "New Project": ":material/add_box:",
    "Demo": ":material/play_circle:",
}
MAX_UPLOAD_BYTES = 200_000
CATEGORY_LABELS = {
    "normal": "Normal",
    "empty_input": "Empty Input",
    "null_value": "Null Value",
    "invalid_type": "Invalid Data Type",
    "boundary": "Boundary",
    "conditional": "Conditional Logic",
    "combination": "Input Combination",
    "validation": "Input Validation",
    "failure": "Failure Scenario",
    "invalid": "Invalid Input",
    "edge": "Edge Case",
    "exception": "Exception",
    "other": "Other",
}
CATEGORY_OPTIONS = list(CATEGORY_LABELS)
PRIORITY_OPTIONS = ["Low", "Medium", "High", "Critical"]
EXCEPTION_OPTIONS = [
    "", "ValueError", "TypeError", "ZeroDivisionError", "IndexError",
    "KeyError", "AssertionError", "OverflowError", "FileNotFoundError",
]


def _show_error(error: Exception) -> None:
    LOGGER.exception("User action failed", exc_info=error)
    st.error(str(error))


def _category_label(value: object) -> str:
    if not isinstance(value, str):
        return str(value)
    return CATEGORY_LABELS.get(value, value)


def _parse_json_text(value: str | None, field_name: str) -> Any:
    if value is None:
        raise ValidationError(f"{field_name} cannot be empty.")
    return json.loads(value)


def _load_styles() -> None:
    css_file = PROJECT_ROOT / "assets" / "styles.css"
    st.markdown(f"<style>{css_file.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def _logo_markup(width: int) -> str:
    logo = (PROJECT_ROOT / "assets" / "codeguard-logo.svg").read_bytes()
    encoded = base64.b64encode(logo).decode("ascii")
    return (
        f'<img class="cg-logo" src="data:image/svg+xml;base64,{encoded}" '
        f'width="{width}" height="{width}" alt="CodeGuard AI">'
    )


def _select_navigation(destination: str) -> None:
    st.session_state.cg_navigation = destination
    st.session_state.project_open = False
    st.session_state.cg_home_open = False


def _go_home() -> None:
    st.session_state.cg_home_open = True
    st.session_state.cg_navigation = "Home"


def _toggle_recents() -> None:
    st.session_state.recents_expanded = not st.session_state.recents_expanded


def _render_sidebar() -> str:
    legacy_navigation = st.session_state.get("navigation")
    st.session_state.setdefault(
        "cg_navigation",
        legacy_navigation if legacy_navigation in NAVIGATION else "Home",
    )
    if st.session_state.cg_navigation not in (*NAVIGATION, "Home"):
        st.session_state.cg_navigation = "Home"
    st.session_state.setdefault(
        "cg_home_open", st.session_state.cg_navigation == "Home"
    )
    if st.session_state.cg_home_open:
        st.session_state.cg_navigation = "Home"
    elif st.session_state.cg_navigation == "Home":
        st.session_state.cg_home_open = True
    st.session_state.setdefault("recents_expanded", True)

    st.sidebar.markdown(
        '<span class="cg-sidebar-state cg-sidebar-is-collapsed"></span>',
        unsafe_allow_html=True,
    )
    with st.sidebar:
        st.button(
            "CodeGuard AI",
            key="cg-home-brand",
            icon=":material/verified_user:",
            help="CodeGuard AI home",
            type=(
                "primary"
                if st.session_state.cg_navigation == "Home"
                else "tertiary"
            ),
            width="stretch",
            on_click=_go_home,
        )

        for destination in NAVIGATION:
            st.button(
                destination,
                key=f"cg-nav-{destination.lower().replace(' ', '-')}",
                icon=NAVIGATION_ICONS[destination],
                help=destination,
                type=(
                    "primary"
                    if st.session_state.cg_navigation == destination
                    else "tertiary"
                ),
                width="stretch",
                on_click=_select_navigation,
                args=(destination,),
            )

        recent_column, chevron_column = st.columns([5, 1])
        recent_column.button(
            "Recents",
            key="cg-nav-recents",
            icon=":material/schedule:",
            help="Recent projects",
            type="tertiary",
            width="stretch",
            on_click=_toggle_recents,
        )
        chevron_column.button(
            " ",
            key="cg-recents-toggle",
            icon=(
                ":material/keyboard_arrow_down:"
                if st.session_state.recents_expanded
                else ":material/keyboard_arrow_right:"
            ),
            help=(
                "Collapse recent projects"
                if st.session_state.recents_expanded
                else "Expand recent projects"
            ),
            type="tertiary",
            width="stretch",
            on_click=_toggle_recents,
        )
        if st.session_state.recents_expanded:
            projects = services.list_projects()
            if projects:
                for project in projects:
                    project_icon = (
                        ":material/folder_open:"
                        if project["project_type"]
                        else ":material/description:"
                    )
                    project_id = project["id"]
                    project_column, actions_column = st.columns([5, 1], gap="small")
                    project_column.button(
                        project["title"],
                        key=f"cg-recent-project-{project_id}",
                        icon=(
                            ":material/push_pin:"
                            if project["is_pinned"]
                            else project_icon
                        ),
                        help=(
                            f"Pinned project: {project['title']}"
                            if project["is_pinned"]
                            else project["title"]
                        ),
                        type=(
                            "primary"
                            if st.session_state.get("project_open")
                            and st.session_state.get("active_project_id") == project_id
                            else "tertiary"
                        ),
                        width="stretch",
                        on_click=_open_recent_project,
                        args=(project_id,),
                    )
                    actions_column.button(
                        "",
                        key=f"cg-project-menu-{project_id}",
                        icon=":material/more_vert:",
                        help="Project actions (right-click the project for this menu)",
                        type="tertiary",
                        width="stretch",
                        on_click=_toggle_project_actions,
                        args=(project_id,),
                    )

                    if st.session_state.get("cg_project_action_id") == project_id:
                        if st.session_state.get("cg_rename_project_id") == project_id:
                            with st.form(f"cg-rename-form-{project_id}"):
                                new_title = st.text_input(
                                    "Project name",
                                    value=project["title"],
                                    max_chars=120,
                                    key=f"cg-rename-input-{project_id}",
                                )
                                rename_column, cancel_column = st.columns(2)
                                save_rename = rename_column.form_submit_button(
                                    "Save rename", width="stretch"
                                )
                                cancel_rename = cancel_column.form_submit_button(
                                    "Cancel", width="stretch"
                                )
                            if save_rename:
                                if new_title is None:
                                    st.error("Project name cannot be empty.")
                                else:
                                    try:
                                        services.rename_project(project_id, new_title)
                                        st.session_state.cg_rename_project_id = None
                                        st.session_state.cg_project_action_id = None
                                        st.rerun()
                                    except ValidationError as exc:
                                        st.error(str(exc))
                            if cancel_rename:
                                st.session_state.cg_rename_project_id = None
                                st.rerun()
                        elif (
                            st.session_state.get("cg_confirm_delete_project_id")
                            == project_id
                        ):
                            st.warning(
                                f"Delete “{project['title']}”? This permanently removes "
                                "its code, requirements, tests, and reports."
                            )
                            confirm_column, cancel_column = st.columns([3, 2])
                            if confirm_column.button(
                                "Confirm delete",
                                key=f"cg-confirm-delete-{project_id}",
                                icon=":material/delete_forever:",
                                type="primary",
                                width="stretch",
                            ):
                                _delete_project(project_id)
                            cancel_column.button(
                                "Cancel",
                                key=f"cg-cancel-delete-{project_id}",
                                width="stretch",
                                on_click=_cancel_project_deletion,
                            )
                        else:
                            st.button(
                                "Rename",
                                key=f"cg-rename-project-{project_id}",
                                icon=":material/edit:",
                                type="tertiary",
                                width="stretch",
                                on_click=_request_project_rename,
                                args=(project_id,),
                            )
                            st.button(
                                "Unpin" if project["is_pinned"] else "Pin",
                                key=f"cg-toggle-pin-{project_id}",
                                icon=(
                                    ":material/push_pin:"
                                    if not project["is_pinned"]
                                    else ":material/keep_off:"
                                ),
                                type="tertiary",
                                width="stretch",
                                on_click=_toggle_project_pin,
                                args=(project_id, not bool(project["is_pinned"])),
                            )
                            st.button(
                                "Delete",
                                key=f"cg-request-delete-{project_id}",
                                icon=":material/delete:",
                                type="tertiary",
                                width="stretch",
                                on_click=_request_project_deletion,
                                args=(project_id,),
                            )
            else:
                st.markdown(
                    '<div class="cg-recent-empty">No recent projects</div>',
                    unsafe_allow_html=True,
                )
            st.html(
                """
                <script>
                if (!window.__codeguardProjectContextMenu) {
                  window.__codeguardProjectContextMenu = true;
                  document.addEventListener("contextmenu", (event) => {
                    const row = event.target.closest('[class*="st-key-cg-recent-project-"]');
                    if (!row) return;
                    const rowClass = Array.from(row.classList).find((name) =>
                      name.startsWith("st-key-cg-recent-project-")
                    );
                    const projectId = rowClass?.replace("st-key-cg-recent-project-", "");
                    if (!projectId || !/^\\d+$/.test(projectId)) return;
                    event.preventDefault();
                    document.querySelector(
                      `.st-key-cg-project-menu-${projectId} button`
                    )?.click();
                  }, true);
                }
                </script>
                """,
                unsafe_allow_javascript=True,
            )
    return st.session_state.cg_navigation


def _open_recent_project(project_id: int) -> None:
    st.session_state.active_project_id = project_id
    st.session_state.project_open = True
    st.session_state.cg_navigation = "New Project"
    st.session_state.cg_home_open = False
    st.session_state.cg_project_view = "overview"


def _toggle_project_actions(project_id: int) -> None:
    action_key = "cg_project_action_id"
    st.session_state[action_key] = (
        None if st.session_state.get(action_key) == project_id else project_id
    )
    st.session_state.cg_confirm_delete_project_id = None
    st.session_state.cg_rename_project_id = None


def _request_project_rename(project_id: int) -> None:
    st.session_state.cg_rename_project_id = project_id
    st.session_state.cg_confirm_delete_project_id = None


def _toggle_project_pin(project_id: int, pinned: bool) -> None:
    try:
        services.set_project_pinned(project_id, pinned)
        st.session_state.cg_project_action_id = None
    except ValidationError as exc:
        st.error(str(exc))


def _request_project_deletion(project_id: int) -> None:
    st.session_state.cg_confirm_delete_project_id = project_id
    st.session_state.cg_rename_project_id = None


def _cancel_project_deletion() -> None:
    st.session_state.cg_project_action_id = None
    st.session_state.cg_confirm_delete_project_id = None
    st.session_state.cg_rename_project_id = None


def _delete_project(project_id: int) -> None:
    try:
        project = services.get_project(project_id)
        if project is None:
            raise ValidationError("Project no longer exists.")
        services.delete_project(project_id)
    except ValidationError as exc:
        st.error(str(exc))
        return
    except Exception as exc:
        _show_error(exc)
        return

    st.session_state.cg_project_action_id = None
    st.session_state.cg_confirm_delete_project_id = None
    st.session_state.cg_rename_project_id = None
    st.session_state.cg_project_deleted_message = (
        f"Deleted project “{project['title']}”."
    )
    if st.session_state.get("active_project_id") == project_id:
        st.session_state.project_open = False
        st.session_state.active_project_id = None
        st.session_state.cg_project_view = "overview"
        st.session_state.cg_navigation = "New Project"
        st.session_state.cg_home_open = False
    st.rerun()


def render_app() -> None:
    initialize_database()
    _load_styles()
    page = _render_sidebar()
    deleted_message = st.session_state.pop("cg_project_deleted_message", None)
    if deleted_message:
        st.success(deleted_message)
    active_id = st.session_state.get("active_project_id")
    project_open = bool(st.session_state.get("project_open", False))

    if st.session_state.cg_home_open:
        _render_homepage()
        return

    if project_open and active_id and page == "New Project":
        project = services.get_project(active_id)
        if project and not project["is_demo"]:
            _render_project(project)
            return

    if page == "New Project":
        if project_open and active_id:
            project = services.get_project(active_id)
            if project and not project["is_demo"]:
                _render_project(project)
                return
        _render_new_project()
    elif page == "Demo":
        if project_open and active_id:
            project = services.get_project(active_id)
            if project and project["is_demo"]:
                _render_project(project)
            else:
                _render_demo_landing()
        else:
            _render_demo_landing()


def _activate_project(project_id: int) -> None:
    st.session_state.active_project_id = project_id
    st.session_state.project_open = True
    st.session_state.cg_navigation = "Demo"
    st.session_state.cg_home_open = False
    st.session_state.cg_project_view = "overview"
    st.rerun()


def _close_project(destination: str) -> None:
    st.session_state.project_open = False
    st.session_state.cg_navigation = destination
    st.session_state.cg_home_open = False


def _home_icon(name: str) -> str:
    drawings = {
        "requirements": '<path d="M8 3h8l4 4v14H4V3z"/><path d="M16 3v5h5M8 13h8M8 17h6"/>',
        "tests": '<path d="m9 12 2 2 4-4"/><path d="M12 3 5 6v5c0 5 3 8 7 10 4-2 7-5 7-10V6z"/>',
        "code": '<path d="m8 8-4 4 4 4M16 8l4 4-4 4M14 5l-4 14"/>',
        "execute": '<path d="M5 4v16l14-8z"/>',
        "bug": '<path d="M9 8h6a3 3 0 0 1 3 3v6a6 6 0 0 1-12 0v-6a3 3 0 0 1 3-3z"/><path d="M10 8V5a2 2 0 0 1 4 0v3M5 12H3m18 0h-2M5 17H3m18 0h-2M7 6 5 4m12 2 2-2M12 12v5"/>',
        "improve": '<path d="M20 7v5h-5"/><path d="M19 12a7 7 0 1 0 1 4"/>',
        "sparkles": '<path d="m12 3 1.7 5.3L19 10l-5.3 1.7L12 17l-1.7-5.3L5 10l5.3-1.7z"/><path d="m19 14 .8 2.2L22 17l-2.2.8L19 20l-.8-2.2L16 17l2.2-.8z"/>',
        "clipboard": '<rect x="5" y="5" width="14" height="16" rx="2"/><path d="M9 5V3h6v2M8 11h8M8 15h8"/>',
        "container": '<path d="M4 7h16v13H4z"/><path d="M4 11h16M8 7V4h8v3M8 14h2m4 0h2"/>',
        "search": '<circle cx="10.5" cy="10.5" r="6.5"/><path d="m16 16 4 4M8 11l2 2 4-4"/>',
        "wrench": '<path d="M14 6a5 5 0 0 0-6 6l-5 5a2 2 0 0 0 3 3l5-5a5 5 0 0 0 6-6l-3 3-3-3z"/>',
        "coverage": '<circle cx="12" cy="12" r="9"/><path d="m8 12 3 3 5-6"/>',
        "history": '<path d="M3 12a9 9 0 1 0 2.6-6.4L3 8"/><path d="M3 3v5h5m4-1v5l4 2"/>',
        "chart": '<path d="M4 19V5m0 14h17"/><path d="m7 15 4-4 3 2 5-6"/>',
        "report": '<path d="M6 3h9l4 4v14H6z"/><path d="M14 3v5h5M9 13h7M9 17h7"/>',
        "arrow": '<path d="M5 12h14m-6-6 6 6-6 6"/>',
        "check": '<path d="m5 12 4 4L19 6"/>',
    }
    return (
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        'stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" '
        'aria-hidden="true">'
        f'{drawings[name]}</svg>'
    )


def _render_homepage() -> None:
    homepage_markup = f"""
        <div class="cg-home">
          <section class="cg-home-hero">
            <div class="cg-home-hero-copy">
              <div class="cg-home-eyebrow">{_home_icon("sparkles")} CODEGUARD AI</div>
              <h1>Test Smart.<br><span>Detect Early.</span><br>Code Better.</h1>
              <p>An intelligent software testing platform that transforms project requirements into test cases, executes tests against your code, identifies bugs, and helps improve software quality with AI-powered analysis.</p>
            </div>
            <div class="cg-home-visual" role="img" aria-label="Illustration of test execution, passing and failing tests, and an AI bug analysis">
              <div class="cg-editor">
                <div class="cg-editor-bar"><span></span><span></span><span></span><b>calculator.py</b><i>● Python</i></div>
                <div class="cg-editor-code"><small>01</small><code><em>def</em> <strong>calculate_total</strong>(items):</code><small>02</small><code>    total = <em>sum</em>(items)</code><small>03</small><code>    <em>return</em> round(total, 2)</code><small>04</small><code></code><small>05</small><code><em>def</em> <strong>apply_discount</strong>(total, rate):</code><small>06</small><code>    <em>return</em> total * (1 - rate)</code></div>
                <div class="cg-editor-footer"><span>{_home_icon("execute")} pytest</span><span class="cg-sample-status">{_home_icon("check")} Execution complete</span></div>
              </div>
              <div class="cg-result-card cg-result-tests"><div class="cg-result-heading">{_home_icon("tests")} Test results <span>Sample</span></div><div class="cg-result-row"><i class="cg-dot-pass"></i> 08 passed</div><div class="cg-result-row"><i class="cg-dot-fail"></i> 01 needs review</div></div>
              <div class="cg-result-card cg-result-ai"><div class="cg-ai-icon">{_home_icon("sparkles")}</div><div><b>AI failure analysis</b><span>Possible edge case detected</span></div><span class="cg-ai-check">{_home_icon("check")}</span></div>
            </div>
          </section>
          <section class="cg-home-section cg-home-workflow">
            <div class="cg-home-section-heading"><span class="cg-home-eyebrow">A CLEAR PATH TO QUALITY</span><h2>From Requirements to Reliable Code</h2><p>One connected workflow from your first requirement to a retested improvement.</p></div>
            <div class="cg-workflow-grid">
              <article class="cg-workflow-card"><div class="cg-step-top"><span>01</span>{_home_icon("requirements")}</div><h3>Define Requirements</h3><p>Describe the project, objectives, expected inputs and outputs, and functional requirements.</p></article>
              <article class="cg-workflow-card"><div class="cg-step-top"><span>02</span>{_home_icon("tests")}</div><h3>Generate Test Cases</h3><p>Draft tests with AI or create them manually, then review, edit, and approve them.</p></article>
              <article class="cg-workflow-card"><div class="cg-step-top"><span>03</span>{_home_icon("code")}</div><h3>Add Your Code</h3><p>Write Python in the coding workspace or upload an existing Python file.</p></article>
              <article class="cg-workflow-card"><div class="cg-step-top"><span>04</span>{_home_icon("execute")}</div><h3>Execute Tests</h3><p>Run this project's saved test cases with Pytest in an isolated Docker environment and inspect the results.</p></article>
              <article class="cg-workflow-card"><div class="cg-step-top"><span>05</span>{_home_icon("bug")}</div><h3>Analyse Bugs</h3><p>Review actual test failures with AI-assisted explanations and possible causes.</p></article>
              <article class="cg-workflow-card"><div class="cg-step-top"><span>06</span>{_home_icon("improve")}</div><h3>Improve and Retest</h3><p>Review suggested corrections, approve or reject changes, save versions, and run tests again.</p></article>
            </div>
          </section>

          <section class="cg-home-section">
            <div class="cg-home-section-heading"><span class="cg-home-eyebrow">BUILT FOR YOUR WORKFLOW</span><h2>Everything You Need for Smarter Testing</h2><p>Bring test design, execution, code review, and quality tracking together.</p></div>
            <div class="cg-feature-grid">
              <article class="cg-feature-card">{_home_icon("sparkles")}<h3>AI-Assisted Test Case Generation</h3><p>Generate draft test cases from project requirements with your configured local Ollama model.</p></article>
              <article class="cg-feature-card">{_home_icon("clipboard")}<h3>Manual Test Case Management</h3><p>Create, edit, review, approve, or reject test cases before running them.</p></article>
              <article class="cg-feature-card">{_home_icon("container")}<h3>Isolated Test Execution</h3><p>Run saved Python test cases with Pytest inside the Docker sandbox.</p></article>
              <article class="cg-feature-card">{_home_icon("search")}<h3>Intelligent Bug Analysis</h3><p>Ask local AI to explain actual failed test output and likely causes.</p></article>
              <article class="cg-feature-card">{_home_icon("wrench")}<h3>Suggested Code Corrections</h3><p>Review AI-proposed corrections and approve or reject them before applying.</p></article>
              <article class="cg-feature-card">{_home_icon("coverage")}<h3>Requirement Coverage Tracking</h3><p>Track which requirements have linked tests; linkage does not imply a passing test.</p></article>
              <article class="cg-feature-card">{_home_icon("history")}<h3>Code Version History</h3><p>Save code versions and inspect changes associated with corrections and retests.</p></article>
              <article class="cg-feature-card">{_home_icon("chart")}<h3>Code Coverage Analysis</h3><p>View measured line coverage when it is available from a test execution.</p></article>
              <article class="cg-feature-card">{_home_icon("report")}<h3>Detailed Testing Reports</h3><p>Export project test and execution data as PDF, CSV, or JSON.</p></article>
            </div>
          </section>

          <section class="cg-home-section cg-home-preview">
            <div class="cg-home-section-heading"><span class="cg-home-eyebrow">A LOOK INSIDE</span><h2>See Your Test Results at a Glance</h2><p>Execution output brings test status and failure details into one reviewable workspace.</p></div>
            <div class="cg-preview-board">
              <div class="cg-preview-top"><div><span class="cg-preview-label">ILLUSTRATIVE EXAMPLE</span><h3>Test execution summary</h3></div><span class="cg-preview-status">{_home_icon("check")} Completed</span></div>
              <div class="cg-preview-metrics"><div><span>Total tests</span><b>09</b></div><div><span>Passed</span><b class="cg-metric-pass">08</b></div><div><span>Failed</span><b class="cg-metric-fail">01</b></div><div><span>Execution</span><b class="cg-metric-ready">Complete</b></div></div>
              <div class="cg-preview-details"><article><div class="cg-preview-detail-title">{_home_icon("bug")} Example failure analysis</div><p><code>test_discount_boundary</code> returned an unexpected value at the discount limit. AI-assisted analysis can help investigate the failure and related code.</p></article><article><div class="cg-preview-detail-title">{_home_icon("wrench")} Example suggested correction</div><p>A proposed boundary-condition change can be reviewed and approved or rejected before it becomes a saved code version.</p></article></div>
              <div class="cg-preview-note">Illustrative sample values only — not results from your project.</div>
            </div>
          </section>

          <section class="cg-home-section cg-home-start">
            <div class="cg-home-section-heading"><span class="cg-home-eyebrow">GET STARTED</span><h2>Start Testing in Simple Steps</h2><p>Set up a project, approve its tests, then run and review them in one workspace.</p></div>
            <div class="cg-start-list"><div><span>01</span>Create a project</div><div><span>02</span>Enter project requirements</div><div><span>03</span>Generate or create test cases</div><div><span>04</span>Review and approve test cases</div><div><span>05</span>Add your Python code</div><div><span>06</span>Run tests and analyse results</div></div>
            <div class="cg-home-start-action" id="cg-home-start-slot"></div>
          </section>

          <section class="cg-home-closing"><div class="cg-home-eyebrow">CODEGUARD AI</div><h2>Write Better Code. Build with Confidence.</h2><p>Bring requirements, software testing, bug analysis, and code improvement together in one intelligent workspace.</p><div id="cg-home-closing-slot"></div></section>
          <footer class="cg-home-footer">CodeGuard AI <span>|</span> Test Smart. Detect Early. Code Better.</footer>
        </div>
        """
    homepage_markup = re.sub(r"(?m)^[ \t]+", "", homepage_markup)
    hero_markup, sections_markup = homepage_markup.split(
        '<section class="cg-home-section cg-home-workflow">', maxsplit=1
    )
    st.html(hero_markup + "</div>")
    hero_actions = st.columns([1, 1, 2.2])
    if hero_actions[0].button(
        "Create New Project",
        key="cg-home-create-project",
        type="primary",
        icon=":material/add:",
        width="stretch",
    ):
        _select_navigation("New Project")
        st.rerun()
    if hero_actions[1].button(
        "Explore Demo",
        key="cg-home-explore-demo",
        type="secondary",
        icon=":material/play_arrow:",
        width="stretch",
    ):
        _select_navigation("Demo")
        st.rerun()

    start_marker = (
        '<div class="cg-home-start-action" id="cg-home-start-slot"></div>\n'
        "</section>"
    )
    before_start, after_start = sections_markup.split(start_marker, maxsplit=1)
    st.html(
        '<div class="cg-home"><section class="cg-home-section cg-home-workflow">'
        + before_start
        + "</section></div>",
    )
    if st.button(
        "Start Your First Project",
        key="cg-home-start-project",
        type="primary",
        icon=":material/arrow_forward:",
    ):
        _select_navigation("New Project")
        st.rerun()

    closing_marker = '<div id="cg-home-closing-slot"></div></section>'
    before_closing, after_closing = after_start.split(closing_marker, maxsplit=1)
    st.html(
        '<div class="cg-home">' + before_closing + "</section></div>",
    )
    if st.button(
        "Create New Project",
        key="cg-home-closing-project",
        type="primary",
        icon=":material/arrow_forward:",
    ):
        _select_navigation("New Project")
        st.rerun()
        st.html(
            '<div class="cg-home">'
            + after_closing.replace("</div>", "", 1)
            + "</div>",
        )


def _render_new_project() -> None:
    flash = st.session_state.pop("flash_message", "")
    if flash:
        st.success(flash)
    with st.container(border=True):
        hero_logo, hero_copy = st.columns([1, 8], vertical_alignment="center")
        with hero_logo:
            st.markdown(_logo_markup(64), unsafe_allow_html=True)
        with hero_copy:
            st.markdown('<div class="cg-kicker">Test Smart. Detect Early. Code Better.</div>',
                        unsafe_allow_html=True)
            st.title("Start with your requirements")
            st.caption(
                "Create a Python testing workspace, then review requirements, tests, and code versions together."
            )
    st.markdown('<div class="cg-section-label">Create a new project</div>', unsafe_allow_html=True)
    code_mode = st.radio(
        "Starting code",
        ["Write Code Manually", "Upload Existing Code"],
        horizontal=True,
    )
    uploaded = None
    if code_mode == "Upload Existing Code":
        uploaded = st.file_uploader(
            "Python source file (.py, maximum 200 KB)", type=["py"]
        )
    with st.form("create_project_form", clear_on_submit=False):
        title = st.text_input("Project title", max_chars=120)
        description_col, objectives_col = st.columns(2)
        with description_col:
            description = st.text_area("Description", height=68)
        with objectives_col:
            objectives = st.text_area("Objectives", height=68)
        left, right = st.columns(2)
        with left:
            language = st.selectbox("Programming language", ["Python"])
            project_type = st.text_input("Project type", placeholder="For example: CLI utility")
            features = st.text_area("Main features", height=68)
        with right:
            expected_inputs = st.text_area("Expected inputs", height=68)
            expected_outputs = st.text_area("Expected outputs", height=68)
            constraints = st.text_area("Additional constraints", height=68)
        requirement_text = st.text_area(
            "Functional requirements",
            height=130,
            help="Enter one requirement per line. IDs such as REQ-001 are assigned automatically.",
        )
        manual_source = ""
        if code_mode == "Write Code Manually":
            manual_source = st.text_area(
                "Python source code",
                height=220,
                help=(
                    "Optional. If supplied, CodeGuard will analyze it and start "
                    "Qwen 2.5 3B test generation after you submit requirements."
                ),
            )
        submitted = st.form_submit_button("Submit Requirements", type="primary")
    if not submitted:
        return
    filename = "main.py"
    code = manual_source
    kind = "manual"
    if code_mode == "Upload Existing Code":
        if uploaded is None:
            st.error("Choose a .py file to upload.")
            return
        filename = Path(uploaded.name).name
        raw = uploaded.getvalue()
        if (
            Path(filename).suffix.lower() != ".py"
            or len(raw) > MAX_UPLOAD_BYTES
            or not raw
            or b"\0" in raw
        ):
            st.error("Upload a non-empty .py file no larger than 200 KB without binary content.")
            return
        try:
            code = raw.decode("utf-8")
            compile(code, filename, "exec")
        except UnicodeDecodeError:
            st.error("Uploaded code must be UTF-8 text.")
            return
        except SyntaxError as exc:
            st.error(f"Uploaded Python has a syntax error on line {exc.lineno}: {exc.msg}")
            return
        kind = "original-upload"
    if not title.strip():
        st.error("Project title is required.")
        return
    try:
        project_id = services.create_project(
            title=title,
            description=description,
            objectives=objectives,
            language=language,
            project_type=project_type,
            main_features=features,
            expected_inputs=expected_inputs,
            expected_outputs=expected_outputs,
            constraints=constraints,
            requirements=requirement_text,
            filename=filename,
            code=code,
            code_kind=kind,
        )
    except (ValidationError, ValueError) as exc:
        st.error(str(exc))
        return
    st.session_state.active_project_id = project_id
    st.session_state.project_open = True
    st.session_state.cg_navigation = "New Project"
    st.session_state.cg_home_open = False
    st.session_state.cg_project_view = "workspace"
    st.rerun()


def _create_demo() -> int:
    existing = next(
        (project for project in services.list_projects(include_demo=True)
         if project["is_demo"]),
        None,
    )
    if existing:
        return existing["id"]
    code = (
        "def add(a, b):\n"
        "    return a + b\n\n"
        "def divide(a, b):\n"
        "    if b == 0:\n"
        "        raise ValueError('division by zero')\n"
        "    return a / b\n"
    )
    project_id = services.create_project(
        title="CodeGuard AI Sample Calculator",
        description="A clearly labeled sample project used to demonstrate the CodeGuard workflow.",
        objectives="Demonstrate requirement mapping and real isolated pytest execution.",
        project_type="Demo · calculator",
        main_features="Addition and guarded division",
        expected_inputs="Two numeric arguments",
        expected_outputs="Numeric result",
        constraints="Division by zero raises ValueError.",
        requirements=[
            "The calculator should support adding two numbers.",
            "Division should return the quotient for non-zero divisors.",
            "Division by zero should raise ValueError.",
        ],
        code=code,
        code_kind="demo-initial",
        is_demo=True,
    )
    cases = [
        {
            "name": "Add positive numbers", "description": "Adds two positive integers.",
            "requirement_id": "REQ-001", "category": "normal",
            "function_name": "add", "inputs": [2, 3], "expected": 5,
            "priority": "High",
        },
        {
            "name": "Add across zero", "description": "Adds a negative and a positive integer.",
            "requirement_id": "REQ-001", "category": "boundary",
            "function_name": "add", "inputs": [-2, 2], "expected": 0,
            "priority": "Medium",
        },
        {
            "name": "Divide non-zero values", "description": "Returns a real quotient.",
            "requirement_id": "REQ-002", "category": "normal",
            "function_name": "divide", "inputs": [8, 2], "expected": 4,
            "priority": "High",
        },
        {
            "name": "Reject zero divisor", "description": "Raises the documented exception.",
            "requirement_id": "REQ-003", "category": "exception",
            "function_name": "divide", "inputs": [8, 0], "expected": None,
            "expected_exception": "ValueError", "priority": "High",
        },
    ]
    for case in cases:
        case_id = services.add_test_case(project_id, case)
        test = next(item for item in services.list_test_cases(project_id)
                    if item["test_id"] == case_id)
        services.set_test_status(project_id, test["id"], "approved")
    return project_id


def _render_demo_landing() -> None:
    st.markdown('<div class="cg-kicker">Isolated sample workspace</div>', unsafe_allow_html=True)
    st.title("Demo")
    st.write(
        "Open a separate calculator project with sample requirements and approved tests. "
        "Its records are labeled as demo data and excluded from Recents."
    )
    if st.button("Open calculator demo", type="primary"):
        try:
            _activate_project(_create_demo())
        except Exception as exc:
            _show_error(exc)


def _render_project(project: dict) -> None:
    destination = "Demo" if project["is_demo"] else "New Project"
    project_view = st.session_state.setdefault(
        "cg_project_view", "overview"
    )
    header_left, header_actions = st.columns([3, 2], vertical_alignment="center")
    with header_left:
        st.markdown(
            '<div class="cg-kicker">Demo data · isolated sample</div>'
            if project["is_demo"]
            else '<div class="cg-kicker">Project</div>',
            unsafe_allow_html=True,
        )
        st.title(project["title"])
        st.caption(project["description"] or project["project_type"] or "Python project")
    with header_actions:
        st.button(
            "Project Overview",
            key=f"project-overview-{project['id']}",
            type="primary" if project_view == "overview" else "secondary",
            icon=":material/dashboard:",
            width="stretch",
            on_click=_set_project_view,
            args=("overview",),
        )
        st.button(
            "Coding Workspace",
            key=f"open-workspace-{project['id']}",
            type="primary" if project_view == "workspace" else "secondary",
            icon=":material/code:",
            width="stretch",
            on_click=_set_project_view,
            args=("workspace",),
        )
    st.button(
        "Back to Demo" if project["is_demo"] else "Back to Projects",
        key=f"back-{project['id']}",
        on_click=_close_project,
        args=(destination,),
    )
    st.divider()
    if project_view == "workspace":
        _render_workspace(project)
    else:
        _render_project_overview(project)


def _set_project_view(view: str) -> None:
    if view not in {"overview", "workspace"}:
        raise ValueError(f"Unknown project view: {view}")
    st.session_state.cg_project_view = view


def _render_project_overview(project: dict) -> None:
    project_id = project["id"]
    requirements = services.list_requirements(project_id)
    tests = services.list_test_cases(project_id)
    versions = services.list_code_versions(project_id)
    executions = services.list_executions(project_id)
    analyses = services.list_analyses(project_id)
    corrections = services.list_corrections(project_id)
    original = services.get_original_code(project_id)
    latest = services.get_execution(executions[0]["id"]) if executions else None

    st.subheader("Project overview")
    st.caption(
        f"Created {project['created_at']} · Last updated {project['updated_at']}"
    )
    details_left, details_right = st.columns(2)
    with details_left:
        st.markdown("**Description**")
        st.write(project["description"] or "No description provided.")
        st.markdown("**Objectives**")
        st.write(project["objectives"] or "No objectives provided.")
        st.markdown("**Programming language**")
        st.write(project["language"])
        st.markdown("**Project type**")
        st.write(project["project_type"] or "Not specified")
    with details_right:
        st.markdown("**Main features**")
        st.write(project["main_features"] or "No main features provided.")
        st.markdown("**Expected inputs**")
        st.write(project["expected_inputs"] or "Not specified")
        st.markdown("**Expected outputs**")
        st.write(project["expected_outputs"] or "Not specified")
        st.markdown("**Additional constraints**")
        st.write(project["constraints"] or "None specified")

    st.subheader("Functional requirements")
    if requirements:
        for requirement in requirements:
            st.markdown(
                f"**{requirement['id']}** · {requirement['description']}"
            )
    else:
        st.info("No functional requirements have been added.")
    complexity = analyze_requirements(requirements)
    st.subheader("Requirements analysis")
    complexity_columns = st.columns(4)
    for column, label, value in zip(
        complexity_columns,
        ("Total", "Basic", "Intermediate", "Advanced"),
        (
            complexity["total"],
            complexity["basic"],
            complexity["intermediate"],
            complexity["advanced"],
        ),
    ):
        column.metric(label, value)
    if complexity["advanced_warning"]:
        st.warning(
            "Advanced testing requirements have been identified.\n\n"
            "The currently configured Qwen 2.5 3B model is being used for basic "
            "and intermediate test-case generation. Some advanced scenarios may "
            "exceed its reliable capabilities.\n\n"
            "For more comprehensive advanced test-case generation, we recommend "
            "using a higher-capability LLM.\n\n"
            "Advanced LLM integration is not currently available in CodeGuard AI."
        )
        st.markdown("**Advanced scenarios pending**")
        for item in complexity["advanced_requirements"]:
            st.markdown(
                f"- **{item['id']}** · {item['description']}  \n"
                f"Reason: {item['reason']}"
            )

    st.subheader("Source code")
    if original:
        st.write(f"Source file: **{original['filename']}**")
        st.caption(
            f"Original source: {len(original['original_content'].splitlines())} lines · "
            f"{len(original['original_content'].encode('utf-8')):,} bytes"
        )
        with st.expander("View original Python source"):
            st.code(original["original_content"], language="python")
    if versions:
        st.caption(
            f"Current version: v{versions[0]['version_number']} "
            f"({versions[0]['kind']}, saved {versions[0]['created_at']})"
        )
        with st.expander("View current Python source"):
            st.code(versions[0]["content"], language="python")
    else:
        st.info("No Python source version has been saved yet.")
    if versions:
        st.subheader("Code version history")
        st.dataframe(
            pd.DataFrame([
                {
                    "Version": f"v{item['version_number']}",
                    "Type": item["kind"],
                    "Saved": item["created_at"],
                    "Lines": len(item["content"].splitlines()),
                    "Bytes": len(item["content"].encode("utf-8")),
                }
                for item in versions
            ]),
            width="stretch",
            hide_index=True,
        )

    st.subheader("Test cases")
    test_metrics = st.columns(3)
    test_metrics[0].metric("Total test cases", len(tests))
    test_metrics[1].metric(
        "Approved",
        sum(item["approval_status"] == "approved" for item in tests),
    )
    test_metrics[2].metric(
        "Pending review",
        sum(item["approval_status"] == "pending" for item in tests),
    )
    if tests:
        latest_results = {
            item["test_case_id"]: item for item in (latest or {}).get("results", [])
        }
        requirement_complexity = {
            item["id"]: item["complexity"]
            for item in complexity["requirements"]
        }
        test_rows = []
        for item in tests:
            result = latest_results.get(item["id"])
            target = (
                f"{item['target_class']}."
                if item.get("target_class") else ""
            ) + item["function_name"]
            test_rows.append({
                "Test ID": item["test_id"],
                "Name": item["name"],
                "Description": item["description"],
                "Testing objective": item.get("testing_objective", ""),
                "Target": target,
                "Requirement": item["requirement_id"] or "Unlinked",
                "Category": item["category"],
                "Complexity": item.get(
                    "complexity",
                    requirement_complexity.get(
                        item["requirement_id"], "Unclassified"
                    ),
                ),
                "Priority": item["priority"],
                "Approval": item["approval_status"],
                "Validation": (
                    "Validated at approval"
                    if item["approval_status"] == "approved"
                    else "Pending review"
                ),
                "Generated model": (
                    "Qwen 2.5 3B" if item["generation_source"] == "ollama"
                    else item["generation_source"]
                ),
                "Latest result": result["status"] if result else "Not run",
            })
        st.dataframe(pd.DataFrame(test_rows), width="stretch", hide_index=True)
        for item in tests:
            with st.expander(f"{item['test_id']} · {item['name']} · full details"):
                st.write(item["description"] or "No description provided.")
                st.caption(
                    f"Target: {(item['target_class'] + '.') if item.get('target_class') else ''}"
                    f"{item['function_name']} ({item['target_kind']}) · "
                    f"Generated by: {item['generation_source']} · "
                    f"Expected exception: {item['expected_exception'] or 'None'}"
                )
                st.markdown("**Positional inputs**")
                st.code(item["inputs_json"], language="json")
                st.markdown("**Keyword inputs**")
                st.code(item.get("keyword_inputs_json", "{}"), language="json")
                if item.get("constructor_inputs_json", "[]") != "[]" or item.get(
                    "constructor_keyword_inputs_json", "{}"
                ) != "{}":
                    st.markdown("**Constructor inputs**")
                    st.code(item["constructor_inputs_json"], language="json")
                    st.code(
                        item.get("constructor_keyword_inputs_json", "{}"),
                        language="json",
                    )
                if item.get("setup_json", "[]") != "[]":
                    st.markdown("**Setup steps**")
                    st.code(item["setup_json"], language="json")
                st.markdown("**Expected result**")
                st.code(item["expected_json"], language="json")
                if item.get("requirement_description"):
                    st.markdown("**Linked requirement**")
                    st.write(item["requirement_description"])
                if item.get("preconditions"):
                    st.markdown("**Preconditions**")
                    st.write(item["preconditions"])
                if item.get("database_assertions_json", "[]") != "[]":
                    st.markdown("**Database assertions**")
                    st.code(item["database_assertions_json"], language="json")

    if executions:
        if latest:
            st.subheader("Latest test execution")
            latest_metrics = st.columns(6)
            for column, label, value in zip(
                latest_metrics,
                ("Status", "Passed", "Failed", "Skipped", "Errors", "Duration"),
                (
                    latest["status"],
                    latest["passed"],
                    latest["failed"],
                    latest["skipped"],
                    latest["errors"],
                    f"{latest['duration']:.2f}s",
                ),
            ):
                column.metric(label, value)
            if latest["coverage"].get("available"):
                coverage = latest["coverage"]
                st.caption(
                    f"Measured Coverage.py line coverage: "
                    f"{coverage['percentage']:.1f}% "
                    f"({coverage['covered_lines']} covered, "
                    f"{coverage['missed_lines']} missed)."
                )
            else:
                st.info("Coverage.py data is unavailable for this execution.")
            if latest["results"]:
                st.dataframe(
                    pd.DataFrame([
                        {
                            "Test ID": item["test_id"],
                            "Test": item["name"],
                            "Result": item["status"],
                            "Details": item["message"],
                            "Duration (s)": item["duration"],
                        }
                        for item in latest["results"]
                    ]),
                    width="stretch",
                    hide_index=True,
                )
                for item in latest["results"]:
                    with st.expander(
                        f"{item['test_id']} · {item['name']} · "
                        f"{item['status']} details"
                    ):
                        st.write(item["message"] or "No additional details.")
                        if item["source_location"]:
                            st.caption(item["source_location"])
                        if item["traceback"]:
                            st.code(item["traceback"], language="text")
            else:
                st.info("No per-test results were recorded for this execution.")
            with st.expander("View latest Pytest output"):
                st.code(latest["output"] or "No execution output.", language="text")
        st.subheader("Execution history")
        st.dataframe(
            pd.DataFrame([
                {
                    "Execution": f"#{item['id']}",
                    "Created": item["created_at"],
                    "Code version": f"v{item['version_number']}",
                    "Status": item["status"],
                    "Passed": item["passed"],
                    "Failed": item["failed"],
                    "Errors": item["errors"],
                    "Skipped": item["skipped"],
                    "Duration (s)": item["duration"],
                }
                for item in executions
            ]),
            width="stretch",
            hide_index=True,
        )

    if analyses:
        st.subheader("AI failure analyses")
        for item in analyses:
            with st.expander(
                f"Execution #{item['execution_id']} · {item['model']} · "
                f"{item['created_at']}"
            ):
                st.markdown(item["content"])

    if corrections:
        st.subheader("Suggested correction history")
        for item in corrections:
            with st.expander(
                f"Correction #{item['id']} · {item['status']} · "
                f"Execution #{item['execution_id']}"
            ):
                st.write(item["explanation"])
                st.code(item["proposed_code"], language="python")

    if requirements:
        coverage = services.requirement_coverage(project_id)
        st.subheader("Requirement coverage")
        st.progress(coverage["linked"] / coverage["total"])
        st.caption(
            f"{coverage['linked']} of {coverage['total']} requirements have linked tests. "
            "This measures test linkage, not execution success."
        )

    if versions or executions:
        st.subheader("Project activity")
        activity = []
        if versions:
            activity.append(
                f"Latest code version: v{versions[0]['version_number']} "
                f"({versions[0]['kind']}, {versions[0]['created_at']})"
            )
        if executions:
            activity.append(
                f"Latest execution: #{executions[0]['id']} "
                f"({executions[0]['status']}, {executions[0]['created_at']})"
            )
        for entry in activity:
            st.write(entry)

    st.button(
        "Open Coding Workspace",
        key=f"overview-open-workspace-{project_id}",
        type="primary",
        icon=":material/code:",
        on_click=_set_project_view,
        args=("workspace",),
    )


def _render_requirements(project: dict) -> None:
    st.subheader("Functional requirements")
    requirements = services.list_requirements(project["id"])
    if requirements:
        st.dataframe(
            pd.DataFrame(requirements)[["id", "description", "created_at"]],
            width="stretch",
            hide_index=True,
        )
    else:
        st.info("No requirements have been added.")
    with st.form(f"add-requirement-{project['id']}"):
        description = st.text_area("Add a requirement", max_chars=2000)
        submitted = st.form_submit_button("Add requirement")
    if submitted:
        try:
            requirement_id = services.add_requirement(project["id"], description)
            generation_key = f"cg_initial_generation_status_{project['id']}"
            if st.session_state.get(generation_key) == "no_requirements":
                st.session_state[generation_key] = None
            st.success(f"Added {requirement_id}.")
            st.rerun()
        except ValidationError as exc:
            st.error(str(exc))
    coverage = services.requirement_coverage(project["id"])
    st.subheader("Requirement-to-test mapping")
    if coverage["mapping"]:
        for item in coverage["mapping"]:
            linked = item["tests"]
            with st.expander(f"{item['requirement_id']} · {item['description']}"):
                if not linked:
                    st.warning("No test cases are linked to this requirement.")
                else:
                    for test in linked:
                        st.write(
                            f"**{test['test_id']} {test['name']}** — "
                            f"execution: {test['execution_status']}"
                        )


def _manual_case_form(project: dict, requirements: list[dict]) -> None:
    st.subheader("Create a manual test")
    requirement_options = ["Unlinked"] + [item["id"] for item in requirements]
    with st.form(f"manual-test-{project['id']}"):
        name = st.text_input("Name")
        description = st.text_area("Description", max_chars=2000)
        testing_objective = st.text_area(
            "Testing objective",
            max_chars=1000,
            help="Explain the specific behavior or condition this case verifies.",
        )
        requirement = st.selectbox("Related requirement", requirement_options)
        function_name = st.text_input("Function or method name", value="solve")
        target_class = st.text_input("Target class (blank for module function)")
        target_kind = st.selectbox(
            "Callable kind",
            ["function", "instance_method", "class_method", "static_method"],
            format_func=lambda value: value.replace("_", " ").title(),
        )
        category = st.selectbox(
            "Category", CATEGORY_OPTIONS, format_func=_category_label
        )
        inputs_text = st.text_area("Input values as JSON array", value="[]")
        keyword_inputs_text = st.text_area(
            "Keyword arguments as JSON object", value="{}"
        )
        constructor_inputs_text = st.text_area(
            "Constructor arguments as JSON array", value="[]"
        )
        constructor_keyword_inputs_text = st.text_area(
            "Constructor keyword arguments as JSON object", value="{}"
        )
        setup_text = st.text_area(
            "Setup steps as JSON array",
            value="[]",
            help=(
                "Each step has method, inputs, keyword_inputs, and optional store_as. "
                "Reference stored returns with {\"$setup_ref\":\"name\"}."
            ),
        )
        expected_text = st.text_area("Expected output as JSON", value="null")
        database_assertions_text = st.text_area(
            "SQLite state assertions as JSON array", value="[]"
        )
        expected_exception = st.selectbox("Expected exception (optional)", EXCEPTION_OPTIONS)
        preconditions = st.text_input("Preconditions")
        priority = st.selectbox("Priority", PRIORITY_OPTIONS, index=1)
        submitted = st.form_submit_button("Add draft test")
    if submitted:
        try:
            case = {
                "name": name,
                "description": description,
                "testing_objective": testing_objective or description,
                "requirement_id": None if requirement == "Unlinked" else requirement,
                "function_name": function_name,
                "target_class": target_class,
                "target_kind": target_kind,
                "category": category,
                "inputs": _parse_json_text(inputs_text, "Input values"),
                "keyword_inputs": _parse_json_text(
                    keyword_inputs_text, "Keyword arguments"
                ),
                "constructor_inputs": _parse_json_text(
                    constructor_inputs_text, "Constructor arguments"
                ),
                "constructor_keyword_inputs": _parse_json_text(
                    constructor_keyword_inputs_text,
                    "Constructor keyword arguments",
                ),
                "setup": _parse_json_text(setup_text, "Setup steps"),
                "expected": _parse_json_text(expected_text, "Expected output"),
                "database_assertions": _parse_json_text(
                    database_assertions_text, "SQLite state assertions"
                ),
                "expected_exception": expected_exception,
                "preconditions": preconditions,
                "priority": priority,
                "generation_source": "manual",
            }
            test_id = services.add_test_case(project["id"], case)
            st.success(f"Created {test_id} as a draft for review.")
            st.rerun()
        except (json.JSONDecodeError, ValidationError) as exc:
            st.error(f"Could not add test case: {exc}")


def _render_test_cases(project: dict) -> None:
    requirements = services.list_requirements(project["id"])
    versions = services.list_code_versions(project["id"])
    source_code = versions[0]["content"] if versions else ""
    st.subheader("AI test generation")
    st.caption(
        f"Uses local {MODEL}. Test generation starts automatically after requirements "
        "are submitted with source code, or after code is added in the workspace."
    )
    if not requirements:
        st.info("Add a requirement before generating AI test cases.")
    _manual_case_form(project, requirements)
    st.subheader("Test cases")
    cases = services.list_test_cases(project["id"])
    if not cases:
        st.info("No test cases yet. Add one manually or generate cases with Ollama.")
        return
    for case in cases:
        with st.expander(f"{case['test_id']} · {case['name']}"):
            _render_test_editor(project, case, requirements)


def _render_test_editor(project: dict, case: dict, requirements: list[dict]) -> None:
    requirement_options = ["Unlinked"] + [item["id"] for item in requirements]
    current_requirement = case["requirement_id"] or "Unlinked"
    if current_requirement not in requirement_options:
        current_requirement = "Unlinked"
    with st.form(f"edit-test-{project['id']}-{case['id']}"):
        name = st.text_input("Name", value=case["name"], max_chars=160)
        description = st.text_area("Description", value=case["description"], max_chars=2000)
        testing_objective = st.text_area(
            "Testing objective",
            value=case.get("testing_objective", ""),
            max_chars=1000,
            key=f"objective-{case['id']}",
        )
        requirement = st.selectbox(
            "Related requirement", requirement_options,
            index=requirement_options.index(current_requirement),
            key=f"req-{case['id']}",
        )
        function_name = st.text_input("Function", value=case["function_name"])
        target_class = st.text_input(
            "Target class (blank for module function)",
            value=case.get("target_class", ""),
            key=f"target-class-{case['id']}",
        )
        current_target_kind = case.get(
            "target_kind", "instance_method" if target_class else "function"
        )
        target_kind_options = [
            "function", "instance_method", "class_method", "static_method"
        ]
        target_kind = st.selectbox(
            "Callable kind",
            target_kind_options,
            index=target_kind_options.index(current_target_kind),
            format_func=lambda value: value.replace("_", " ").title(),
            key=f"target-kind-{case['id']}",
        )
        category = st.selectbox(
            "Category",
            CATEGORY_OPTIONS,
            index=CATEGORY_OPTIONS.index(case["category"]),
            format_func=_category_label,
        )
        inputs_text = st.text_area("Inputs JSON", value=case["inputs_json"])
        keyword_inputs_text = st.text_area(
            "Keyword inputs JSON",
            value=case.get("keyword_inputs_json", "{}"),
            key=f"keyword-inputs-{case['id']}",
        )
        constructor_inputs_text = st.text_area(
            "Constructor arguments JSON",
            value=case.get("constructor_inputs_json", "[]"),
            key=f"constructor-inputs-{case['id']}",
        )
        constructor_keyword_inputs_text = st.text_area(
            "Constructor keyword arguments JSON",
            value=case.get("constructor_keyword_inputs_json", "{}"),
            key=f"constructor-keyword-inputs-{case['id']}",
        )
        setup_text = st.text_area(
            "Setup steps JSON",
            value=case.get("setup_json", "[]"),
            key=f"setup-{case['id']}",
        )
        expected_text = st.text_area("Expected JSON", value=case["expected_json"])
        database_assertions_text = st.text_area(
            "SQLite state assertions JSON",
            value=case.get("database_assertions_json", "[]"),
            key=f"database-assertions-{case['id']}",
        )
        exception_value = case["expected_exception"] or ""
        expected_exception = st.selectbox(
            "Expected exception", EXCEPTION_OPTIONS,
            index=EXCEPTION_OPTIONS.index(exception_value),
        )
        preconditions = st.text_input("Preconditions", value=case["preconditions"])
        priority = st.selectbox(
            "Priority", PRIORITY_OPTIONS,
            index=PRIORITY_OPTIONS.index(case["priority"]),
        )
        save = st.form_submit_button("Save edits")
    if save:
        try:
            services.update_test_case(
                project["id"], case["id"], {
                    "name": name, "description": description,
                    "testing_objective": testing_objective,
                    "requirement_id": None if requirement == "Unlinked" else requirement,
                    "function_name": function_name, "target_class": target_class,
                    "target_kind": target_kind, "category": category,
                    "inputs": _parse_json_text(inputs_text, "Input values"),
                    "keyword_inputs": _parse_json_text(
                        keyword_inputs_text, "Keyword arguments"
                    ),
                    "constructor_inputs": _parse_json_text(
                        constructor_inputs_text, "Constructor arguments"
                    ),
                    "constructor_keyword_inputs": _parse_json_text(
                        constructor_keyword_inputs_text,
                        "Constructor keyword arguments",
                    ),
                    "setup": _parse_json_text(setup_text, "Setup steps"),
                    "expected": _parse_json_text(expected_text, "Expected output"),
                    "database_assertions": _parse_json_text(
                        database_assertions_text, "SQLite state assertions"
                    ),
                    "expected_exception": expected_exception,
                    "preconditions": preconditions, "priority": priority,
                }, status="pending",
            )
            st.success("Test case updated and returned to pending review.")
            st.rerun()
        except (json.JSONDecodeError, ValidationError) as exc:
            st.error(f"Could not save test case: {exc}")
    target = (
        f"{case['target_class']}.{case['function_name']}"
        if case.get("target_class") else case["function_name"]
    )
    st.caption(
        f"Target: {target} · Source: {case['generation_source']} · "
        f"Requirement: {case['requirement_id'] or 'unlinked'}"
    )
    review_columns = st.columns(2)
    if case["approval_status"] != "approved" and review_columns[0].button(
        "Approve test",
        key=f"approve-test-{project['id']}-{case['id']}",
        disabled=not bool(services.list_code_versions(project["id"])),
    ):
        try:
            services.set_test_status(project["id"], case["id"], "approved")
            st.success("Test approved against the latest source version.")
            st.rerun()
        except ValidationError as exc:
            st.error(str(exc))
    if case["approval_status"] != "rejected" and review_columns[1].button(
        "Reject test",
        key=f"reject-test-{project['id']}-{case['id']}",
    ):
        try:
            services.set_test_status(project["id"], case["id"], "rejected")
            st.rerun()
        except ValidationError as exc:
            st.error(str(exc))
    if st.button(
        "Delete test case",
        icon=":material/delete:",
        help="Delete this test case immediately",
        key=f"delete-test-{project['id']}-{case['id']}",
    ):
        try:
            services.delete_test_case(project["id"], case["id"])
            st.rerun()
        except ValidationError as exc:
            st.error(str(exc))


def _workspace_source(project: dict, latest_version: dict | None) -> tuple[str, str]:
    original = services.get_original_code(project["id"])
    filename = original["filename"] if original else "main.py"
    return filename, latest_version["content"] if latest_version else ""


def _generate_workspace_tests(
    project: dict,
    requirements: list[dict],
    source_code: str,
    latest_version: dict | None,
) -> list[str]:
    project_id = project["id"]
    analysis = analyze_requirements(requirements)
    report_key = f"cg_generation_report_{project_id}"
    st.session_state[report_key] = {
        **analysis,
        "generated": 0,
        "duplicates_removed": 0,
        "rejected": 0,
        "model": MODEL,
        "executable": bool(source_code.strip()),
    }
    if not supported_requirements(analysis):
        raise ValidationError(
            "All requirements are classified as advanced. Qwen 2.5 3B can "
            "generate only basic and intermediate cases."
        )
    with st.status("Analyzing requirements…", expanded=True) as progress:
        progress.write("Analyzing requirements and available Python source.")
        _save_workspace_code(project, source_code, latest_version)
        progress.update(label="Detecting complexity…")
        progress.write(
            f"Classified {analysis['basic']} basic, "
            f"{analysis['intermediate']} intermediate, and "
            f"{analysis['advanced']} advanced requirements."
        )
        progress.update(label="Generating basic and intermediate test cases…")

        def report_generation_progress(message: str) -> None:
            progress.update(label=message)

        generated = generate_test_cases(
            project,
            requirements,
            source_code=source_code,
            progress_callback=report_generation_progress,
        )
        progress.update(label="Validating test cases…")
        unique_cases, duplicate_count = remove_duplicate_cases(generated)
        if not unique_cases:
            raise OllamaError(
                "Qwen returned only duplicate test cases; nothing was saved."
            )
        progress.update(label="Removing duplicates…")
        progress.write(f"Removed {duplicate_count} duplicate case(s).")
        progress.update(label="Saving results…")
        category_assessments = getattr(
            generated, "category_assessments", None
        )
        identifiers = services.add_generated_test_cases(
            project_id,
            unique_cases,
            **(
                {
                    "source_code": source_code,
                    "category_assessments": category_assessments,
                }
                if category_assessments is not None else {}
            ),
        )
        st.session_state[report_key].update({
            "generated": len(identifiers),
            "duplicates_removed": duplicate_count,
            "rejected": getattr(generated, "rejected_cases", 0),
            "failed_to_generate": getattr(generated, "failed_to_generate", 0),
            "category_assessments": category_assessments or [],
            "failed_batches": getattr(generated, "failed_batches", []),
        })
        progress.update(label="Generation completed.", state="complete")
    return identifiers


def _run_initial_test_generation(
    project: dict,
    requirements: list[dict],
    source_code: str,
    latest_version: dict | None,
) -> None:
    project_id = project["id"]
    status_key = f"cg_initial_generation_status_{project_id}"
    if not requirements:
        st.session_state[status_key] = "no_requirements"
        return
    if not source_code.strip():
        st.session_state[status_key] = "no_source"
        return
    try:
        _generate_workspace_tests(project, requirements, source_code, latest_version)
        st.session_state[status_key] = "completed"
    except (OllamaError, ValidationError) as exc:
        st.session_state[status_key] = "failed"
        st.session_state[f"{status_key}_error"] = str(exc)
    except Exception as exc:
        LOGGER.exception("Initial test generation failed for project %s", project_id)
        st.session_state[status_key] = "failed"
        st.session_state[f"{status_key}_error"] = (
            "Initial test generation failed unexpectedly. Check the application logs "
            "and retry."
        )


def _render_test_case_panel(
    project: dict,
    requirements: list[dict],
    source_code: str,
    latest_version: dict | None,
    latest_execution: dict | None,
) -> None:
    project_id = project["id"]
    status_key = f"cg_initial_generation_status_{project_id}"
    cases = services.list_test_cases(project_id)
    generation_status = st.session_state.get(status_key)

    if generation_status is None:
        if cases:
            st.session_state[status_key] = "existing"
        else:
            _run_initial_test_generation(
                project, requirements, source_code, latest_version
            )
        generation_status = st.session_state.get(status_key)
        if generation_status == "completed":
            st.rerun()
    if generation_status == "no_source" and source_code.strip() and not cases:
        _run_initial_test_generation(
            project, requirements, source_code, latest_version
        )
        generation_status = st.session_state.get(status_key)
        if generation_status == "completed":
            st.rerun()

    if generation_status == "failed":
        st.error(st.session_state.get(
            f"{status_key}_error", "AI test generation failed."
        ))
        if st.button("Retry AI test generation", key=f"retry-generation-{project_id}"):
            st.session_state[status_key] = None
            _run_initial_test_generation(
                project, requirements, source_code, latest_version
            )
            st.rerun()
    elif generation_status == "no_requirements":
        st.info("Add functional requirements before generating AI test cases.")
    elif generation_status == "no_source":
        st.info(
            "Test scenario generation is waiting for Python source. Write or paste "
            "your implementation in the editor; CodeGuard will save it as a .py "
            "version and automatically generate and validate tests."
        )
    elif generation_status == "completed":
        st.success("Initial AI test drafts are ready for review and approval.")

    cases = services.list_test_cases(project_id)
    st.subheader(f"Test cases ({len(cases)})")
    complexity = analyze_requirements(requirements)
    st.caption(
        f"Requirements: {complexity['basic']} basic · "
        f"{complexity['intermediate']} intermediate · "
        f"{complexity['advanced']} advanced · Model: Qwen 2.5 3B"
    )
    if complexity["advanced_warning"]:
        st.warning(
            complexity["advanced_limitation_warning"]
        )
        for item in complexity["advanced_requirements"]:
            st.caption(f"{item['id']}: {item['reason']}")
    generation_report = st.session_state.get(
        f"cg_generation_report_{project_id}", {}
    )
    if generation_report:
        st.caption(
            f"Generated: {generation_report['generated']} · "
            f"Duplicates removed: {generation_report['duplicates_removed']} · "
            f"Invalid cases rejected: {generation_report['rejected']} · "
            f"Failed to generate: {generation_report.get('failed_to_generate', 0)}"
        )
        for failed_batch in generation_report.get("failed_batches", []):
            st.warning(f"Some cases could not be generated for {failed_batch}")
    saved_generation_report = services.get_generation_report(
        project_id, source_code
    )
    if saved_generation_report:
        latest_status = {
            result["test_case_id"]: result["status"]
            for result in (latest_execution or {}).get("results", [])
            if result.get("test_case_id") is not None
        }
        category_rows = []
        for assessment in saved_generation_report["category_assessments"]:
            category = assessment["category"]
            matching = [case for case in cases if case["category"] == category]
            category_rows.append({
                "Category": CATEGORY_LABELS.get(category, category.title()),
                "Applicability": (
                    "Applicable" if assessment["applicable"] else "Not Applicable"
                ),
                "Generated": len(matching),
                "Approved": sum(
                    case["approval_status"] == "approved" for case in matching
                ),
                "Passed in latest run": sum(
                    latest_status.get(case["id"]) == "passed" for case in matching
                ),
                "Reason": assessment["reason"],
            })
        st.caption("Intermediate category coverage for the current source version")
        st.dataframe(
            pd.DataFrame(category_rows),
            width="stretch",
            hide_index=True,
        )

    with st.expander("Create a manual test case"):
        _manual_case_form(project, requirements)

    cases = services.list_test_cases(project_id)
    if not cases:
        st.caption("No test cases have been saved for this project.")
        return
    complexity_by_requirement = {
        item["id"]: item["complexity"]
        for item in complexity["requirements"]
    }
    search = st.text_input(
        "Search tests",
        key=f"workspace-test-search-{project_id}",
    ).strip().casefold()
    category_filter = st.selectbox(
        "Test category",
        ["All categories", *CATEGORY_OPTIONS],
        format_func=lambda value: (
            "All categories" if value == "All categories"
            else _category_label(value)
        ),
        key=f"workspace-test-category-{project_id}",
    )
    complexity_filter = st.selectbox(
        "Test complexity",
        ["All complexity levels", "Basic", "Intermediate", "Advanced"],
        key=f"workspace-test-complexity-{project_id}",
    )
    status_filter = st.selectbox(
        "Test status",
        ["All statuses", "pending", "approved", "rejected"],
        key=f"workspace-test-status-{project_id}",
    )
    cases = [
        case for case in cases
        if (
            not search
            or search in " ".join(
                str(case.get(field, ""))
                for field in (
                    "test_id", "name", "description", "function_name",
                    "testing_objective", "target_class", "requirement_id",
                )
            ).casefold()
        )
        and (
            category_filter == "All categories"
            or case["category"] == category_filter
        )
        and (
            complexity_filter == "All complexity levels"
            or case.get(
                "complexity",
                complexity_by_requirement.get(case["requirement_id"]),
            ) == complexity_filter
        )
        and (
            status_filter == "All statuses"
            or case["approval_status"] == status_filter
        )
    ]
    search = st.text_input(
        "Search test cases",
        key=f"test-search-{project_id}",
        placeholder="Search by ID, title, target, or requirement",
    ).strip().casefold()
    selected_category = st.selectbox(
        "Filter by category",
        ["All categories", *CATEGORY_OPTIONS],
        format_func=lambda value: (
            "All categories" if value == "All categories"
            else _category_label(value)
        ),
        key=f"test-category-filter-{project_id}",
    )
    selected_status = st.selectbox(
        "Filter by status",
        ["All statuses", "pending", "approved", "rejected"],
        key=f"test-status-filter-{project_id}",
    )
    cases = [
        case for case in cases
        if (
            not search
            or search in " ".join(
                str(case.get(field, ""))
                for field in (
                    "test_id", "name", "description", "function_name",
                    "testing_objective", "target_class", "requirement_id",
                )
            ).casefold()
        )
        and (
            selected_category == "All categories"
            or case["category"] == selected_category
        )
        and (
            selected_status == "All statuses"
            or case["approval_status"] == selected_status
        )
    ]
    results_by_case_id = {
        result["test_case_id"]: result
        for result in (latest_execution or {}).get("results", [])
        if result.get("test_case_id") is not None
    }
    for case in cases:
        result = results_by_case_id.get(case["id"])
        status = result["status"].upper() if result else "NOT RUN"
        with st.expander(
            f"{case['test_id']} · {case['name']} · "
            f"{CATEGORY_LABELS.get(case['category'], case['category'].title())} · "
            f"{case.get('complexity', complexity_by_requirement.get(case['requirement_id'], 'Unclassified'))} · "
            f"{case['priority']} · {case['approval_status'].upper()} · {status}"
        ):
            st.caption(
                f"Target: "
                f"{case['target_class'] + '.' if case.get('target_class') else ''}"
                f"{case['function_name']} · "
                f"Constructor inputs: {case.get('constructor_inputs_json', '[]')} · "
                f"Inputs: {case['inputs_json']} · "
                f"Requirement: {case['requirement_id'] or 'Unlinked'} · "
                f"Objective: {case.get('testing_objective', '')} · "
                f"Expected output: {case['expected_json']}"
            )
            st.caption(
                f"Validation: {'Validated at approval' if case['approval_status'] == 'approved' else 'Pending review'} · "
                f"Generated model: {'Qwen 2.5 3B' if case['generation_source'] == 'ollama' else 'Manual'}"
            )
            if result:
                if result["status"] == "passed":
                    st.success("Passed")
                elif result["status"] in {"failed", "error"}:
                    st.error(result["status"].title())
                else:
                    st.warning(result["status"].title())
                if result["message"]:
                    st.caption(result["message"])
            _render_test_editor(project, case, requirements)


def _render_execution_results(execution: dict | None) -> None:
    if not execution:
        st.info("No test execution data is available yet.")
        return
    metrics = st.columns(6)
    for column, label, value in zip(
        metrics,
        ("Status", "Passed", "Failed", "Skipped", "Errors", "Duration"),
        (
            execution["status"],
            execution["passed"],
            execution["failed"],
            execution["skipped"],
            execution["errors"],
            f"{execution['duration']:.2f}s",
        ),
    ):
        column.metric(label, value)
    if execution["coverage"].get("available"):
        st.metric(
            "Measured line coverage",
            f"{execution['coverage']['percentage']:.1f}%",
        )
    for result in execution["results"]:
        with st.expander(f"{result['test_id']} · {result['name']} · {result['status']}"):
            st.write(result["message"] or "No error message.")
            if result["source_location"]:
                st.caption(result["source_location"])
            if result["traceback"]:
                st.code(result["traceback"], language="text")


def _render_workspace(project: dict) -> None:
    with st.container(key=f"cg-workspace-shell-{project['id']}"):
        _render_workspace_layout(project)


def _render_workspace_layout(project: dict) -> None:
    project_id = project["id"]
    requirements = services.list_requirements(project_id)
    cases = services.list_test_cases(project_id)
    versions = services.list_code_versions(project_id)
    latest_version = versions[0] if versions else None
    executions = services.list_executions(project_id)
    latest_execution = (
        services.get_execution(executions[0]["id"]) if executions else None
    )
    filename, source = _workspace_source(project, latest_version)

    left, center, right = st.columns([1.15, 4.0, 2.2], gap="small")
    with left:
        st.subheader("Explorer")
        st.caption(project["title"])
        st.markdown(f"`{filename}`")
        if latest_version:
            st.caption(
                f"v{latest_version['version_number']} · {latest_version['kind']}"
            )
        with st.expander("Project details"):
            st.write(project["description"] or "No description.")
            st.caption(f"{project['language']} · {project['project_type'] or 'Project'}")
            st.caption(f"Updated {project['updated_at']}")
        with st.expander(f"Requirements ({len(requirements)})"):
            _render_requirements(project)
        with st.expander("Requirement coverage"):
            coverage = services.requirement_coverage(project_id)
            if coverage["total"]:
                st.progress(coverage["linked"] / coverage["total"])
                st.caption(
                    f"{coverage['linked']} of {coverage['total']} requirements have "
                    "linked tests (not a passing-test percentage)."
                )
            else:
                st.info("Add requirements to calculate linkage coverage.")
        with st.expander("Version history"):
            _render_history(project)
        with st.expander("Reports"):
            _render_reports(project)

    with center:
        st.subheader(filename)
        st.caption(
            f"Python · Version {latest_version['version_number'] if latest_version else 0}"
        )
        st.caption(
            "Write or edit your Python code here. CodeGuard tests your code; it does "
            "not write your implementation."
        )
        editor_key = (
            f"code-editor-{project_id}-{latest_version['id'] if latest_version else 0}"
        )
        edited_code = st_ace(
            value=source,
            language="python",
            theme="tomorrow_night",
            key=editor_key,
            height=520,
            font_size=14,
            show_gutter=True,
            show_print_margin=False,
            wrap=False,
            auto_update=True,
            readonly=False,
        )
        if edited_code is None:
            edited_code = source
        if st.button(
            "Run Tests",
            key=f"run-tests-{project_id}",
            type="primary",
            icon=":material/play_arrow:",
            disabled=not edited_code.strip(),
            help=(
                "Save the current editor contents as a .py version, validate approved "
                "tests against it, then run them in Docker."
            ),
        ):
            ready_cases = [item for item in cases if item["approval_status"] == "approved"]
            _run_project_tests(project, edited_code, latest_version, ready_cases)

    with right:
        with st.container(key=f"cg-workspace-tests-{project_id}"):
            _render_test_case_panel(
                project, requirements, edited_code, latest_version, latest_execution
            )

    st.html(
        f"""
        <script>
        (() => {{
          const projectId = {project_id};
          const storageKey = `codeguard-workspace-panes-${{projectId}}`;
          const rootSelector = `.st-key-cg-workspace-shell-${{projectId}}`;
          const minimums = [150, 360, 260];

          function getWeights() {{
            try {{
              const saved = JSON.parse(localStorage.getItem(storageKey));
              if (Array.isArray(saved) && saved.length === 3 &&
                  saved.every((value) => Number.isFinite(value) && value > 0)) {{
                return saved;
              }}
            }} catch (error) {{
              console.warn("Could not restore CodeGuard workspace panel sizes.", error);
            }}
            return [1.15, 4, 2.2];
          }}

          function getColumns(row) {{
            return Array.from(row.querySelectorAll(":scope > [data-testid='stColumn']"));
          }}

          function saveWeights(weights) {{
            try {{
              localStorage.setItem(storageKey, JSON.stringify(weights));
            }} catch (error) {{
              console.warn("Could not save CodeGuard workspace panel sizes.", error);
            }}
          }}

          function installHandle(row, index) {{
            let handle = row.querySelector(
              `:scope > .cg-pane-resizer[data-pane-index="${{index}}"]`
            );
            if (handle) return handle;
            handle = document.createElement("div");
            handle.className = "cg-pane-resizer";
            handle.dataset.paneIndex = String(index);
            handle.setAttribute("role", "separator");
            handle.setAttribute("aria-orientation", "vertical");
            handle.setAttribute("aria-label", "Resize workspace panels");
            handle.setAttribute("tabindex", "0");
            handle.title = "Drag to resize panels; use arrow keys for fine adjustment";

            handle.addEventListener("pointerdown", (event) => {{
              if (event.button !== 0) return;
              event.preventDefault();
              const columns = getColumns(row);
              const first = columns[index];
              const second = columns[index + 1];
              const firstStart = first.getBoundingClientRect().width;
              const secondStart = second.getBoundingClientRect().width;
              const pairWidth = firstStart + secondStart;
              const pairWeight = Number(first.style.flexGrow) +
                Number(second.style.flexGrow);
              const startX = event.clientX;
              handle.setPointerCapture(event.pointerId);

              const resize = (clientX) => {{
                const lowerBound = minimums[index];
                const upperBound = pairWidth - minimums[index + 1];
                if (upperBound <= lowerBound) return;
                const firstWidth = Math.max(
                  lowerBound,
                  Math.min(upperBound, firstStart + clientX - startX)
                );
                const firstWeight = pairWeight * firstWidth / pairWidth;
                const weights = columns.map((column) =>
                  Number(column.style.flexGrow)
                );
                weights[index] = firstWeight;
                weights[index + 1] = pairWeight - firstWeight;
                first.style.flex = `${{weights[index]}} 1 0%`;
                second.style.flex = `${{weights[index + 1]}} 1 0%`;
                return weights;
              }};

              const onMove = (moveEvent) => resize(moveEvent.clientX);
              const onUp = (upEvent) => {{
                const weights = resize(upEvent.clientX);
                handle.removeEventListener("pointermove", onMove);
                handle.removeEventListener("pointerup", onUp);
                handle.removeEventListener("pointercancel", onUp);
                if (weights) saveWeights(weights);
              }};
              handle.addEventListener("pointermove", onMove);
              handle.addEventListener("pointerup", onUp);
              handle.addEventListener("pointercancel", onUp);
            }});

            handle.addEventListener("keydown", (event) => {{
              if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
              event.preventDefault();
              const columns = getColumns(row);
              const first = columns[index];
              const second = columns[index + 1];
              const pairWeight = Number(first.style.flexGrow) +
                Number(second.style.flexGrow);
              const change = pairWeight * 0.03 *
                (event.key === "ArrowRight" ? 1 : -1);
              const nextFirst = Math.max(
                pairWeight * 0.1,
                Math.min(pairWeight * 0.9, Number(first.style.flexGrow) + change)
              );
              const weights = columns.map((column) =>
                Number(column.style.flexGrow)
              );
              weights[index] = nextFirst;
              weights[index + 1] = pairWeight - nextFirst;
              first.style.flex = `${{weights[index]}} 1 0%`;
              second.style.flex = `${{weights[index + 1]}} 1 0%`;
              saveWeights(weights);
            }});

            row.insertBefore(handle, getColumns(row)[index + 1]);
            return handle;
          }}

          function applyResizeControls() {{
            const root = document.querySelector(rootSelector);
            if (!root) return false;
            const row = Array.from(
              root.querySelectorAll("[data-testid='stHorizontalBlock']")
            ).find((candidate) =>
              getColumns(candidate).length === 3 &&
              candidate.querySelector("[class*='st-key-run-tests-']")
            );
            if (!row) return false;
            row.classList.add("cg-workspace-resizable");
            const columns = getColumns(row);
            const weights = getWeights();
            columns.forEach((column, index) => {{
              column.style.minWidth = "0";
              column.style.flex = `${{weights[index]}} 1 0%`;
            }});
            installHandle(row, 0);
            installHandle(row, 1);
            return true;
          }}

          if (!window.__codeguardWorkspaceResizer) {{
            const observer = new MutationObserver(() => {{
              window.__codeguardWorkspaceResizer.apply();
            }});
            window.__codeguardWorkspaceResizer = {{
              apply: applyResizeControls,
              observer,
            }};
            observer.observe(document.body, {{ childList: true, subtree: true }});
          }} else {{
            window.__codeguardWorkspaceResizer.apply = applyResizeControls;
          }}
          requestAnimationFrame(applyResizeControls);
        }})();
        </script>
        """,
        unsafe_allow_javascript=True,
    )

    st.divider()
    st.subheader("Execution and analysis")
    output_tab, results_tab, analysis_tab, corrections_tab, overview_tab = st.tabs(
        [
            "Execution Output",
            "Test Results",
            "AI Bug Analysis",
            "Suggested Corrections",
            "Project Overview",
        ]
    )
    with output_tab:
        if latest_execution:
            st.caption(
                f"Execution #{latest_execution['id']} · "
                f"v{latest_execution['version_number']} · "
                f"{latest_execution['status']} · "
                f"{latest_execution['duration']:.2f}s"
            )
            st.code(latest_execution["output"] or "No execution output.", language="text")
        else:
            st.info(
                "Execution output will appear here after you run this project's tests."
            )
    with results_tab:
        _render_execution_results(latest_execution)
    with analysis_tab:
        if latest_execution:
            _render_ai_analysis(project, latest_execution)
        else:
            st.info("AI analysis is available after a real test execution with failures.")
    with corrections_tab:
        _render_suggested_corrections(project, latest_execution)
    with overview_tab:
        _render_project_overview(project)


def _run_project_tests(
    project: dict,
    code: str,
    latest_version: dict | None,
    test_cases: list[dict],
) -> None:
    try:
        if latest_version is None:
            raise ValidationError("Project code version is missing.")
        if code != latest_version["content"]:
            version_id = services.save_code_version(project["id"], code, "user-edit")
        else:
            version_id = latest_version["id"]
        if not test_cases:
            st.info(
                f"Your code was saved as version {version_id}. Generate or add test "
                "cases and approve them before running tests."
            )
            return
        try:
            with st.spinner("Preparing the isolated Docker sandbox and running Pytest…"):
                result = run_approved_tests(code, test_cases)
        except SandboxError as exc:
            result = {
                "status": "infrastructure_error",
                "total": 0,
                "passed": 0,
                "failed": 0,
                "skipped": 0,
                "errors": 0,
                "duration": 0,
                "coverage": {"available": False},
                "output": str(exc),
                "error_kind": "docker",
                "results": [],
            }
        services.save_execution(project["id"], version_id, result)
        if result["status"] in {"infrastructure_error", "timeout"}:
            st.error(result["output"])
        else:
            st.success(
                f"Pytest finished: {result['passed']} passed, {result['failed']} failed, "
                f"{result['errors']} errors, {result['skipped']} skipped. "
                "Results are actual sandbox output."
            )
        st.rerun()
    except (ValidationError, SandboxError) as exc:
        st.error(str(exc))
    except Exception as exc:
        _show_error(exc)


def _save_workspace_code(
    project: dict, code: str, latest_version: dict | None
) -> int:
    if not code.strip():
        raise ValidationError("Write Python code in the editor before generating tests.")
    if latest_version is not None and code == latest_version["content"]:
        return latest_version["id"]
    return services.save_code_version(project["id"], code, "user-edit")


def _execution_has_failures(execution: dict) -> bool:
    return any(
        result["status"] in {"failed", "error"}
        for result in execution["results"]
    )


def _render_ai_analysis(project: dict, execution: dict) -> None:
    if not _execution_has_failures(execution):
        st.info("AI failure analysis requires actual failed or errored test results.")
        return
    st.caption("Ollama analysis is a suggestion based only on recorded test failures.")
    if st.button("Analyze actual failures", key=f"analyze-{execution['id']}"):
        try:
            with st.spinner("Asking local Ollama to analyze the recorded failures…"):
                analysis = analyze_failures(
                    project, execution, services.list_requirements(project["id"])
                )
            services.save_analysis(execution["id"], MODEL, analysis)
            st.success("Analysis received from local Ollama and saved.")
            st.rerun()
        except (OllamaError, ValidationError) as exc:
            st.error(str(exc))
        except Exception as exc:
            _show_error(exc)
    analyses = [
        item for item in services.list_analyses(project["id"])
        if item["execution_id"] == execution["id"]
    ]
    if analyses:
        st.markdown("**Saved analysis**")
        st.markdown(analyses[0]["content"])


def _render_suggested_corrections(
    project: dict, execution: dict | None
) -> None:
    if execution and _execution_has_failures(execution) and st.button(
        "Suggest code correction from latest failures",
        key=f"suggest-{execution['id']}",
    ):
        try:
            with st.spinner("Requesting a code correction suggestion from local Ollama…"):
                proposed, explanation = suggest_correction(execution)
            correction_id = services.create_correction(
                project["id"], execution["id"], execution["code_version_id"],
                proposed, explanation,
            )
            st.success(f"Correction suggestion {correction_id} saved for review.")
            st.rerun()
        except (OllamaError, ValidationError) as exc:
            st.error(str(exc))
        except Exception as exc:
            _show_error(exc)
    elif execution is None:
        st.info("Suggested corrections become available after real failed test results.")
    corrections = services.list_corrections(project["id"])
    if not corrections:
        if execution and not _execution_has_failures(execution):
            st.info("The latest execution has no failed tests to correct.")
        else:
            st.caption("No code correction suggestions have been saved.")
    for correction in corrections:
        with st.expander(
            f"Correction {correction['id']} · {correction['status']} · review before applying"
        ):
            st.write(correction["explanation"])
            original = services.get_code_version(correction["original_version_id"])
            original_code = original["content"] if original else ""
            old_lines = original_code.splitlines()
            new_lines = correction["proposed_code"].splitlines()
            original_column, proposed_column = st.columns(2)
            with original_column:
                st.markdown("**Original code**")
                st.code(original_code, language="python")
            with proposed_column:
                st.markdown("**Proposed code**")
                st.code(correction["proposed_code"], language="python")
            st.markdown("**Added, removed, and modified lines**")
            st.code(
                "\n".join(difflib.unified_diff(
                    old_lines, new_lines, fromfile="original.py", tofile="suggested.py"
                )),
                language="diff",
            )
            if correction["status"] == "pending":
                accept, reject = st.columns(2)
                if accept.button("Approve and save new version", key=f"accept-{correction['id']}"):
                    try:
                        version_id = services.decide_correction(correction["id"], True)
                        if version_id is None:
                            raise ValidationError(
                                "The correction was approved but no new code version was created."
                            )
                        version = services.get_code_version(version_id)
                        if version is None:
                            raise ValidationError(
                                "The approved code version could not be loaded."
                            )
                        st.success(
                            f"Approved and saved as version v{version['version_number']}. "
                            "Retest it to verify the change."
                        )
                        st.rerun()
                    except ValidationError as exc:
                        st.error(str(exc))
                if reject.button("Reject suggestion", key=f"decline-{correction['id']}"):
                    try:
                        services.decide_correction(correction["id"], False)
                        st.rerun()
                    except ValidationError as exc:
                        st.error(str(exc))


def _render_history(project: dict) -> None:
    versions = services.list_code_versions(project["id"])
    executions = services.list_executions(project["id"])
    corrections = services.list_corrections(project["id"])
    st.subheader("Code versions")
    if not versions:
        st.info("No code versions.")
    else:
        st.dataframe(
            pd.DataFrame(versions)[["version_number", "kind", "created_at"]],
            width="stretch",
            hide_index=True,
        )
        version_ids = [item["id"] for item in versions]
        first = st.selectbox(
            "Compare version", version_ids, format_func=lambda value: _version_label(versions, value),
            key=f"compare-a-{project['id']}",
        )
        second = st.selectbox(
            "With version", version_ids, index=min(1, len(version_ids) - 1),
            format_func=lambda value: _version_label(versions, value),
            key=f"compare-b-{project['id']}",
        )
        if first is None or second is None:
            st.error("Select two code versions to compare.")
        else:
            left_version = services.get_code_version(first)
            right_version = services.get_code_version(second)
            if left_version is None or right_version is None:
                st.error("A selected code version could not be loaded.")
            else:
                diff = difflib.unified_diff(
                    left_version["content"].splitlines(),
                    right_version["content"].splitlines(),
                    fromfile=f"v{left_version['version_number']}",
                    tofile=f"v{right_version['version_number']}",
                    lineterm="",
                )
                st.code("\n".join(diff) or "No differences.", language="diff")
                if st.button(
                    "Restore first version as a new version",
                    key=f"restore-{project['id']}",
                ):
                    try:
                        new_id = services.save_code_version(
                            project["id"],
                            left_version["content"],
                            f"restore-v{left_version['version_number']}",
                        )
                        restored = services.get_code_version(new_id)
                        if restored is None:
                            raise ValidationError(
                                "The restored code version could not be loaded."
                            )
                        st.success(
                            "Restored as version "
                            f"v{restored['version_number']}."
                        )
                        st.rerun()
                    except ValidationError as exc:
                        st.error(str(exc))
    st.subheader("Execution history")
    if executions:
        st.dataframe(
            pd.DataFrame(executions)[
                ["id", "created_at", "version_number", "status", "total", "passed",
                 "failed", "errors", "skipped", "duration"]
            ],
            width="stretch",
            hide_index=True,
        )
        if len(executions) >= 2:
            option_ids = [item["id"] for item in executions]
            before, after = st.columns(2)
            before_id = before.selectbox(
                "Earlier run", option_ids, index=min(1, len(option_ids) - 1),
                format_func=lambda value: f"Execution {value}",
                key=f"before-{project['id']}",
            )
            after_id = after.selectbox(
                "Later run", option_ids, index=0,
                format_func=lambda value: f"Execution {value}",
                key=f"after-{project['id']}",
            )
            if st.button("Compare executions", key=f"compare-runs-{project['id']}"):
                if before_id is None or after_id is None:
                    st.error("Select two executions to compare.")
                else:
                    try:
                        comparison = services.compare_executions(before_id, after_id)
                        st.write(f"Fixed failures: {comparison['fixed'] or 'None'}")
                        st.write(f"Still failing: {comparison['still_failing'] or 'None'}")
                        st.write(f"Regressions: {comparison['regressions'] or 'None'}")
                        st.write(f"New failures: {comparison['new_failures'] or 'None'}")
                        st.write(
                            f"Counts: {comparison['before']['passed']}/{comparison['before']['total']} "
                            f"passed → {comparison['after']['passed']}/{comparison['after']['total']} passed; "
                            f"duration {comparison['before']['duration']:.2f}s → "
                            f"{comparison['after']['duration']:.2f}s."
                        )
                    except ValidationError as exc:
                        st.error(str(exc))
    else:
        st.info("No execution history yet.")
    st.subheader("Correction review history")
    if corrections:
        st.dataframe(
            pd.DataFrame(corrections)[
                ["id", "execution_id", "original_version_number", "status", "created_at", "decided_at"]
            ],
            width="stretch",
            hide_index=True,
        )
    else:
        st.caption("No AI correction suggestions have been recorded.")


def _version_label(versions: list[dict], version_id: int) -> str:
    match = next(item for item in versions if item["id"] == version_id)
    return f"v{match['version_number']} · {match['kind']}"


def _render_reports(project: dict) -> None:
    data = services.project_report_data(project["id"])
    st.subheader("Downloads")
    versions = data["versions"]
    original = services.get_original_code(project["id"])
    if versions:
        st.download_button(
            "Download final Python code",
            data=versions[0]["content"].encode("utf-8"),
            file_name=original["filename"] if original else "main.py",
            mime="text/x-python",
        )
    if original:
        st.download_button(
            "Download original Python code",
            data=original["original_content"].encode("utf-8"),
            file_name=f"original_{original['filename']}",
            mime="text/x-python",
        )
    st.download_button(
        "Test cases JSON",
        data=json_export({"project": data["project"], "test_cases": data["test_cases"]}),
        file_name="test-cases.json",
        mime="application/json",
    )
    st.download_button(
        "Test cases CSV",
        data=tests_csv(data),
        file_name="test-cases.csv",
        mime="text/csv",
    )
    st.download_button(
        "Execution results JSON",
        data=json_export(data),
        file_name="execution-results.json",
        mime="application/json",
    )
    st.download_button(
        "Execution results CSV",
        data=executions_csv(data),
        file_name="execution-results.csv",
        mime="text/csv",
    )
    latest_execution_id = data["executions"][0]["id"] if data["executions"] else 0
    pdf_key = (
        f"pdf_data_{project['id']}_{latest_execution_id}_"
        f"{len(data['versions'])}_{len(data['test_cases'])}"
    )
    if st.button("Prepare complete PDF report", key=f"pdf-{project['id']}"):
        try:
            st.session_state[pdf_key] = pdf_report(data)
            services.add_report_metadata(project["id"], "pdf")
            st.rerun()
        except Exception as exc:
            _show_error(exc)
    if pdf_key in st.session_state:
        st.download_button(
            "Download complete PDF report",
            data=st.session_state[pdf_key],
            file_name="codeguard-report.pdf",
            mime="application/pdf",
            key=f"pdf-download-{project['id']}",
        )
        st.success("PDF report generated from stored project data.")
    metadata = services.list_report_metadata(project["id"])
    if metadata:
        st.caption(f"PDF report generations recorded: {len(metadata)}")
    coverage = services.requirement_coverage(project["id"])
    st.subheader("Requirement coverage")
    st.write(
        f"Linked tests: {coverage['linked']} of {coverage['total']} requirements · "
        f"Uncovered: {coverage['uncovered']}. Linkage does not imply a passing test."
    )
    latest = services.get_execution(data["executions"][0]["id"]) if data["executions"] else None
    if latest and latest["coverage"].get("available"):
        item = latest["coverage"]
        st.subheader("Measured code coverage")
        st.write(
            f"Coverage.py measured {item['percentage']:.1f}% line coverage: "
            f"{item['covered_lines']} covered, {item['missed_lines']} missed; "
            f"uncovered lines {item['uncovered_lines']}."
        )
    else:
        st.info("No Coverage.py measurement is available yet.")
