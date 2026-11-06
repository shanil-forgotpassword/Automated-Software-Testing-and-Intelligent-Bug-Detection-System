from __future__ import annotations

import html
import io
import json
import difflib
import re
from datetime import datetime
from typing import Any

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

_CSV_FORMULA_PREFIX = re.compile(r"^[\s\x00-\x1f]*[=+\-@]")


def _csv_safe_value(value: Any) -> Any:
    if isinstance(value, str) and _CSV_FORMULA_PREFIX.match(value):
        return "'" + value
    return value


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    for column in frame.columns:
        frame[column] = frame[column].map(_csv_safe_value)
    return frame.to_csv(index=False).encode("utf-8-sig")


def json_export(data: dict[str, Any]) -> bytes:
    return json.dumps(data, ensure_ascii=False, indent=2, default=str).encode("utf-8")


def tests_csv(data: dict[str, Any]) -> bytes:
    columns = [
        "test_id", "name", "description", "testing_objective", "complexity",
        "requirement_id", "category",
        "function_name", "inputs_json", "expected_json", "expected_exception",
        "preconditions", "priority", "generation_source", "approval_status",
    ]
    frame = pd.DataFrame(data["test_cases"], columns=columns)
    return _csv_bytes(frame)


def executions_csv(data: dict[str, Any]) -> bytes:
    rows = []
    for execution in data["executions"]:
        for result in execution.get("results", []):
            rows.append({
                "execution_id": execution["id"],
                "created_at": execution["created_at"],
                "code_version": execution["version_number"],
                "execution_status": execution["status"],
                "test_id": result["test_id"],
                "test_name": result["name"],
                "test_status": result["status"],
                "message": result["message"],
                "traceback": result["traceback"],
                "duration_seconds": result["duration"],
            })
    columns = [
        "execution_id", "created_at", "code_version", "execution_status",
        "test_id", "test_name", "test_status", "message", "traceback",
        "duration_seconds",
    ]
    return _csv_bytes(pd.DataFrame(rows, columns=columns))


def execution_comparisons(data: dict[str, Any]) -> list[dict[str, Any]]:
    chronological = list(reversed(data["executions"]))
    comparisons = []
    for before, after in zip(chronological, chronological[1:]):
        before_status = {item["test_id"]: item["status"] for item in before["results"]}
        after_status = {item["test_id"]: item["status"] for item in after["results"]}
        common = before_status.keys() & after_status.keys()
        comparisons.append({
            "before_id": before["id"],
            "after_id": after["id"],
            "fixed": [
                test_id for test_id in common
                if before_status[test_id] == "failed" and after_status[test_id] == "passed"
            ],
            "still_failing": [
                test_id for test_id in common
                if after_status[test_id] in {"failed", "error"}
            ],
            "regressions": [
                test_id for test_id in common
                if before_status[test_id] == "passed"
                and after_status[test_id] in {"failed", "error"}
            ],
            "new_failures": [
                test_id for test_id in after_status.keys() - before_status.keys()
                if after_status[test_id] in {"failed", "error"}
            ],
            "before_passed": before["passed"],
            "before_total": before["total"],
            "after_passed": after["passed"],
            "after_total": after["total"],
            "before_duration": before["duration"],
            "after_duration": after["duration"],
        })
    return comparisons


def pdf_report(data: dict[str, Any]) -> bytes:
    output = io.BytesIO()
    document = SimpleDocTemplate(
        output,
        pagesize=letter,
        rightMargin=42,
        leftMargin=42,
        topMargin=42,
        bottomMargin=42,
        title=f"CodeGuard AI report — {data['project']['title']}",
    )
    styles = getSampleStyleSheet()
    story = [
        Paragraph("CodeGuard AI — Testing Report", styles["Title"]),
        Paragraph(html.escape(data["project"]["title"]), styles["Heading2"]),
        Paragraph(
            f"Generated {datetime.now().astimezone().isoformat(timespec='minutes')}",
            styles["Normal"],
        ),
        Spacer(1, 12),
        Paragraph("Project information", styles["Heading2"]),
        Paragraph(html.escape(data["project"]["description"] or "No description."),
                  styles["BodyText"]),
        Paragraph(
            "Objectives: " + html.escape(data["project"]["objectives"] or "Not provided."),
            styles["BodyText"],
        ),
        Paragraph(
            "Project type: " + html.escape(data["project"]["project_type"] or "Not specified."),
            styles["BodyText"],
        ),
        Paragraph(
            "Features: " + html.escape(data["project"]["main_features"] or "Not provided."),
            styles["BodyText"],
        ),
        Paragraph(
            "Expected inputs: " + html.escape(data["project"]["expected_inputs"] or "Not provided.")
            + " · Expected outputs: "
            + html.escape(data["project"]["expected_outputs"] or "Not provided."),
            styles["BodyText"],
        ),
        Paragraph(
            "Constraints: " + html.escape(data["project"]["constraints"] or "Not provided."),
            styles["BodyText"],
        ),
    ]
    story.extend([
        Paragraph("Requirements", styles["Heading2"]),
        Paragraph(
            "Complexity: "
            f"{data.get('requirement_analysis', {}).get('basic', 0)} basic, "
            f"{data.get('requirement_analysis', {}).get('intermediate', 0)} "
            "intermediate, "
            f"{data.get('requirement_analysis', {}).get('advanced', 0)} advanced.",
            styles["BodyText"],
        ),
        _table(
            [["ID", "Requirement"]]
            + [[item["id"], item["description"]] for item in data["requirements"]]
        ),
        Paragraph("Advanced Testing Limitations", styles["Heading3"]),
        *(
            [
                Paragraph(
                    html.escape(
                        data["requirement_analysis"]["advanced_limitation_warning"]
                    ),
                    styles["BodyText"],
                )
            ]
            if data.get("requirement_analysis", {}).get("advanced_warning")
            else []
        ),
        _table(
            [["Requirement", "Why additional testing is needed"]]
            + [
                [item["id"], item["reason"]]
                for item in data.get("requirement_analysis", {}).get(
                    "advanced_requirements", []
                )
            ]
            if data.get("requirement_analysis", {}).get("advanced_requirements")
            else [["None identified", ""]]
        ),
        Paragraph("Test cases and approvals", styles["Heading2"]),
        _table(
            [["ID", "Name", "Category", "Objective", "Requirement", "Status"]]
            + [[item["test_id"], item["name"], item["category"],
                item.get("testing_objective", item["description"]),
                item["requirement_id"] or "—",
                item["approval_status"]] for item in data["test_cases"]]
        ),
        Paragraph("Intermediate category coverage", styles["Heading2"]),
        _table(
            [["Category", "Applicability", "Reason"]]
            + [
                [
                    item["category"],
                    "Applicable" if item["applicable"] else "Not Applicable",
                    item["reason"],
                ]
                for item in (data.get("generation_report") or {}).get(
                    "category_assessments", []
                )
            ]
            if data.get("generation_report")
            else [["Not recorded", "—", "No category assessment is available."]]
        ),
        Paragraph("Code versions", styles["Heading2"]),
        _table(
            [["Version", "Type", "Created"]]
            + [[f"v{item['version_number']}", item["kind"], item["created_at"]]
               for item in reversed(data["versions"])]
        ),
        Paragraph("Actual test executions", styles["Heading2"]),
    ])
    if not data["executions"]:
        story.append(Paragraph("No test executions have been recorded.", styles["BodyText"]))
    for execution in reversed(data["executions"]):
        story.extend([
            Paragraph(
                f"Execution {execution['id']} — v{execution['version_number']} "
                f"({html.escape(execution['status'])})",
                styles["Heading3"],
            ),
            Paragraph(
                f"Total: {execution['total']} | Passed: {execution['passed']} | "
                f"Failed: {execution['failed']} | Errors: {execution['errors']} | "
                f"Skipped: {execution['skipped']} | Duration: {execution['duration']:.2f}s",
                styles["BodyText"],
            ),
        ])
        coverage = execution.get("coverage", {})
        if coverage.get("available"):
            story.append(Paragraph(
                f"Coverage.py line coverage: {coverage.get('percentage', 0):.1f}% "
                f"({coverage.get('covered_lines', 0)} covered, "
                f"{coverage.get('missed_lines', 0)} missed); uncovered lines: "
                f"{html.escape(str(coverage.get('uncovered_lines', [])))}",
                styles["BodyText"],
            ))
        for result in execution.get("results", []):
            if result["status"] not in {"passed", "skipped"}:
                story.append(Paragraph(
                    f"{html.escape(result['test_id'])} — "
                    f"{html.escape(result['status'])}: "
                    f"{html.escape(result['message'])}",
                    styles["BodyText"],
                ))
                if result["traceback"]:
                    story.append(Preformatted(
                        html.escape(result["traceback"][:3000]), styles["Code"]
                    ))
    comparisons = execution_comparisons(data)
    if comparisons:
        story.append(Paragraph("Before-and-after execution comparisons", styles["Heading2"]))
        comparison_rows = [["Runs", "Fixed", "Still failing", "Regressions", "Pass counts"]]
        for comparison in comparisons:
            comparison_rows.append([
                    f"#{comparison['before_id']} → #{comparison['after_id']}",
                    ", ".join(comparison["fixed"]) or "None",
                    ", ".join(comparison["still_failing"]) or "None",
                    ", ".join(comparison["regressions"]) or "None",
                    f"{comparison['before_passed']}/{comparison['before_total']} → "
                    f"{comparison['after_passed']}/{comparison['after_total']}",
            ])
        story.append(_table(comparison_rows))
    story.append(Paragraph("Requirement linkage coverage", styles["Heading2"]))
    mappings = []
    for requirement in data["requirements"]:
        linked = [
            case for case in data["test_cases"]
            if case["requirement_id"] == requirement["id"]
        ]
        mappings.append([
            requirement["id"],
            "Linked" if linked else "Uncovered",
            ", ".join(case["test_id"] for case in linked) or "No linked tests",
        ])
    story.append(_table([["Requirement", "Linkage", "Tests"], *mappings]))
    story.append(Paragraph("AI analyses (unverified suggestions)", styles["Heading2"]))
    for analysis in data["analyses"]:
        story.extend([
            Paragraph(f"Execution {analysis['execution_id']} · {analysis['model']}",
                      styles["Heading3"]),
            Preformatted(html.escape(analysis["content"][:6000]), styles["Code"]),
        ])
    story.append(Paragraph("Correction review history", styles["Heading2"]))
    version_by_id = {version["id"]: version for version in data["versions"]}
    for correction in data["corrections"]:
        story.append(Paragraph(
            f"Correction {correction['id']} — {html.escape(correction['status'])}; "
            f"{html.escape(correction['explanation'])}",
            styles["BodyText"],
        ))
        if correction["applied_version_id"]:
            original = version_by_id.get(correction["original_version_id"], {})
            applied = version_by_id.get(correction["applied_version_id"], {})
            diff = difflib.unified_diff(
                original.get("content", "").splitlines(),
                applied.get("content", "").splitlines(),
                fromfile=f"v{original.get('version_number', '?')}",
                tofile=f"v{applied.get('version_number', '?')}",
            )
            story.append(Preformatted(
                html.escape("\n".join(diff)[:6000]), styles["Code"]
            ))
    document.build(story)
    return output.getvalue()


def _table(rows: list[list[Any]]) -> Table:
    values = [
        [Paragraph(html.escape(str(value) or "—"), getSampleStyleSheet()["BodyText"])
         for value in row]
        for row in rows
    ]
    table = Table(values, repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#19151F")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#A78BFA")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#A4B0AA")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#F3F5F4")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return table
