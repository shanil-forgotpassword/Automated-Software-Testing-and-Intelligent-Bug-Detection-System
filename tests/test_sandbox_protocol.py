from __future__ import annotations

from sandbox import runner


def test_junit_results_distinguish_assertions_from_exceptions(tmp_path, monkeypatch):
    results_file = tmp_path / "results.xml"
    results_file.write_text(
        """<testsuite>
        <testcase name="test_contract[TC-001]" time="0.1" />
        <testcase name="test_contract[TC-002]" time="0.2">
          <properties>
            <property name="expected_output" value="3" />
            <property name="actual_output" value="2" />
          </properties>
          <failure type="AssertionError" message="expected 3, got 2">assert details</failure>
        </testcase>
        <testcase name="test_contract[TC-003]" time="0.3">
          <failure type="ValueError" message="bad input">traceback details</failure>
        </testcase>
        <testcase name="test_contract[TC-004]" time="0">
          <skipped message="not applicable" />
        </testcase>
        </testsuite>""",
        encoding="utf-8",
    )
    monkeypatch.setattr(runner, "RESULTS_PATH", results_file)
    cases = [{"test_id": f"TC-{index:03d}", "name": f"Case {index}"}
             for index in range(1, 5)]
    results = runner._read_results(cases, "")
    assert [item["status"] for item in results] == [
        "passed", "failed", "error", "skipped"
    ]
    assert results[1]["expected_output"] == "3"
    assert results[1]["actual_output"] == "2"
