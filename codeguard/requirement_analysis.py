from __future__ import annotations

import re
import json
from collections import Counter
from typing import Any


_ADVANCED_PATTERNS = (
    (re.compile(r"\b(concurren\w*|parallel\w*|race condition|deadlock)\b", re.I),
     "Concurrency, parallel execution, and race-condition behavior require specialized synchronization tests."),
    (re.compile(r"\b(distributed|multi[- ]service|microservice|cross[- ]service)\b", re.I),
     "Distributed or multi-service behavior requires integration environments and coordinated service fixtures."),
    (re.compile(r"\b(transaction|rollback|isolation level|replication|shard)\b", re.I),
     "Complex transaction, replication, or sharding behavior needs dedicated database integration coverage."),
    (re.compile(r"\b(penetration test|exploit|security audit|threat model|cryptograph)\b", re.I),
     "Security assessment requires specialist threat models and security testing, not inferred functional tests."),
    (re.compile(r"\b(stress test|load test|performance test|throughput|latency target)\b", re.I),
     "Performance and load requirements need controlled workload and measurement infrastructure."),
    (re.compile(r"\b(state machine|state transition|workflow orchestration|saga)\b", re.I),
     "Complex state transitions or orchestrated workflows need systematic multi-step state coverage."),
    (re.compile(r"\b(multiple modules|multi[- ]module|integration test|external system)\b", re.I),
     "Cross-module or external-system interactions require integration fixtures and dependency control."),
    (re.compile(r"\b(combinatorial|many interacting|deeply nested|optimization problem)\b", re.I),
     "Interacting conditions or specialized algorithms require broader systematic analysis."),
)
_INTERMEDIATE_PATTERN = re.compile(
    r"\b(empty|null|none|invalid|boundary|edge case|exception|error handling|"
    r"out[- ]of[- ]range|malformed|optional|zero|negative|duplicate|missing|required)\b",
    re.I,
)
ADVANCED_LIMITATION_WARNING = (
    "Advanced testing requirements have been identified. "
    "The currently configured Qwen 2.5 3B model is being used for basic and "
    "intermediate test-case generation. Some advanced scenarios may exceed its "
    "reliable capabilities. For more comprehensive advanced test-case generation, "
    "we recommend using a higher-capability LLM. Advanced LLM integration is not "
    "currently available in CodeGuard AI."
)


def analyze_requirements(requirements: list[dict[str, Any]]) -> dict[str, Any]:
    """Classify requirement complexity using explicit, explainable indicators."""
    analyzed = []
    counts: Counter[str] = Counter()
    for requirement in requirements:
        description = str(requirement.get("description", "")).strip()
        complexity = "Basic"
        reason = "Direct functional behavior with ordinary inputs and outputs."
        for pattern, advanced_reason in _ADVANCED_PATTERNS:
            if pattern.search(description):
                complexity = "Advanced"
                reason = advanced_reason
                break
        else:
            if _INTERMEDIATE_PATTERN.search(description):
                complexity = "Intermediate"
                reason = (
                    "The requirement mentions validation, boundary, edge, or "
                    "exception behavior."
                )
        counts[complexity] += 1
        analyzed.append({
            "id": requirement.get("id", ""),
            "description": description,
            "complexity": complexity,
            "reason": reason,
        })
    return {
        "requirements": analyzed,
        "total": len(analyzed),
        "basic": counts["Basic"],
        "intermediate": counts["Intermediate"],
        "advanced": counts["Advanced"],
        "advanced_requirements": [
            item for item in analyzed if item["complexity"] == "Advanced"
        ],
        "advanced_warning": bool(counts["Advanced"]),
        "advanced_limitation_warning": (
            ADVANCED_LIMITATION_WARNING if counts["Advanced"] else ""
        ),
    }


def supported_requirements(analysis: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"id": item["id"], "description": item["description"]}
        for item in analysis["requirements"]
        if item["complexity"] in {"Basic", "Intermediate"}
    ]


def remove_duplicate_cases(cases: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for case in cases:
        signature = {
            field: case.get(field)
            for field in (
                "requirement_id",
                "target_class",
                "target_kind",
                "function_name",
                "inputs",
                "keyword_inputs",
                "constructor_inputs",
                "constructor_keyword_inputs",
                "setup",
                "expected",
                "expected_exception",
            )
        }
        fingerprint = json.dumps(
            signature, sort_keys=True, ensure_ascii=False, default=str
        )
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        unique.append(case)
    return unique, len(cases) - len(unique)
