import re

with open(r"c:\Users\vijay\Desktop\code guard ai\codeguard\ai.py", "r", encoding="utf-8") as f:
    content = f.read()

start = content.find("def generate_test_cases(")
end = content.find("def analyze_failures(")
if start == -1 or end == -1:
    print("ERROR: Could not find function boundaries")
    exit(1)

new_function = '''def _build_context(
    project: dict,
    requirements: list[dict],
    source_code: str,
    inspection: dict[str, Any],
    requirement_analysis: dict[str, Any],
) -> dict[str, Any]:
    supported = supported_requirements(requirement_analysis)
    return {
        "title": project.get("title", ""),
        "description": project.get("description", ""),
        "objectives": project.get("objectives", ""),
        "language": project.get("language", "Python"),
        "project_type": project.get("project_type", ""),
        "main_features": project.get("main_features", ""),
        "expected_inputs": project.get("expected_inputs", ""),
        "expected_outputs": project.get("expected_outputs", ""),
        "constraints": project.get("constraints", ""),
        "source_code": source_code[:50_000],
        "source_inspection": inspection,
        "source_testing_opportunities": _source_testing_opportunities(inspection),
        "requirement_complexity_analysis": requirement_analysis,
        "requirements": [
            {"id": item["id"], "description": item["description"]}
            for item in supported
        ],
    }


def _make_case_schema(requirement_ids: list[str], has_classes: bool) -> dict[str, Any]:
    expected_value_schema = {
        "anyOf": [
            {"type": "null"},
            {"type": "boolean"},
            {"type": "number"},
            {"type": "string"},
            {"type": "array", "items": {}},
            {"type": "object", "additionalProperties": {}},
        ]
    }
    schema = {
        "type": "object",
        "properties": {
            "test_cases": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "description": {"type": "string"},
                        "testing_objective": {"type": "string"},
                        "requirement_id": {
                            "type": "string",
                            "enum": requirement_ids,
                        },
                        "category": {
                            "type": "string",
                            "enum": list(GENERATION_CATEGORIES),
                        },
                        "function_name": {"type": "string"},
                        "target_class": {
                            "type": "string",
                            "enum": [""] + (
                                [item["name"] for item in inspection["classes"]]
                                if has_classes else []
                            ),
                        },
                        "target_kind": {
                            "type": "string",
                            "enum": [
                                "function", "instance_method",
                                "class_method", "static_method",
                            ],
                        },
                        "inputs": {"type": "array", "items": {}},
                        "keyword_inputs": {
                            "type": "object",
                            "additionalProperties": {},
                        },
                        "constructor_inputs": {"type": "array", "items": {}},
                        "constructor_keyword_inputs": {
                            "type": "object",
                            "additionalProperties": {},
                        },
                        "setup": {
                            "type": "array",
                            "maxItems": 10,
                            "items": {
                                "type": "object",
                                "properties": {
                                    "method": {"type": "string"},
                                    "inputs": {"type": "array", "items": {}},
                                    "keyword_inputs": {
                                        "type": "object",
                                        "additionalProperties": {},
                                    },
                                    "store_as": {"type": "string"},
                                },
                                "required": ["method", "inputs", "keyword_inputs", "store_as"],
                                "additionalProperties": False,
                            },
                        },
                        "expected": expected_value_schema,
                        "database_assertions": {
                            "type": "array",
                            "items": {},
                        },
                        "expected_exception": {
                            "type": "string",
                            "enum": list(ALLOWED_EXCEPTIONS),
                        },
                        "preconditions": {"type": "string"},
                        "priority": {
                            "type": "string",
                            "enum": ["Low", "Medium", "High", "Critical"],
                        },
                    },
                    "required": [
                        "name", "description", "testing_objective",
                        "requirement_id", "category", "function_name",
                        "target_class", "target_kind", "inputs",
                        "keyword_inputs", "constructor_inputs",
                        "constructor_keyword_inputs", "setup", "expected",
                        "database_assertions", "expected_exception",
                        "preconditions", "priority",
                    ],
                    "additionalProperties": False,
                },
                "maxItems": CASES_PER_BATCH,
            },
            "category_assessments": {
                "type": "object",
                "properties": {
                    cat: {
                        "type": "object",
                        "properties": {
                            "applicable": {"type": "boolean"},
                            "reason": {"type": "string"},
                        },
                        "required": ["applicable", "reason"],
                        "additionalProperties": False,
                    }
                    for cat in GENERATION_CATEGORIES
                },
                "required": list(GENERATION_CATEGORIES),
                "additionalProperties": False,
            },
        },
        "required": ["test_cases", "category_assessments"],
        "additionalProperties": False,
    }
    if not has_classes:
        schema["properties"]["target_class"] = {
            "type": "string",
            "enum": [""],
        }
        schema["properties"]["target_kind"] = {
            "type": "string",
            "enum": ["function"],
        }
        schema["properties"]["setup"]["maxItems"] = 0
        schema["properties"]["constructor_inputs"]["maxItems"] = 0
        schema["properties"]["constructor_keyword_inputs"]["maxProperties"] = 0
    return schema


def _plan_applicable_categories(
    context: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Stage 2: Ask Qwen which categories apply and why."""
    prompt = (
        "You are an intermediate-level Python software testing assistant.\n"
        "Analyze the supplied Python source code and determine which test "
        "categories are applicable.\n\n"
        "Categories: normal, empty_input, null_value, invalid_type, boundary, "
        "edge, exception, conditional, combination, validation, failure.\n\n"
        "For each category, return applicable=true only when the source code or "
        "requirements support a meaningful test case in that category. "
        "Set applicable=false and explain why otherwise.\n"
        "Do not invent behavior just to fill a category.\n"
        "Consider the actual functions, parameters, branches, comparison "
        "constants, raised exceptions, and type checks visible in the source.\n"
        "Return valid JSON only with a 'category_assessments' object keyed by "
        "each category, each value having boolean 'applicable' and non-empty "
        "string 'reason'.\n\n"
        "Context as JSON:\n"
        + json.dumps(context, ensure_ascii=False)
    )
    for attempt in range(MAX_GENERATION_ATTEMPTS):
        try:
            response_text = _request(prompt, json_output=True)
            response = _extract_json_from_text(response_text)
            assessments = response.get("category_assessments")
            if not isinstance(assessments, dict):
                raise OllamaError("Ollama must return category_assessments as an object.")
            result: dict[str, dict[str, Any]] = {}
            for category in GENERATION_CATEGORIES:
                assessment = assessments.get(category)
                if not isinstance(assessment, dict):
                    raise OllamaError(
                        f"Missing or invalid assessment for category {category!r}."
                    )
                if not isinstance(assessment.get("applicable"), bool):
                    raise OllamaError(
                        f"Category {category!r} assessment must have boolean 'applicable'."
                    )
                if not isinstance(assessment.get("reason"), str) or not assessment["reason"].strip():
                    raise OllamaError(
                        f"Category {category!r} assessment needs a non-empty 'reason'."
                    )
                result[category] = {
                    "category": category,
                    "applicable": assessment["applicable"],
                    "reason": assessment["reason"].strip()[:1000],
                }
            return result
        except OllamaError as exc:
            if attempt + 1 == MAX_GENERATION_ATTEMPTS:
                raise
            LOGGER.warning(
                "Category planning attempt %s failed: %s", attempt + 1, exc
            )
            prompt += (
                "\nThe previous response was rejected. Validation feedback: "
                + str(exc)
                + ". Return valid JSON only."
            )
    raise OllamaError("Ollama could not produce a valid category plan.")


def _generate_category_batch(
    batch_categories: tuple[str, ...],
    context: dict[str, Any],
    applicable_categories: dict[str, dict[str, Any]],
    inspection: dict[str, Any],
    requirement_ids: list[str],
) -> list[dict[str, Any]]:
    """Stage 3: Generate test cases for one category batch via Qwen 2.5 3B."""
    cats_in_batch = [c for c in batch_categories if applicable_categories[c]["applicable"]]
    if not cats_in_batch:
        return []

    cat_descriptions = {
        "empty_input": "empty string, empty list, empty dict, or missing input",
        "null_value": "None / null input values",
        "invalid_type": "wrong data type for a parameter (string, list, dict, etc.)",
        "boundary": "minimum, maximum, just below/above allowed limits",
        "edge": "extreme or unusual input values",
        "exception": "expected exceptions and error handling",
        "conditional": "if/elif/else branches and threshold conditions",
        "combination": "multiple inputs combined together",
        "validation": "input validation rules and preconditions",
        "failure": "common failure scenarios and error conditions",
        "normal": "typical valid inputs and expected behavior",
    }

    cat_details = []
    for cat in cats_in_batch:
        cat_details.append(
            f"- {cat}: {cat_descriptions.get(cat, '')} "
            f"(reason: {applicable_categories[cat]['reason']})"
        )

    prompt = (
        "You are an intermediate-level Python software testing assistant.\n"
        "Generate intermediate-level test cases for the following categories:\n"
        + "\n".join(cat_details)
        + "\n\n"
        "Requirements:\n"
        "- Generate " + str(CASES_PER_CATEGORY) + " distinct test cases per applicable category in this batch.\n"
        "- Do NOT stop after generating only three test cases total.\n"
        "- Generate multiple distinct scenarios for each applicable category.\n"
        "- Use actual functions, parameters, return values, and validation rules from the source code.\n"
        "- Do not invent functions or functionality.\n"
        "- Do not duplicate scenarios.\n"
        "- For positional arguments, use raw values directly in inputs arrays.\n"
        "- For named arguments, use keyword_inputs with exact parameter names.\n"
        "- Use pytest.raises() style expected_exception when an exception is expected.\n"
        "- Every test case must have: name, description, testing_objective, "
        "requirement_id, category, function_name, target_class, target_kind, "
        "inputs, keyword_inputs, constructor_inputs, constructor_keyword_inputs, "
        "setup, expected, database_assertions, expected_exception, preconditions, priority.\n"
        "- Return valid JSON only with 'test_cases' array and 'category_assessments' object.\n"
        "- category_assessments must have every category from GENERATION_CATEGORIES.\n\n"
        "Source code and context as JSON:\n"
        + json.dumps(context, ensure_ascii=False)
    )

    schema = _make_case_schema(requirement_ids, bool(inspection.get("classes")))

    last_error: str | None = None
    for attempt in range(MAX_GENERATION_ATTEMPTS):
        try:
            response_text = _request(prompt, json_output=schema, num_predict=OLLAMA_NUM_PREDICT)
            response = _extract_json_from_text(response_text)
            cases = response.get("test_cases")
            if not isinstance(cases, list):
                raise OllamaError("Ollama returned no test_cases array.")
            valid_cases: list[dict[str, Any]] = []
            for item in cases:
                if not isinstance(item, dict):
                    continue
                try:
                    _validate_single_case(item, requirement_ids, inspection, applicable_categories)
                    valid_cases.append(item)
                except OllamaError:
                    continue
            if valid_cases:
                return valid_cases
            raise OllamaError("No valid test cases in batch response.")
        except OllamaError as exc:
            last_error = str(exc)
            if attempt + 1 == MAX_GENERATION_ATTEMPTS:
                LOGGER.warning(
                    "Batch %s failed after %s attempts: %s",
                    ", ".join(batch_categories), MAX_GENERATION_ATTEMPTS, last_error
                )
                return []
            LOGGER.warning(
                "Batch %s attempt %s failed: %s",
                ", ".join(batch_categories), attempt + 1, last_error
            )
            prompt += (
                "\nThe previous response was rejected. Validation feedback: "
                + last_error
                + ". Return valid JSON only, fixing the issues."
            )
    LOGGER.warning(
        "Batch %s failed after %s attempts: %s",
        ", ".join(batch_categories), MAX_GENERATION_ATTEMPTS, last_error
    )
    return []


def _validate_single_case(
    item: dict[str, Any],
    requirement_ids: list[str],
    inspection: dict[str, Any],
    applicable_categories: dict[str, dict[str, Any]],
) -> None:
    required_fields = {
        "name", "description", "testing_objective", "requirement_id",
        "category", "function_name", "target_class", "target_kind", "inputs",
        "keyword_inputs", "constructor_inputs", "constructor_keyword_inputs",
        "setup", "expected", "database_assertions", "expected_exception",
        "preconditions", "priority",
    }
    if set(item) != required_fields:
        missing = required_fields - set(item)
        unexpected = set(item) - required_fields
        detail = []
        if missing:
            detail.append("missing " + ", ".join(sorted(missing)))
        if unexpected:
            detail.append("unexpected " + ", ".join(sorted(unexpected)))
        raise OllamaError("Test case has fields outside required schema: " + "; ".join(detail))

    if item.get("requirement_id") not in requirement_ids:
        raise OllamaError("Test case references unknown requirement_id.")

    if item.get("category") not in GENERATION_CATEGORIES:
        raise OllamaError("Test case has unsupported category.")

    if item.get("expected_exception", "") not in ALLOWED_EXCEPTIONS:
        raise OllamaError("Test case has unsupported expected_exception.")

    if not isinstance(item.get("name"), str) or not item["name"].strip():
        raise OllamaError("Test case has no name.")

    if not isinstance(item.get("testing_objective"), str) or not item["testing_objective"].strip():
        raise OllamaError("Test case has no testing objective.")

    try:
        json.dumps(item["inputs"], allow_nan=False)
        json.dumps(item["expected"], allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise OllamaError("Test case has non-JSON values.") from exc

    validated_item = {
        "name": item["name"],
        "description": item.get("description", ""),
        "testing_objective": item["testing_objective"].strip(),
        "complexity": (
            "Intermediate" if item["category"] in INTERMEDIATE_CATEGORIES else "Basic"
        ),
        "requirement_id": item["requirement_id"],
        "category": item["category"],
        "function_name": item["function_name"],
        "target_class": item.get("target_class", ""),
        "target_kind": item.get("target_kind", "function"),
        "inputs": item["inputs"],
        "keyword_inputs": item.get("keyword_inputs", {}),
        "constructor_inputs": item.get("constructor_inputs", []),
        "constructor_keyword_inputs": item.get("constructor_keyword_inputs", {}),
        "setup": item.get("setup", []),
        "expected": item["expected"],
        "database_assertions": item.get("database_assertions", []),
        "expected_exception": item.get("expected_exception", ""),
        "preconditions": item.get("preconditions", ""),
        "priority": item.get("priority", "Medium"),
    }

    try:
        validate_case_target(validated_item, inspection)
    except SourceAnalysisError as exc:
        raise OllamaError(
            f"Test case incompatible with inspected source: {exc}"
        ) from exc

    if not _case_matches_intermediate_category(
        validated_item, _target_behavior(inspection, validated_item)
    ):
        raise OllamaError(
            f"Test case inputs/objective do not match inspected source behavior for category {item['category']}."
        )

    if not applicable_categories[item["category"]]["applicable"]:
        raise OllamaError(
            f"Generated a {item['category']} test after marking that category not applicable."
        )


def generate_test_cases(
    project: dict,
    requirements: list[dict],
    *,
    source_code: str = "",
) -> list[dict]:
    """Multi-stage intermediate test case generation using Qwen 2.5 3B.

    Stage 1: Source code analysis
    Stage 2: Test planning (category applicability)
    Stage 3: Category-wise generation (multiple batches)
    Stage 4: Combine results
    Stage 5: Validation
    Stage 6: Return combined results with category assessments
    """
    if not requirements:
        raise ValidationError("Add at least one requirement before generating tests.")
    if len(source_code.encode("utf-8")) > 200_000:
        raise ValidationError("Python source must be 200 KB or smaller.")
    if not source_code.strip():
        raise ValidationError(
            "Add the current Python source before generating tests so targets and "
            "signatures can be validated."
        )

    # Stage 1: Source code analysis
    try:
        inspection = analyze_source(source_code)
    except SourceAnalysisError as exc:
        raise OllamaError(str(exc)) from exc

    requirement_analysis = analyze_requirements(requirements)
    supported = supported_requirements(requirement_analysis)
    if not supported:
        raise ValidationError(
            "All current requirements were classified as advanced. Qwen 2.5 3B "
            "generation is limited to basic and intermediate scenarios."
        )

    requirement_ids = [item["id"] for item in supported]
    context = _build_context(
        project, requirements, source_code, inspection, requirement_analysis
    )

    # Stage 2: Test planning - determine applicable categories
    applicable_categories = _plan_applicable_categories(context)

    # Stage 3: Category-wise generation
    all_cases: list[dict[str, Any]] = []
    failed_batches: list[str] = []
    for batch in CATEGORY_BATCHES:
        batch_cases = _generate_category_batch(
            batch, context, applicable_categories, inspection, requirement_ids
        )
        if batch_cases:
            all_cases.extend(batch_cases)
            LOGGER.info(
                "Batch %s generated %d test case(s)",
                ", ".join(batch), len(batch_cases)
            )
        else:
            if any(applicable_categories[c]["applicable"] for c in batch):
                failed_batches.append(", ".join(batch))

    if not all_cases:
        raise OllamaError(
            "Qwen 2.5 3B generated no valid test cases across all batches. "
            + (
                " Failed batches: " + ", ".join(failed_batches)
                if failed_batches else ""
            )
        )

    # Stage 4 & 5: Validation already done per-case in _generate_category_batch
    assessments = list(applicable_categories.values())

    if len(all_cases) > MAX_GENERATED_CASES:
        all_cases = all_cases[:MAX_GENERATED_CASES]

    return GeneratedTestCases(all_cases, assessments)


'''

new_content = content[:start] + new_function + content[end:]

with open(r"c:\Users\vijay\Desktop\code guard ai\codeguard\ai.py", "w", encoding="utf-8") as f:
    f.write(new_content)

print("Done. Replaced generate_test_cases function.")
