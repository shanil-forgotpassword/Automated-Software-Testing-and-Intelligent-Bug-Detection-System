from __future__ import annotations

import ast
from typing import Any


class SourceAnalysisError(ValueError):
    """Raised when Python source cannot be inspected without executing it."""


class _ReturnVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.expressions: list[str] = []
        self.assignments: dict[str, str] = {}

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        return

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        return

    def visit_Lambda(self, node: ast.Lambda) -> None:
        return

    def visit_Return(self, node: ast.Return) -> None:
        if node.value is not None:
            self.expressions.append(ast.unparse(node.value))
        else:
            self.expressions.append("None")

    def visit_Assign(self, node: ast.Assign) -> None:
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            self.assignments[node.targets[0].id] = ast.unparse(node.value)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if isinstance(node.target, ast.Name) and node.value is not None:
            self.assignments[node.target.id] = ast.unparse(node.value)
        self.generic_visit(node)


class _BehaviorVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.branches: list[dict[str, Any]] = []
        self.raises: list[str] = []
        self.calls: list[str] = []
        self.boundary_values: list[int | float] = []
        self.type_checks: list[dict[str, str]] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        return

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        return

    def visit_Lambda(self, node: ast.Lambda) -> None:
        return

    def visit_If(self, node: ast.If) -> None:
        self.branches.append({
            "condition": ast.unparse(node.test),
            "has_else": bool(node.orelse),
            "line": node.lineno,
        })
        self.generic_visit(node)

    def visit_IfExp(self, node: ast.IfExp) -> None:
        self.branches.append({
            "condition": ast.unparse(node.test),
            "has_else": True,
            "line": node.lineno,
        })
        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare) -> None:
        for comparator in [node.left, *node.comparators]:
            try:
                value = ast.literal_eval(comparator)
            except (ValueError, TypeError, SyntaxError):
                continue
            if (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
            ):
                self.boundary_values.append(value)
        self.generic_visit(node)

    def visit_Raise(self, node: ast.Raise) -> None:
        self.raises.append(ast.unparse(node.exc) if node.exc else "bare raise")

    def visit_Call(self, node: ast.Call) -> None:
        self.calls.append(ast.unparse(node.func))
        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "isinstance"
            and len(node.args) >= 2
        ):
            self.type_checks.append({
                "value": ast.unparse(node.args[0]),
                "expected_type": ast.unparse(node.args[1]),
            })
        self.generic_visit(node)


def _arguments(
    arguments: ast.arguments,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    positional = [*arguments.posonlyargs, *arguments.args]
    defaults_offset = len(positional) - len(arguments.defaults)
    positional_only_count = len(arguments.posonlyargs)
    parameters = [
        {
            "name": argument.arg,
            "kind": (
                "positional_only" if index < positional_only_count
                else "positional_or_keyword"
            ),
            "required": index < defaults_offset,
            "default": (
                ast.unparse(arguments.defaults[index - defaults_offset])
                if index >= defaults_offset else None
            ),
            "annotation": ast.unparse(argument.annotation) if argument.annotation else "",
        }
        for index, argument in enumerate(positional)
    ]
    keyword_only = [
        {
            "name": argument.arg,
            "required": default is None,
            "default": ast.unparse(default) if default is not None else None,
            "annotation": ast.unparse(argument.annotation) if argument.annotation else "",
        }
        for argument, default in zip(arguments.kwonlyargs, arguments.kw_defaults)
    ]
    return parameters, keyword_only


def _return_contract(node: ast.FunctionDef | ast.AsyncFunctionDef) -> dict[str, Any]:
    visitor = _ReturnVisitor()
    for statement in node.body:
        visitor.visit(statement)
    expressions = visitor.expressions
    returned_names = [
        expression for expression in expressions
        if expression.isidentifier() and expression in visitor.assignments
    ]
    for index, expression in enumerate(expressions):
        seen: set[str] = set()
        while expression.isidentifier() and expression in visitor.assignments:
            if expression in seen:
                break
            seen.add(expression)
            expression = visitor.assignments[expression]
        expressions[index] = expression
    if not expressions:
        kind = "none"
    elif all(expression.endswith(".lastrowid") for expression in expressions):
        kind = "integer"
    elif all(expression.endswith(".fetchone()") for expression in expressions):
        kind = "row_or_none"
    elif all(expression in {"True", "False"} for expression in expressions):
        kind = "boolean"
    elif all(expression == "None" for expression in expressions):
        kind = "none"
    elif all(expression.startswith(("'", '"')) for expression in expressions):
        kind = "string"
    elif all(expression.lstrip("-").isdigit() for expression in expressions):
        kind = "integer"
    elif all(_is_number_literal(expression) for expression in expressions):
        kind = "number"
    elif all(expression.startswith(("[", "(")) for expression in expressions):
        kind = "collection"
    elif all(expression.startswith("{") for expression in expressions):
        kind = "object"
    else:
        kind = "unknown"
    return {
        "kind": kind,
        "expressions": expressions,
        "returned_names": returned_names,
    }


def _is_number_literal(expression: str) -> bool:
    try:
        value = ast.literal_eval(expression)
    except (ValueError, SyntaxError):
        return False
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _method_info(
    node: ast.FunctionDef | ast.AsyncFunctionDef, *, class_member: bool
) -> dict[str, Any]:
    parameters, required_keyword_only = _arguments(node.args)
    decorators = {
        item.id if isinstance(item, ast.Name) else
        item.attr if isinstance(item, ast.Attribute) else ""
        for item in node.decorator_list
    }
    kind = (
        "static_method" if "staticmethod" in decorators else
        "class_method" if "classmethod" in decorators else
        "instance_method" if class_member else
        "function"
    )
    if kind in {"instance_method", "class_method"} and parameters:
        parameters = parameters[1:]
    behavior = _BehaviorVisitor()
    for statement in node.body:
        behavior.visit(statement)
    return {
        "name": node.name,
        "kind": kind,
        "parameters": parameters,
        "maximum_arguments": (
            None if node.args.vararg is not None else len(parameters)
        ),
        "accepts_var_keyword": node.args.kwarg is not None,
        "keyword_only_parameters": required_keyword_only,
        "required_keyword_only": [
            item["name"] for item in required_keyword_only if item["required"]
        ],
        "iterable_items": _iterable_items(node),
        "return": _return_contract(node),
        "async": isinstance(node, ast.AsyncFunctionDef),
        "branches": behavior.branches,
        "raises": list(dict.fromkeys(behavior.raises)),
        "calls": list(dict.fromkeys(behavior.calls)),
        "boundary_values": sorted(set(behavior.boundary_values)),
        "type_checks": behavior.type_checks,
    }


def _iterable_items(node: ast.FunctionDef | ast.AsyncFunctionDef) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for child in ast.walk(node):
        if (
            isinstance(child, ast.For)
            and isinstance(child.iter, ast.Name)
        ):
            if isinstance(child.target, ast.Name):
                result[child.iter.id] = [child.target.id]
            elif (
                isinstance(child.target, (ast.Tuple, ast.List))
            ):
                names: list[str] = []
                for item in child.target.elts:
                    if not isinstance(item, ast.Name):
                        names = []
                        break
                    names.append(item.id)
                if names:
                    result[child.iter.id] = names
    return result


def _import_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Import):
        return ", ".join(item.name for item in node.names)
    if isinstance(node, ast.ImportFrom):
        imported = ", ".join(item.name for item in node.names)
        return f"from {node.module or ''} import {imported}"
    return None


def analyze_source(source: str) -> dict[str, Any]:
    """Describe Python callables and contracts using syntax only; never execute source."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise SourceAnalysisError(
            f"Cannot inspect Python source at line {exc.lineno}: {exc.msg}"
        ) from exc

    imports = list(dict.fromkeys(
        name for node in ast.walk(tree)
        if (name := _import_name(node)) is not None
    ))
    functions = [
        _method_info(node, class_member=False)
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    class_nodes = {
        node.name: node for node in tree.body if isinstance(node, ast.ClassDef)
    }
    resolved: dict[str, dict[str, Any]] = {}

    def inspect_class(name: str, trail: set[str] | None = None) -> dict[str, Any]:
        if name in resolved:
            return resolved[name]
        trail = set() if trail is None else trail
        if name in trail:
            return {"constructor": None, "methods": {}}
        trail.add(name)
        node = class_nodes[name]
        methods: dict[str, dict[str, Any]] = {}
        constructor = None
        base_infos = []
        for base in node.bases:
            base_name = base.id if isinstance(base, ast.Name) else ""
            if base_name in class_nodes:
                base_infos.append(inspect_class(base_name, trail))
        for base_info in reversed(base_infos):
            methods.update(base_info["methods"])
        for base_info in base_infos:
            if base_info["constructor"] is not None:
                constructor = base_info["constructor"]
                break
        local_methods = [
            method for method in node.body
            if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        for method_node in local_methods:
            info = _method_info(method_node, class_member=True)
            info["declared_in"] = name
            if method_node.name == "__init__":
                constructor = info
            else:
                methods[method_node.name] = info
        result = {
            "constructor": constructor,
            "methods": methods,
        }
        resolved[name] = result
        trail.remove(name)
        return result

    classes = []
    for name, node in class_nodes.items():
        class_info = inspect_class(name)
        classes.append({
            "name": name,
            "bases": [ast.unparse(base) for base in node.bases],
            "constructor": class_info["constructor"],
            "methods": list(class_info["methods"].values()),
        })
    return {
        "imports": imports,
        "functions": functions,
        "classes": classes,
    }


def _validate_call_arguments(
    callable_info: dict[str, Any],
    positional: list[Any],
    keyword: dict[str, Any],
    label: str,
    *,
    allow_missing_required: bool = False,
) -> None:
    if not isinstance(positional, list) or not isinstance(keyword, dict):
        raise SourceAnalysisError(f"{label} arguments must be positional and keyword values.")
    parameters = callable_info.get("parameters", [])
    maximum = callable_info.get("maximum_arguments", len(parameters))
    if maximum is not None and len(positional) > maximum:
        raise SourceAnalysisError(
            f"{label} accepts at most {maximum} positional arguments, "
            f"but {len(positional)} were supplied."
        )

    positional_parameters = parameters[:len(positional)]
    supplied_names = {item["name"] for item in positional_parameters}
    positional_by_name = {item["name"]: item for item in parameters}
    for parameter, value in zip(positional_parameters, positional):
        if (
            isinstance(value, dict)
            and set(value) == {"name", "value"}
            and value.get("name") == parameter["name"]
        ):
            raise SourceAnalysisError(
                f"{label} positional argument {parameter['name']!r} is wrapped as a "
                "named argument object. Supply its raw value in the positional array "
                "or pass it through keyword_inputs."
            )
    keyword_only = {
        item["name"]: item
        for item in callable_info.get("keyword_only_parameters", [])
    }
    for name in keyword:
        if not isinstance(name, str) or not name.isidentifier():
            raise SourceAnalysisError(f"{label} has an invalid keyword argument name.")
        if name in supplied_names:
            raise SourceAnalysisError(
                f"{label} supplies {name!r} both positionally and by keyword."
            )
        parameter = positional_by_name.get(name) or keyword_only.get(name)
        if parameter is None:
            if not callable_info.get("accepts_var_keyword"):
                raise SourceAnalysisError(
                    f"{label} does not accept a keyword argument named {name!r}."
                )
        elif parameter.get("kind") == "positional_only":
            raise SourceAnalysisError(
                f"{label} parameter {name!r} is positional-only."
            )
    for parameter in parameters:
        if not allow_missing_required and parameter["required"] and (
            parameter["name"] not in supplied_names
            and parameter["name"] not in keyword
        ):
            raise SourceAnalysisError(
                f"{label} is missing required argument {parameter['name']!r}."
            )
    for parameter in keyword_only.values():
        if (
            not allow_missing_required
            and parameter["required"]
            and parameter["name"] not in keyword
        ):
            raise SourceAnalysisError(
                f"{label} is missing required keyword-only argument "
                f"{parameter['name']!r}."
            )


def _validate_setup_references(
    value: Any, step_count: int, stored_names: set[str], label: str
) -> None:
    if isinstance(value, dict):
        if "$setup_return" in value:
            index = value["$setup_return"]
            if (
                set(value) != {"$setup_return"}
                or not isinstance(index, int)
                or isinstance(index, bool)
                or not 0 <= index < step_count
            ):
                raise SourceAnalysisError(f"{label} has an invalid setup return reference.")
            return
        if "$setup_ref" in value:
            name = value["$setup_ref"]
            if set(value) != {"$setup_ref"} or name not in stored_names:
                raise SourceAnalysisError(f"{label} has an unknown setup value reference.")
            return
        if "$actual_return" in value:
            raise SourceAnalysisError(f"{label} cannot reference a target return value.")
        for item in value.values():
            _validate_setup_references(item, step_count, stored_names, label)
    elif isinstance(value, list):
        for item in value:
            _validate_setup_references(item, step_count, stored_names, label)


def validate_case_target(
    case: dict[str, Any], inspection: dict[str, Any]
) -> None:
    """Reject generated calls that do not match an AST-discovered target/signature."""
    target_class = case.get("target_class") or ""
    function_name = case.get("function_name", "")
    target_kind = case.get("target_kind", "instance_method" if target_class else "function")
    inputs = case.get("inputs", [])
    keyword_inputs = case.get("keyword_inputs", {})
    constructor_inputs = case.get("constructor_inputs", [])
    constructor_keyword_inputs = case.get("constructor_keyword_inputs", {})
    setup = case.get("setup", [])
    allow_missing_required = case.get("expected_exception") == "TypeError"
    if not isinstance(setup, list):
        raise SourceAnalysisError("Setup steps must be an array.")

    if target_class:
        class_info = next(
            (item for item in inspection["classes"] if item["name"] == target_class),
            None,
        )
        if class_info is None:
            raise SourceAnalysisError(f"Unknown target class: {target_class}.")
        method = next(
            (item for item in class_info["methods"] if item["name"] == function_name),
            None,
        )
        if method is None:
            raise SourceAnalysisError(
                f"{target_class} has no inspected method named {function_name}."
            )
        if method["async"]:
            raise SourceAnalysisError("Asynchronous methods are not supported yet.")
        if target_kind != method["kind"]:
            raise SourceAnalysisError(
                f"Target kind {target_kind!r} does not match "
                f"{target_class}.{function_name} ({method['kind']})."
            )
        if method["kind"] == "instance_method":
            constructor = class_info["constructor"]
            if constructor is None:
                if constructor_inputs or constructor_keyword_inputs:
                    raise SourceAnalysisError(
                        f"{target_class} has no constructor accepting arguments."
                    )
            elif constructor["async"]:
                raise SourceAnalysisError("Asynchronous constructors are not supported.")
            else:
                _validate_call_arguments(
                    constructor,
                    constructor_inputs,
                    constructor_keyword_inputs,
                    f"{target_class} constructor (__init__)",
                    allow_missing_required=allow_missing_required,
                )
            setup_methods = class_info["methods"]
        else:
            if constructor_inputs or constructor_keyword_inputs:
                raise SourceAnalysisError(
                    f"{target_class} {method['kind']} calls do not accept constructor inputs."
                )
            setup_methods = class_info["methods"]
        _validate_call_arguments(
            method,
            inputs,
            keyword_inputs,
            f"{target_class}.{function_name}",
            allow_missing_required=allow_missing_required,
        )
        _validate_expected_contract(case, method["return"], target_class, function_name)

        stored_names: set[str] = set()
        for index, step in enumerate(setup):
            if not isinstance(step, dict):
                raise SourceAnalysisError("Each setup step must be an object.")
            step_name = step.get("method")
            step_method = next(
                (item for item in setup_methods if item["name"] == step_name), None
            )
            if step_method is None:
                raise SourceAnalysisError(
                    f"{target_class} has no inspected setup method named {step_name!r}."
                )
            if step_method["async"]:
                raise SourceAnalysisError("Asynchronous setup methods are not supported.")
            if (
                method["kind"] == "instance_method"
                and step_method["kind"] != "instance_method"
            ):
                raise SourceAnalysisError(
                    f"Setup method {target_class}.{step_name} must be an instance method."
                )
            if (
                method["kind"] in {"class_method", "static_method"}
                and step_method["kind"] not in {"class_method", "static_method"}
            ):
                raise SourceAnalysisError(
                    f"Setup method {target_class}.{step_name} must be callable on the class."
                )
            step_inputs = step.get("inputs", [])
            step_keywords = step.get("keyword_inputs", {})
            _validate_setup_references(
                [step_inputs, step_keywords], index, stored_names,
                f"Setup step {index + 1}",
            )
            _validate_call_arguments(
                step_method,
                step_inputs,
                step_keywords,
                f"Setup method {target_class}.{step_name}",
                allow_missing_required=allow_missing_required,
            )
            store_as = step.get("store_as", "")
            if store_as:
                if not isinstance(store_as, str) or not store_as.isidentifier():
                    raise SourceAnalysisError(
                        f"Setup step {index + 1} has an invalid return variable name."
                    )
                if store_as in stored_names:
                    raise SourceAnalysisError(
                        f"Setup return variable {store_as!r} is assigned more than once."
                    )
                stored_names.add(store_as)

        _validate_setup_references(
            [inputs, keyword_inputs], len(setup), stored_names, "Target call"
        )
        return

    if target_kind != "function":
        raise SourceAnalysisError(
            "A class target is required for method test cases; methods are never "
            "resolved as module-level functions."
        )
    if constructor_inputs or constructor_keyword_inputs or setup:
        raise SourceAnalysisError(
            "Module-level function tests cannot specify constructor inputs or setup steps."
        )
    function = next(
        (item for item in inspection["functions"] if item["name"] == function_name),
        None,
    )
    if function is None:
        raise SourceAnalysisError(
            f"No module-level function named {function_name} was found. "
            "Specify its exact target_class if this is a method."
        )
    if function["async"]:
        raise SourceAnalysisError("Asynchronous functions are not supported yet.")
    _validate_call_arguments(
        function,
        inputs,
        keyword_inputs,
        function_name,
        allow_missing_required=allow_missing_required,
    )
    _validate_expected_contract(case, function["return"], "", function_name)


def _validate_expected_contract(
    case: dict[str, Any],
    contract: dict[str, Any],
    target_class: str,
    function_name: str,
) -> None:
    if case.get("expected_exception"):
        return
    expected = case.get("expected")
    if isinstance(expected, dict) and "$type" in expected:
        expected_type = expected["$type"]
    elif expected is None:
        expected_type = "none"
    elif isinstance(expected, bool):
        expected_type = "boolean"
    elif isinstance(expected, int):
        expected_type = "integer"
    elif isinstance(expected, float):
        expected_type = "number"
    elif isinstance(expected, str):
        expected_type = "string"
    elif isinstance(expected, list):
        expected_type = "array"
    elif isinstance(expected, dict):
        expected_type = "object"
    else:
        expected_type = "unknown"

    return_kind = contract["kind"]
    compatible = {
        "integer": {"integer"},
        "number": {"number", "integer"},
        "boolean": {"boolean"},
        "string": {"string"},
        "none": {"none"},
        "collection": {"array"},
        "object": {"object"},
        "row_or_none": {"row", "none"},
    }.get(return_kind)
    if compatible is not None and expected_type not in compatible:
        target = f"{target_class}.{function_name}" if target_class else function_name
        raise SourceAnalysisError(
            f"Expected output type {expected_type!r} does not match the inspected "
            f"return contract for {target} ({return_kind})."
        )
