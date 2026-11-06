from __future__ import annotations

import json
import sqlite3
import pytest
from pathlib import Path

from codeguard import execution
from codeguard.source_analysis import analyze_source, validate_case_target
from sandbox import runner


SMART_INVENTORY_CODE = """
import sqlite3
import re


class InventorySystem:

    def __init__(self, database="inventory.db"):
        self.conn = sqlite3.connect(database)
        self.cursor = self.conn.cursor()
        self.create_tables()

    def create_tables(self):
        self.cursor.execute(\"\"\"
            CREATE TABLE IF NOT EXISTS products (
                product_id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                price REAL NOT NULL,
                stock INTEGER NOT NULL
            )
        \"\"\")

        self.cursor.execute(\"\"\"
            CREATE TABLE IF NOT EXISTS customers (
                customer_id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT NOT NULL
            )
        \"\"\")

        self.cursor.execute(\"\"\"
            CREATE TABLE IF NOT EXISTS orders (
                order_id INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_id INTEGER,
                total REAL,
                status TEXT DEFAULT 'PLACED'
            )
        \"\"\")

        self.cursor.execute(\"\"\"
            CREATE TABLE IF NOT EXISTS order_items (
                item_id INTEGER PRIMARY KEY AUTOINCREMENT,
                order_id INTEGER,
                product_id INTEGER,
                quantity INTEGER
            )
        \"\"\")

        self.conn.commit()

    def add_product(self, name, price, stock):
        self.cursor.execute(
            "INSERT INTO products (name, price, stock) VALUES (?, ?, ?)",
            (name, price, stock)
        )
        self.conn.commit()
        return self.cursor.lastrowid

    def add_customer(self, name, email):
        if not re.match(r"[^@]+@[^@]+\\.[^@]+", email):
            raise ValueError("Invalid email address")

        self.cursor.execute(
            "INSERT INTO customers (name, email) VALUES (?, ?)",
            (name, email)
        )
        self.conn.commit()
        return self.cursor.lastrowid

    def get_product(self, product_id):
        self.cursor.execute(
            "SELECT * FROM products WHERE product_id = ?",
            (product_id,)
        )
        return self.cursor.fetchone()

    def create_order(self, customer_id, items, discount=0):
        total = 0

        for product_id, quantity in items:
            product = self.get_product(product_id)

            if product is None:
                raise ValueError("Product not found")

            if product[3] < quantity:
                raise ValueError("Insufficient stock")

            total += product[2] * quantity

        total = total - (total * discount / 100)

        self.cursor.execute(
            "INSERT INTO orders (customer_id, total) VALUES (?, ?)",
            (customer_id, total)
        )

        order_id = self.cursor.lastrowid

        for product_id, quantity in items:
            self.cursor.execute(
                "UPDATE products SET stock = stock - ? WHERE product_id = ?",
                (quantity, product_id)
            )

            self.cursor.execute(
                \"\"\"INSERT INTO order_items
                (order_id, product_id, quantity)
                VALUES (?, ?, ?)\"\"\",
                (order_id, product_id, quantity)
            )

        self.conn.commit()
        return order_id

    def cancel_order(self, order_id):
        self.cursor.execute(
            "SELECT status FROM orders WHERE order_id = ?",
            (order_id,)
        )

        order = self.cursor.fetchone()

        if order is None:
            raise ValueError("Order not found")

        self.cursor.execute(
            "SELECT product_id, quantity FROM order_items WHERE order_id = ?",
            (order_id,)
        )

        items = self.cursor.fetchall()

        for product_id, quantity in items:
            self.cursor.execute(
                "UPDATE products SET stock = stock + ? WHERE product_id = ?",
                (quantity, product_id)
            )

        self.cursor.execute(
            "UPDATE orders SET status = 'CANCELLED' WHERE order_id = ?",
            (order_id,)
        )

        self.conn.commit()

    def close(self):
        self.conn.close()
"""


# 1. Standalone Python function
def test_regression_01_standalone_function(tmp_path):
    code = "def calculate_total(price, quantity):\n    return price * quantity\n"
    case = {
        "id": 1,
        "test_id": "REG-01",
        "name": "Standalone function test",
        "function_name": "calculate_total",
        "target_class": "",
        "target_kind": "function",
        "inputs_json": json.dumps([50, 2]),
        "constructor_inputs_json": json.dumps([]),
        "setup_json": json.dumps([]),
        "expected_json": json.dumps(100),
        "database_assertions_json": json.dumps([]),
        "expected_exception": "",
        "approval_status": "approved",
    }
    result = execution.run_approved_tests(code, [case])
    assert result["status"] == "completed"
    assert result["passed"] == 1
    assert result["failed"] == 0
    assert result["coverage"]["available"]
    assert result["coverage"]["percentage"] > 0


# 2. Class with a simple method
def test_regression_02_class_simple_method():
    code = """
class Greeter:
    def greet(self, name):
        return f"Hello, {name}!"
"""
    case = {
        "id": 2,
        "test_id": "REG-02",
        "name": "Class simple method",
        "function_name": "greet",
        "target_class": "Greeter",
        "target_kind": "instance_method",
        "inputs_json": json.dumps(["Alice"]),
        "constructor_inputs_json": json.dumps([]),
        "setup_json": json.dumps([]),
        "expected_json": json.dumps("Hello, Alice!"),
        "database_assertions_json": json.dumps([]),
        "expected_exception": "",
        "approval_status": "approved",
    }
    result = execution.run_approved_tests(code, [case])
    assert result["status"] == "completed"
    assert result["passed"] == 1


# 3. Class with constructor arguments
def test_regression_03_class_with_constructor_arguments():
    code = """
class BankAccount:
    def __init__(self, initial_balance):
        self.balance = initial_balance

    def deposit(self, amount):
        self.balance += amount
        return self.balance
"""
    case = {
        "id": 3,
        "test_id": "REG-03",
        "name": "Class with constructor arguments",
        "function_name": "deposit",
        "target_class": "BankAccount",
        "target_kind": "instance_method",
        "inputs_json": json.dumps([250]),
        "constructor_inputs_json": json.dumps([1000]),
        "setup_json": json.dumps([]),
        "expected_json": json.dumps(1250),
        "database_assertions_json": json.dumps([]),
        "expected_exception": "",
        "approval_status": "approved",
    }
    result = execution.run_approved_tests(code, [case])
    assert result["status"] == "completed"
    assert result["passed"] == 1


# 4. Class using SQLite
def test_regression_04_class_using_sqlite():
    code = """
import sqlite3

class NoteStore:
    def __init__(self, db_path="__CODEGUARD_TEMP_DB__"):
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("CREATE TABLE IF NOT EXISTS notes (id INTEGER PRIMARY KEY, text TEXT)")
        self.conn.commit()

    def add_note(self, text):
        cur = self.conn.execute("INSERT INTO notes (text) VALUES (?)", (text,))
        self.conn.commit()
        return cur.lastrowid
"""
    case = {
        "id": 4,
        "test_id": "REG-04",
        "name": "Class using SQLite",
        "function_name": "add_note",
        "target_class": "NoteStore",
        "target_kind": "instance_method",
        "inputs_json": json.dumps(["Meeting notes"]),
        "constructor_inputs_json": json.dumps(["__CODEGUARD_TEMP_DB__"]),
        "setup_json": json.dumps([]),
        "expected_json": json.dumps({"$type": "integer", "minimum": 1}),
        "database_assertions_json": json.dumps([{
            "table": "notes",
            "where": {"id": {"$actual_return": True}},
            "values": {"text": "Meeting notes"}
        }]),
        "expected_exception": "",
        "approval_status": "approved",
    }
    result = execution.run_approved_tests(code, [case])
    assert result["status"] == "completed"
    assert result["passed"] == 1


# 5. Method with optional parameters
def test_regression_05_method_with_optional_parameters():
    code = """
class Billing:
    def apply_discount(self, amount, percentage=10):
        return amount - (amount * percentage / 100)
"""
    # Test using default discount
    case1 = {
        "id": 5,
        "test_id": "REG-05A",
        "name": "Default optional param",
        "function_name": "apply_discount",
        "target_class": "Billing",
        "target_kind": "instance_method",
        "inputs_json": json.dumps([200.0]),
        "constructor_inputs_json": json.dumps([]),
        "setup_json": json.dumps([]),
        "expected_json": json.dumps(180.0),
        "database_assertions_json": json.dumps([]),
        "expected_exception": "",
        "approval_status": "approved",
    }
    # Test supplying explicit discount
    case2 = {
        "id": 6,
        "test_id": "REG-05B",
        "name": "Explicit optional param",
        "function_name": "apply_discount",
        "target_class": "Billing",
        "target_kind": "instance_method",
        "inputs_json": json.dumps([200.0, 25]),
        "constructor_inputs_json": json.dumps([]),
        "setup_json": json.dumps([]),
        "expected_json": json.dumps(150.0),
        "database_assertions_json": json.dumps([]),
        "expected_exception": "",
        "approval_status": "approved",
    }
    result = execution.run_approved_tests(code, [case1, case2])
    assert result["status"] == "completed"
    assert result["passed"] == 2


# 6. Method returning an integer
def test_regression_06_method_returning_integer():
    code = """
class Counter:
    def __init__(self):
        self.count = 42

    def get_count(self):
        return self.count
"""
    case = {
        "id": 7,
        "test_id": "REG-06",
        "name": "Method returning integer",
        "function_name": "get_count",
        "target_class": "Counter",
        "target_kind": "instance_method",
        "inputs_json": json.dumps([]),
        "constructor_inputs_json": json.dumps([]),
        "setup_json": json.dumps([]),
        "expected_json": json.dumps({"$type": "integer", "minimum": 40, "maximum": 50}),
        "database_assertions_json": json.dumps([]),
        "expected_exception": "",
        "approval_status": "approved",
    }
    result = execution.run_approved_tests(code, [case])
    assert result["status"] == "completed"
    assert result["passed"] == 1


# 7. Method returning a string
def test_regression_07_method_returning_string():
    code = """
class Formatter:
    def format_title(self, text):
        return text.strip().title()
"""
    case = {
        "id": 8,
        "test_id": "REG-07",
        "name": "Method returning string",
        "function_name": "format_title",
        "target_class": "Formatter",
        "target_kind": "instance_method",
        "inputs_json": json.dumps(["  code guard ai  "]),
        "constructor_inputs_json": json.dumps([]),
        "setup_json": json.dumps([]),
        "expected_json": json.dumps("Code Guard Ai"),
        "database_assertions_json": json.dumps([]),
        "expected_exception": "",
        "approval_status": "approved",
    }
    result = execution.run_approved_tests(code, [case])
    assert result["status"] == "completed"
    assert result["passed"] == 1


# 8. Method raising an exception
def test_regression_08_method_raising_exception():
    code = SMART_INVENTORY_CODE
    case = {
        "id": 9,
        "test_id": "REG-08",
        "name": "Customer invalid email raises ValueError",
        "function_name": "add_customer",
        "target_class": "InventorySystem",
        "target_kind": "instance_method",
        "inputs_json": json.dumps(["John Doe", "invalid-email-address"]),
        "constructor_inputs_json": json.dumps([]),
        "setup_json": json.dumps([]),
        "expected_json": json.dumps(None),
        "database_assertions_json": json.dumps([]),
        "expected_exception": "ValueError",
        "approval_status": "approved",
    }
    result = execution.run_approved_tests(code, [case])
    assert result["status"] == "completed"
    assert result["passed"] == 1


# 9. Method modifying database records
def test_regression_09_method_modifying_database_records():
    code = SMART_INVENTORY_CODE
    # Step 1: add product
    # Step 2: add customer
    # Step 3: create order (modifies product stock in database)
    case = {
        "id": 10,
        "test_id": "REG-09",
        "name": "Create order updates stock and inserts order record",
        "function_name": "create_order",
        "target_class": "InventorySystem",
        "target_kind": "instance_method",
        "setup": [
            {
                "method": "add_product",
                "inputs": ["Mechanical Keyboard", 100.0, 10],
                "store_as": "product_id",
            },
            {
                "method": "add_customer",
                "inputs": ["Alice", "alice@example.com"],
                "store_as": "customer_id",
            },
        ],
        "inputs_json": json.dumps([
            {"$setup_ref": "customer_id"},
            [[{"$setup_ref": "product_id"}, 3]],
        ]),
        "keyword_inputs_json": json.dumps({"discount": 10}),
        "constructor_inputs_json": json.dumps([]),
        "setup_json": json.dumps([
            {
                "method": "add_product",
                "inputs": ["Mechanical Keyboard", 100.0, 10],
                "store_as": "product_id",
            },
            {
                "method": "add_customer",
                "inputs": ["Alice", "alice@example.com"],
                "store_as": "customer_id",
            },
        ]),
        "expected_json": json.dumps({"$type": "integer", "minimum": 1}),
        "database_assertions_json": json.dumps([
            {
                "table": "orders",
                "where": {"order_id": {"$actual_return": True}},
                "values": {
                    "customer_id": {"$setup_ref": "customer_id"},
                    "total": 270.0,
                    "status": "PLACED",
                }
            },
            {
                "table": "products",
                "where": {"product_id": {"$setup_ref": "product_id"}},
                "values": {"stock": 7}  # stock reduced from 10 to 7
            }
        ]),
        "expected_exception": "",
        "approval_status": "approved",
    }
    result = execution.run_approved_tests(code, [case])
    assert result["status"] == "completed"
    assert result["passed"] == 1


# 10. Method with invalid input
def test_regression_10_method_with_invalid_input():
    code = SMART_INVENTORY_CODE
    # Insufficient stock raises ValueError
    case = {
        "id": 11,
        "test_id": "REG-10",
        "name": "Order exceeding stock raises ValueError",
        "function_name": "create_order",
        "target_class": "InventorySystem",
        "target_kind": "instance_method",
        "setup_json": json.dumps([
            {"method": "add_product", "inputs": ["Mouse", 25.0, 2]},
            {"method": "add_customer", "inputs": ["Bob", "bob@example.com"]}
        ]),
        "inputs_json": json.dumps([
            {"$setup_return": 1},
            [[{"$setup_return": 0}, 10]],  # requesting 10 but only 2 in stock
        ]),
        "constructor_inputs_json": json.dumps([]),
        "expected_json": json.dumps(None),
        "database_assertions_json": json.dumps([]),
        "expected_exception": "ValueError",
        "approval_status": "approved",
    }
    result = execution.run_approved_tests(code, [case])
    assert result["status"] == "completed"
    assert result["passed"] == 1


# Methods are never inferred as module-level functions when target_class is absent.
def test_method_target_is_not_inferred_without_target_class():
    case = {
        "id": 101,
        "test_id": "TC-001",
        "name": "test_add_product",
        "function_name": "add_product",
        "target_class": "",
        "target_kind": "function",
        "inputs_json": json.dumps(["Laptop", 50000.0, 10]),
        "constructor_inputs_json": json.dumps([]),
        "setup_json": json.dumps([]),
        "expected_json": json.dumps({"$type": "integer", "minimum": 1}),
        "database_assertions_json": json.dumps([{
            "table": "products",
            "where": {"product_id": {"$actual_return": True}},
            "values": {"name": "Laptop", "price": 50000.0, "stock": 10}
        }]),
        "expected_exception": "",
        "approval_status": "approved",
    }
    with pytest.raises(ValueError, match="target_class"):
        execution.run_approved_tests(SMART_INVENTORY_CODE, [case])


# AST source analysis contract verification for Smart Inventory
def test_smart_inventory_ast_inspection_contracts():
    inspection = analyze_source(SMART_INVENTORY_CODE)
    classes = {cls["name"]: cls for cls in inspection["classes"]}
    assert "InventorySystem" in classes
    inventory = classes["InventorySystem"]
    methods = {m["name"]: m for m in inventory["methods"]}
    
    assert "add_product" in methods
    assert [p["name"] for p in methods["add_product"]["parameters"]] == ["name", "price", "stock"]
    assert methods["add_product"]["return"]["kind"] == "integer"
    
    assert "add_customer" in methods
    assert [p["name"] for p in methods["add_customer"]["parameters"]] == ["name", "email"]
    assert methods["add_customer"]["return"]["kind"] == "integer"
    
    assert "get_product" in methods
    assert [p["name"] for p in methods["get_product"]["parameters"]] == ["product_id"]
    assert methods["get_product"]["return"]["kind"] == "row_or_none"
    
    assert "cancel_order" in methods
    assert methods["cancel_order"]["return"]["kind"] == "none"


def test_saved_case_with_nonexistent_method_is_rejected_before_docker(monkeypatch):
    case = {
        "id": 201,
        "test_id": "TC-007",
        "name": "Invalid customer registration target",
        "function_name": "register_customer",
        "target_class": "",
        "target_kind": "function",
        "inputs_json": json.dumps([{"name": "name", "value": "Rahul"}]),
        "constructor_inputs_json": json.dumps([]),
        "setup_json": json.dumps([]),
        "expected_json": json.dumps(None),
        "database_assertions_json": json.dumps([]),
        "expected_exception": "ValueError",
        "approval_status": "approved",
    }

    def docker_must_not_be_reached():
        pytest.fail("invalid saved test reached Docker")

    monkeypatch.setattr(execution, "_ensure_image", docker_must_not_be_reached)
    with pytest.raises(ValueError, match="register_customer"):
        execution.run_approved_tests(SMART_INVENTORY_CODE, [case])


def test_saved_case_with_method_arguments_in_constructor_is_rejected(monkeypatch):
    case = {
        "id": 202,
        "test_id": "TC-008",
        "name": "Invalid constructor arguments",
        "function_name": "create_order",
        "target_class": "InventorySystem",
        "target_kind": "instance_method",
        "inputs_json": json.dumps([1, [(1, 1)]]),
        "constructor_inputs_json": json.dumps([
            {"name": "customer_id", "value": 1},
            {"name": "products", "value": [{"id": 1, "quantity": 1}]},
            {"name": "discount", "value": 0},
        ]),
        "setup_json": json.dumps([]),
        "expected_json": json.dumps(None),
        "database_assertions_json": json.dumps([]),
        "expected_exception": "ValueError",
        "approval_status": "approved",
    }

    def docker_must_not_be_reached():
        pytest.fail("invalid saved test reached Docker")

    monkeypatch.setattr(execution, "_ensure_image", docker_must_not_be_reached)
    with pytest.raises(ValueError, match="constructor"):
        execution.run_approved_tests(SMART_INVENTORY_CODE, [case])
