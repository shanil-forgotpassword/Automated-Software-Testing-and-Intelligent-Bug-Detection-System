import sys


def main() -> None:
    sys.path.insert(0, r"c:\Users\vijay\Desktop\code guard ai")
    from codeguard.ai import generate_test_cases, MODEL, OLLAMA_NUM_PREDICT

    print(f"Model: {MODEL}")
    print(f"Num predict: {OLLAMA_NUM_PREDICT}")
    source = """def calculate_grade(name, marks):
    if not name or not name.strip():
        raise ValueError("Name cannot be empty")
    if not isinstance(marks, list) or len(marks) == 0:
        raise ValueError("Marks must be a non-empty list")
    for m in marks:
        if not isinstance(m, (int, float)):
            raise TypeError("All marks must be numbers")
        if m < 0 or m > 100:
            raise ValueError("Marks must be between 0 and 100")
    average = sum(marks) / len(marks)
    if average >= 90:
        return "A"
    elif average >= 80:
        return "B"
    elif average >= 70:
        return "C"
    elif average >= 60:
        return "D"
    else:
        return "F"
"""
    project = {"title": "Student Grade System", "description": "Grade management"}
    requirements = [
        {"id": "REQ-001", "description": "Calculate student grade from marks"}
    ]
    try:
        cases = generate_test_cases(project, requirements, source_code=source)
        print(f"Generated {len(cases)} test cases")
        for index, case in enumerate(cases[:15], start=1):
            print(f"  {index}. [{case['category']}] {case['name']}")
        if len(cases) > 15:
            print(f"  ... and {len(cases) - 15} more")
        if cases.failed_batches:
            print("Incomplete batches:")
            for batch in cases.failed_batches:
                print(f"  {batch}")
    except Exception as exc:
        print(f"Error: {exc}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
