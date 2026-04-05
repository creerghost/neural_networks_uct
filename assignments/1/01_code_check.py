#!/usr/bin/env python3
"""
01_code_check.py
Validation script for the mnist_layers_activations_stdev task.
Checks that the submitted code:
  1) Returns a tuple from main()
  2) The tuple contains two numeric values (mean, stdev)
  3) Generates outputValue.txt with the correct format
     (two lines, each with exactly six decimal places)
"""

import importlib
import sys
import os
import types

# ============================================================
# TODO: SET YOUR SUBMITTED FILENAME HERE (without the .py extension)
# ============================================================
SUBMISSION_FILENAME = "mnist_layers_activations_stdev2"
# ============================================================

def make_default_args():
    """Create a namespace mimicking argparse defaults but with only one epoch."""
    import argparse
    ns = argparse.Namespace(
        activation="relu",
        batch_size=50,
        epochs=1,
        hidden_layer_size=100,
        layers=1,
        seed=42,
        threads=0,
    )
    return ns


def check_tuple(result):
    """Check that main() returned a tuple of two numbers."""
    ok = True

    if not isinstance(result, tuple):
        print(f"[FAIL] main() did not return a tuple. Got type: {type(result).__name__}")
        ok = False
    else:
        print("[PASS] main() returned a tuple.")

    if ok and len(result) != 2:
        print(f"[FAIL] Tuple should have exactly 2 elements (mean, stdev). Got {len(result)} elements.")
        ok = False
    elif ok:
        print("[PASS] Tuple has exactly 2 elements.")

    if ok:
        mean_val, stdev_val = result
        for name, val in [("mean", mean_val), ("stdev", stdev_val)]:
            try:
                float(val)
                print(f"[PASS] {name} is numeric (value={val}).")
            except (TypeError, ValueError):
                print(f"[FAIL] {name} is not numeric. Got: {val!r}")
                ok = False

    return ok


def check_output_file():
    """Check that outputValue.txt exists and has the required format."""
    filepath = "outputValue.txt"
    ok = True

    if not os.path.isfile(filepath):
        print(f"[FAIL] '{filepath}' was not found in the current directory.")
        return False
    else:
        print(f"[PASS] '{filepath}' exists.")

    with open(filepath, "r") as f:
        content = f.read()

    lines = content.strip().splitlines()

    if len(lines) != 2:
        print(f"[FAIL] '{filepath}' should have exactly 2 lines. Found {len(lines)} line(s).")
        print(f"       Content: {content!r}")
        ok = False
    else:
        print(f"[PASS] '{filepath}' has exactly 2 lines.")

    if ok:
        for i, (label, line) in enumerate(zip(["Mean", "Stdev"], lines)):
            line = line.strip()
            # Check it's a valid float
            try:
                val = float(line)
                print(f"[PASS] Line {i+1} ({label}) is a valid number: {val}")
            except ValueError:
                print(f"[FAIL] Line {i+1} ({label}) is not a valid number: {line!r}")
                ok = False
                continue

            # Check exactly six decimal places
            if "." in line:
                decimal_part = line.split(".")[1]
                if len(decimal_part) == 6:
                    print(f"[PASS] Line {i+1} ({label}) has exactly 6 decimal places.")
                else:
                    print(f"[FAIL] Line {i+1} ({label}) has {len(decimal_part)} decimal places (expected 6). Value: {line!r}")
                    ok = False
            else:
                print(f"[FAIL] Line {i+1} ({label}) has no decimal point. Value: {line!r}")
                ok = False

    return ok


def main():
    print("=" * 60)
    print("  Code Check for mnist_layers_activations_stdev task")
    print("=" * 60)
    print(f"Submission module: {SUBMISSION_FILENAME}")
    print()

    # --- Import the submission module ---
    print("--- Importing module ---")
    try:
        module = importlib.import_module(SUBMISSION_FILENAME)
        print(f"[PASS] Successfully imported '{SUBMISSION_FILENAME}'.")
    except Exception as e:
        print(f"[FAIL] Could not import '{SUBMISSION_FILENAME}': {e}")
        sys.exit(1)

    # --- Check that main exists and is callable ---
    if not hasattr(module, "main") or not callable(module.main):
        print(f"[FAIL] Module '{SUBMISSION_FILENAME}' does not have a callable 'main' function.")
        sys.exit(1)
    else:
        print("[PASS] Module has a callable 'main' function.")

    # --- Run main with default args ---
    print()
    print("--- Running main(args) with default arguments ---")
    args = make_default_args()
    print(f"  args = {args}")
    print()

    try:
        result = module.main(args)
    except Exception as e:
        print(f"[FAIL] main(args) raised an exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print()
    print("--- Checking return value ---")
    tuple_ok = check_tuple(result)

    print()
    print("--- Checking outputValue.txt ---")
    file_ok = check_output_file()

    # --- Summary ---
    print()
    print("=" * 60)
    if tuple_ok and file_ok:
        print("  ALL CHECKS PASSED")
    else:
        print("  SOME CHECKS FAILED — please review the output above.")
    print("=" * 60)


if __name__ == "__main__":
    main()
