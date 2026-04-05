#!/usr/bin/env python3
"""
02_code_check.py
Quick self-check for the MNIST task.
Answer each question YES or NO.
"""


def ask(question):
    """Ask a YES/NO question, return True for YES."""
    while True:
        answer = input(f"{question} (yes/no): ").strip().lower()
        if answer in ("yes", "y"):
            return True
        if answer in ("no", "n"):
            return False
        print("  Please answer 'yes' or 'no'.")


def main():
    print("=" * 64)
    print("  Code Check for MNIST – Quick Self-Assessment")
    print("=" * 64)
    print()

    questions = [
        "Does your program allow training with no hidden layers (i.e. zero hidden layers)?",
        "Does your program allow different numbers of neurons in different hidden layers?",
        "Does your program allow training the same model for an additional number of epochs?",
        "Are the originally clicked points still visible after colour interpolation of the whole area?",
        "Is your program fast enough? Can it train a neural network model with only a few neurons in at most 5 clicks of the Train More Epochs button (or within 5 seconds in total)?",
        "Is your program able to show the whole coloured area after training in only a short time (e.g. 2 seconds)?"
    ]

    results = []
    for i, q in enumerate(questions, 1):
        answer = ask(f"  Q{i}: {q}")
        results.append((q, answer))

    # --- Summary ---
    print()
    print("=" * 64)
    all_yes = all(ok for _, ok in results)
    for q, ok in results:
        tag = "[PASS]" if ok else "[FAIL]"
        print(f"  {tag} {q}")

    print()
    if all_yes:
        print("  ALL CHECKS PASSED")
    else:
        print("  SOME CHECKS FAILED — please review your implementation.")
    print("=" * 64)


if __name__ == "__main__":
    main()
