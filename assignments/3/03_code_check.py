#!/usr/bin/env python3
"""
03_code_check.py
Validation script for the gym_cartpole task.
Checks that a saved model can be loaded independently of the training
pipeline and evaluated using the evaluate_model() function from
gym_cartpole.py.
"""

import argparse
import sys
import os

import numpy as np
import torch

# Import the evaluation function directly from the training module
from gym_cartpole import evaluate_model


# ==================================================================
# TODO: SET YOUR MODEL PATH AND EVALUATION PARAMETERS HERE
# ==================================================================
MODEL_PATH = "gym_cartpole_model.pt"   # path to the saved .pt model
# ==================================================================


SEED = 42                               # random seed for reproducibility
EPISODES = 5                            # number of episodes for quick test
RENDER = False                          # set True to watch the agent play


def make_default_args():
    """Create a namespace mimicking argparse defaults but with only one epoch."""
    import argparse
    # --- Build args namespace ---
    ns = argparse.Namespace(
        model=MODEL_PATH,
        seed=SEED,
        episodes=EPISODES,
        render=RENDER,
        device="cpu",
    )
    
    return ns

#-----------------------------------------------------------------------
#TODO:
#
#If you need a different loading strategy (e.g. loading only state_dict
#into a manually constructed architecture), replace the body of this
#function load_the_model(args) with your own implementation.  Make sure it:
#    1) Accepts `args` (at minimum args.model and args.device).
#    2) Returns a torch.nn.Module already in .eval() mode.
#    3) Works with evaluate_model() from gym_cartpole.py, which feeds
#       the model a (1, 4) float32 tensor and expects either 1 or 2
#       output values.
#
#Example skeleton for a custom architecture:
#    model = MyCustomNetwork(...)
#    state = torch.load(args.model, map_location=args.device)
#    model.load_state_dict(state)
#    model.eval()
#    return model
#-----------------------------------------------------------------------
    
    
def load_the_model(args):
    """Load a previously saved CartPole model for evaluation.

    Returns
    -------
    torch.nn.Module
        The loaded model in eval mode, moved to the requested device.
    """
        
    if not os.path.isfile(args.model):
        print(f"[FAIL] Model file '{args.model}' not found.")
        sys.exit(1)

    model = torch.load(args.model, map_location="cpu", weights_only=False)
    model.to(args.device)
    model.eval()
    return model


def check_model_structure(model):
    """Print basic info about the loaded model and run a sanity forward pass."""
    ok = True

    # --- Architecture overview ---
    print(f"Model type: {type(model).__name__}")
    print(f"Architecture:\n{model}")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")

    # --- Sanity forward pass with a dummy observation ---
    print()
    print("--- Sanity forward pass (dummy input) ---")
    try:
        dummy = torch.zeros(1, 4, dtype=torch.float32)
        with torch.no_grad():
            output = model(dummy)
        print(f"[PASS] Forward pass succeeded. Output shape: {tuple(output.shape)}")

        if output.shape[-1] not in (1, 2):
            print(f"[WARN] Expected 1 or 2 outputs, got {output.shape[-1]}. "
                  "evaluate_model() may not handle this.")
            ok = False
        else:
            print(f"[PASS] Output size ({output.shape[-1]}) is compatible with evaluate_model().")
    except Exception as e:
        print(f"[FAIL] Forward pass raised an exception: {e}")
        ok = False

    return ok


def main():
    print("=" * 60)
    print("  Code Check for gym_cartpole – Model Loading & Evaluation")
    print("=" * 60)


    # --- Step 1: Load model ---
    print("--- Loading model ---")
    args = make_default_args()
    model = load_the_model(args)
    print(f"[PASS] Model loaded from '{args.model}'.")
    print()

    # --- Step 2: Structural checks ---
    print("--- Checking model structure ---")
    structure_ok = check_model_structure(model)
    print()

    # --- Step 3: Quick evaluation ---
    print(f"--- Running quick evaluation ({args.episodes} episodes) ---")
    try:
        avg_score = evaluate_model(
            model,
            seed=args.seed,
            episodes=args.episodes,
            render=args.render,
            report_per_episode=True,
            device=args.device,
        )
        print(f"\nAverage score over {args.episodes} episodes: {avg_score}")
        print("[PASS] Agent appears to solve CartPole-v1.")
        eval_ok = True

    except Exception as e:
        print(f"[FAIL] evaluate_model() raised an exception: {e}")
        import traceback
        traceback.print_exc()
        eval_ok = False

    # --- Summary ---
    print()
    print("=" * 60)
    if structure_ok and eval_ok:
        print("  ALL CHECKS PASSED")
    else:
        print("  SOME CHECKS FAILED — please review the output above.")
    print("=" * 60)


if __name__ == "__main__":
    main()
