#!/usr/bin/env python3
"""
04_code_check.py
Validation script for the uppercase task.
Checks that a saved model can be loaded independently of the training
pipeline and evaluated using the eval_model() and predict_model()
functions from uppercase.py (uppercase.py).
"""

import argparse
import importlib
import sys
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ── Import evaluation functions directly from the training module ─────
# The module file is named with a hyphen ("04-uppercase.py"), which is
# not a valid Python identifier, so we use importlib to load it.
_uppercase_module = importlib.import_module("uppercase")   # adjust name if needed
eval_model     = _uppercase_module.eval_model
predict_model  = _uppercase_module.predict_model

# We also need UppercaseData to build data loaders for evaluation.
# It is imported inside the training module; re-import it here directly.
from uppercase_data import UppercaseData


# ==================================================================
# TODO: SET YOUR MODEL PATH AND EVALUATION PARAMETERS HERE
#
# MODEL_PATH          – path to the saved .pt model file produced by
#                       torch.save(model, ...) in your training script.
#
# WINDOW              – the same window size you used during training
#                       (args.window).  The data loader must recreate
#                       identical (2*window+1)-character windows.
#
# ALPHABET_SIZE       – the value you passed as --alphabet_size_used to
#                       the training script.  This is the number of most
#                       frequent characters *before* UppercaseData adds
#                       pad/unk tokens.  If you left it as None during
#                       training, leave it as None here too.
#
#   NOTE: The *actual* alphabet_size_used (after pad/unk, typically
#   ALPHABET_SIZE + 2 but sometimes less) is stored in the model as
#   model.alphabet_size_used and will be read automatically after
#   loading.  The script cross-checks it against the data loader.
#
# BATCH_SIZE          – batch size for the evaluation data loader.
#                       Does not affect results, only speed.
#
# ==================================================================
MODEL_PATH         = "model.pt"     # <-- CHANGE to your saved model path
WINDOW             = None           # <-- CHANGE to the window value used during training  (e.g. 5)
ALPHABET_SIZE      = None           # <-- CHANGE to the --alphabet_size_used arg from training, or leave None
BATCH_SIZE         = 256            # batch size for evaluation (does not affect accuracy)
# ==================================================================

SEED = 42                           # random seed for reproducibility


def make_default_args():
    """Create a namespace with the evaluation parameters defined above."""
    ns = argparse.Namespace(
        model=MODEL_PATH,
        window=WINDOW,
        alphabet_size=ALPHABET_SIZE,
        batch_size=BATCH_SIZE,
        seed=SEED,
        device="cpu",
    )
    return ns


# -----------------------------------------------------------------------
# TODO:
#
# If you need a different loading strategy (e.g. loading only state_dict
# into a manually constructed architecture), replace the body of
# load_the_model(args) with your own implementation.  Make sure it:
#     1) Accepts `args` (at minimum args.model and args.device).
#     2) Returns a torch.nn.Module already in .eval() mode.
#     3) Works with eval_model() / predict_model() from 04-uppercase.py,
#        which feed the model batches of (batch, 2*window+1) int16 tensors
#        and expect (batch, 2) logits as output.
#
# Example skeleton for a custom architecture:
#     from uppercase import SimpleNN          # or your own class
#     model = SimpleNN(
#         input_size=2 * args.window + 1,
#         alphabet_size_used=<read from saved state or model attribute>,
#         hidden_layer_sizes=["400"],
#         input_dropout=0, dropout=0,
#         embedding_dimension=None,
#     )
#     state = torch.load(args.model, map_location=args.device)
#     model.load_state_dict(state)
#     model.eval()
#     return model
# -----------------------------------------------------------------------


def load_the_model(args):
    """Load a previously saved Uppercase model for evaluation.

    The default implementation assumes the whole model was serialised with
    ``torch.save(model, path)``.  If you saved only the state_dict, you
    will need to construct the architecture first — see the TODO above.

    Returns
    -------
    torch.nn.Module
        The loaded model in eval mode, moved to the requested device.
    """
    if not os.path.isfile(args.model):
        print(f"[FAIL] Model file '{args.model}' not found.")
        sys.exit(1)

    # torch.save(model, ...) uses pickle, which stores a *reference* to the
    # class — typically "__main__.SimpleNN" when the training script was run
    # directly.  When torch.load() is called HERE, pickle tries to resolve
    # that reference in the CURRENT __main__, so we must patch the class
    # into __main__'s namespace.  If you used a different model class,
    # patch that one instead.
    import uppercase
    sys.modules['__main__'].SimpleNN = uppercase.SimpleNN

    model = torch.load(args.model, map_location="cpu", weights_only=False)
    model.to(args.device)
    model.eval()
    return model


def check_model_structure(model, window):
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
        input_len = 2 * window + 1
        dummy = torch.zeros(1, input_len, dtype=torch.int16)
        with torch.no_grad():
            output = model(dummy)
        print(f"[PASS] Forward pass succeeded.  Output shape: {tuple(output.shape)}")

        if output.shape[-1] != 2:
            print(f"[WARN] Expected 2 outputs (lowercase / uppercase), got {output.shape[-1]}.")
            ok = False
        else:
            print(f"[PASS] Output size ({output.shape[-1]}) is compatible with eval_model().")
    except Exception as e:
        print(f"[FAIL] Forward pass raised an exception: {e}")
        ok = False

    return ok


def main():
    print("=" * 64)
    print("  Code Check for uppercase – Model Loading & Evaluation")
    print("=" * 64)

    args = make_default_args()

    # --- Sanity: required parameters ---
    if args.window is None:
        print("[FAIL] WINDOW is not set.  Please edit the TODO section at the")
        print("       top of this file and provide the window value used during")
        print("       training (e.g. WINDOW = 5).")
        sys.exit(1)

    # --- Step 1: Load model ---
    print()
    print("--- Loading model ---")
    model = load_the_model(args)
    print(f"[PASS] Model loaded from '{args.model}'.")

    # Read the actual alphabet_size_used stored in the model by training.
    # This is len(uppercase_data.train.alphabet) at training time, which is
    # typically args.alphabet_size_used + 2 (pad + unk) but can be less.
    model_alphabet_size = getattr(model, "alphabet_size_used", None)
    if model_alphabet_size is not None:
        print(f"[INFO] model.alphabet_size_used = {model_alphabet_size} "
              f"(read from the saved model)")
    else:
        print("[WARN] model.alphabet_size_used attribute not found — "
              "the model may have been saved without it.")
    print()

    # --- Step 2: Structural checks ---
    print("--- Checking model structure ---")
    structure_ok = check_model_structure(model, args.window)
    print()

    # --- Step 3: Quick evaluation on the dev set ---
    print("--- Preparing data for evaluation ---")
    try:
        uppercase_data = UppercaseData(args.window, args.alphabet_size)
        dev_loader = DataLoader(uppercase_data.dev, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(uppercase_data.test, batch_size=args.batch_size, shuffle=False)

        data_alphabet_size = len(uppercase_data.train.alphabet)
        print(f"[PASS] UppercaseData loaded  (window={args.window}, "
              f"alphabet_size arg={args.alphabet_size}).")
        print(f"       len(alphabet) from data: {data_alphabet_size}")
        print(f"       Dev samples : {len(uppercase_data.dev)}")
        print(f"       Test samples: {len(uppercase_data.test)}")

        # Cross-check: the data loader's alphabet must match the model's
        if model_alphabet_size is not None:
            if data_alphabet_size == model_alphabet_size:
                print(f"[PASS] Alphabet size matches: "
                      f"data ({data_alphabet_size}) == model ({model_alphabet_size}).")
            else:
                print(f"[FAIL] Alphabet size MISMATCH: "
                      f"data ({data_alphabet_size}) != model ({model_alphabet_size}).  "
                      f"Check your ALPHABET_SIZE setting.")
                structure_ok = False
    except Exception as e:
        print(f"[FAIL] Could not load UppercaseData: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)

    print()
    print("--- Running evaluation on the DEV set ---")
    eval_ok = True
    try:
        # eval_model expects: model, device, data_loader, criterion, writer, epoch
        # We create a lightweight TensorBoard writer (or a no-op) just for this check.
        from torch.utils.tensorboard import SummaryWriter
        import tempfile
        tmp_logdir = tempfile.mkdtemp(prefix="04_code_check_")
        writer = SummaryWriter(tmp_logdir)

        criterion = nn.CrossEntropyLoss()
        dev_acc = eval_model(model, args.device, dev_loader, criterion, writer, epoch=0)

        writer.flush()
        writer.close()

        print(f"\n[INFO] Dev accuracy: {dev_acc:.4f}")
        if dev_acc < 0.90:
            print("[WARN] Dev accuracy is below 90 % — something may be wrong.")
            eval_ok = False

    except Exception as e:
        print(f"[FAIL] eval_model() raised an exception: {e}")
        import traceback; traceback.print_exc()
        eval_ok = False

    # --- Step 4 (optional): Quick prediction sanity check ---
    print()
    print("--- Quick prediction sanity check (first 200 characters) ---")
    predict_ok = True
    try:
        predictions = predict_model(model, args.device, dev_loader)
        snippet_len = min(200, len(uppercase_data.dev.text), predictions.shape[0])

        reconstructed = []
        for i in range(snippet_len):
            ch = uppercase_data.dev.text[i]
            if int(predictions[i]) == 1:
                reconstructed.append(ch.upper())
            else:
                reconstructed.append(ch.lower())
        reconstructed = "".join(reconstructed)

        print(f"  Original : {uppercase_data.dev.text[:snippet_len]}")
        print(f"  Predicted: {reconstructed}")
        print("[PASS] predict_model() produced output without errors.")
    except Exception as e:
        print(f"[FAIL] predict_model() raised an exception: {e}")
        import traceback; traceback.print_exc()
        predict_ok = False

    # --- Summary ---
    print()
    print("=" * 64)
    all_ok = structure_ok and eval_ok and predict_ok
    if all_ok:
        print("  ALL CHECKS PASSED")
    else:
        print("  SOME CHECKS FAILED — please review the output above.")
    print("=" * 64)


if __name__ == "__main__":
    main()
