#!/usr/bin/env python3
"""
06_code_check.py
Validation script for the CIFAR-10 competition task.
Checks that a saved model can be loaded and evaluated on train and dev sets.
"""

import argparse
import sys
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time

from cifar10 import CIFAR10

# ==================================================================
# TODO: SET YOUR MODEL PATH AND PARAMETERS HERE
#
# MODEL_PATH      – path to the saved .pt model file produced by
#                   torch.save(model, ...) or torch.save(model.state_dict(), ...)
#                   in your training script.
#
# USE_WEIGHTS_ONLY – set to True if you saved only the state_dict
#                    (torch.save(model.state_dict(), ...)).
#                    Set to False if you saved the whole model
#                    (torch.save(model, ...)).
#
# BATCH_SIZE      – batch size for the evaluation data loader.
#                   Does not affect results, only speed. You can change it.
#
# ==================================================================
MODEL_PATH         = "model.pt"       # <-- CHANGE to your saved model path
USE_WEIGHTS_ONLY   = False            # <-- True if you saved state_dict only
BATCH_SIZE         = 100
SEED               = 42
# ==================================================================


# -----------------------------------------------------------------------
# TODO:
#
# Implement load_the_model() so that it returns your trained model
# ready for evaluation.
#
# Option A – whole model saved with torch.save(model, path):
#     Set USE_WEIGHTS_ONLY = False above.  If your model class is defined
#     in your training script (e.g. "cifar10_train.py"), you may need to
#     import it and patch it into __main__ so that torch.load can unpickle
#     it.  See the example below.
#
# Option B – only state_dict saved with torch.save(model.state_dict(), path):
#     Set USE_WEIGHTS_ONLY = True above.  You must construct the model
#     architecture here and then call model.load_state_dict(...).
#
# Examples:
#
#   # Option A – whole model (patch class into __main__ if needed):
#   import cifar10_train  # your training script
#   sys.modules['__main__'].MyCNN = cifar10_train.MyCNN
#   model = torch.load(args.model, map_location="cpu", weights_only=False)
#
#   # Option B – state_dict only:
#   from cifar10_train import MyCNN
#   model = MyCNN()
#   checkpoint = torch.load(args.model, map_location="cpu")
#   model.load_state_dict(checkpoint)
#
# -----------------------------------------------------------------------


def load_the_model(args):
    """Load a previously saved CIFAR-10 model for evaluation.

    Modify this function to match how you saved your model.

    Returns
    -------
    torch.nn.Module
        The loaded model in eval mode, moved to the requested device.
    """
    if not os.path.isfile(args.model):
        print(f"[FAIL] Model file '{args.model}' not found.")
        sys.exit(1)

    if USE_WEIGHTS_ONLY:
        # TODO: Construct your model architecture here, then load weights.
        # Example:
        #   from cifar10_train import MyCNN
        #   model = MyCNN()
        #   checkpoint = torch.load(args.model, map_location="cpu")
        #   model.load_state_dict(checkpoint)
        raise NotImplementedError(
            "You chose USE_WEIGHTS_ONLY=True but have not implemented "
            "the model construction in load_the_model(). "
            "Please define your model architecture and load the state_dict."
        )
    else:
        # Whole model saved with torch.save(model, ...).
        # If your model class was defined in __main__ of your training
        # script, you need to patch it here.  Uncomment and adapt:
        #
        #   import cifar10_train
        #   sys.modules['__main__'].MyCNN = cifar10_train.MyCNN
        #
        model = torch.load(args.model, map_location="cpu", weights_only=False)

    model.to(args.device)
    model.eval()
    return model


def eval_model(model, device, data_loader, criterion):
    """Evaluate model accuracy and loss on a given data loader."""
    model.eval()
    start_time = time.time()
    total_loss, total_correct = 0, 0
    num_batches = len(data_loader)

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(data_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            predicted = torch.argmax(outputs.data, 1)
            total_correct += (predicted == labels).sum().item()
            batch_idx += 1
            if batch_idx % 100 == 0:
                print(f"eval loss: {loss.item():>7f}  [{batch_idx:>5d}/{num_batches:>5d}]")

    acc = total_correct / len(data_loader.dataset)
    avg_loss = total_loss / num_batches
    elapsed = time.time() - start_time
    return acc, avg_loss, elapsed


def check_model_structure(model, device):
    """Print model info and run a sanity forward pass with a dummy CIFAR-10 input."""
    ok = True

    print(f"Model type: {type(model).__name__}")
    print(f"Architecture:\n{model}")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")

    print()
    print("--- Sanity forward pass (dummy input) ---")
    try:
        # CIFAR-10 images: 3x32x32, float32
        dummy = torch.zeros(1, CIFAR10.C, CIFAR10.H, CIFAR10.W, dtype=torch.float32).to(device)
        with torch.no_grad():
            output = model(dummy)
        print(f"[PASS] Forward pass succeeded. Output shape: {tuple(output.shape)}")

        if output.shape[-1] != CIFAR10.LABELS:
            print(f"[FAIL] Expected {CIFAR10.LABELS} output classes, got {output.shape[-1]}.")
            ok = False
        else:
            print(f"[PASS] Output size ({output.shape[-1]}) matches CIFAR10.LABELS.")
    except Exception as e:
        print(f"[FAIL] Forward pass raised an exception: {e}")
        import traceback; traceback.print_exc()
        ok = False

    return ok


def main():
    print("=" * 64)
    print("  Code Check for CIFAR-10 Competition – Model Evaluation")
    print("=" * 64)

    args = argparse.Namespace(
        model=MODEL_PATH,
        batch_size=BATCH_SIZE,
        seed=SEED,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print(f"Using device: {args.device}")

    # --- Step 1: Load model ---
    print()
    print("--- Loading model ---")
    model = load_the_model(args)
    print(f"[PASS] Model loaded from '{args.model}'.")
    print()

    # --- Step 2: Structural checks ---
    print("--- Checking model structure ---")
    structure_ok = check_model_structure(model, args.device)
    print()

    # --- Step 3: Prepare data ---
    print("--- Preparing data ---")
    try:
        cifar10 = CIFAR10()
        train_loader = DataLoader(cifar10.train, batch_size=args.batch_size, shuffle=False)
        dev_loader = DataLoader(cifar10.dev, batch_size=args.batch_size, shuffle=False)
        print(f"[PASS] CIFAR10 data loaded.")
        print(f"       Train samples: {len(cifar10.train)}")
        print(f"       Dev samples  : {len(cifar10.dev)}")
    except Exception as e:
        print(f"[FAIL] Could not load CIFAR10 data: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)
    print()

    # --- Step 4: Evaluate on train and dev ---
    criterion = nn.CrossEntropyLoss()
    eval_ok = True

    print("--- Evaluating on TRAIN set ---")
    try:
        train_acc, train_loss, train_time = eval_model(model, args.device, train_loader, criterion)
        print(f"  train_loss: {train_loss:.4f}  train_acc: {train_acc:.4f}  time: {train_time:.2f} s")
    except Exception as e:
        print(f"[FAIL] Evaluation on train set raised an exception: {e}")
        import traceback; traceback.print_exc()
        train_acc = 0.0
        eval_ok = False
    print()

    print("--- Evaluating on DEV set ---")
    try:
        dev_acc, dev_loss, dev_time = eval_model(model, args.device, dev_loader, criterion)
        print(f"  dev_loss: {dev_loss:.4f}  dev_acc: {dev_acc:.4f}  time: {dev_time:.2f} s")
    except Exception as e:
        print(f"[FAIL] Evaluation on dev set raised an exception: {e}")
        import traceback; traceback.print_exc()
        dev_acc = 0.0
        eval_ok = False
    print()

    # --- Step 5: Points estimation ---
    print("--- Points estimation ---")
    dev_pct = dev_acc * 100
    if dev_pct >= 70.50:
        points = max(0, round(dev_pct) - 70)
        points = min(points, 30)
        print(f"  Dev accuracy: {dev_pct:.2f}% -> estimated {points} points (before due date)")
    else:
        print(f"  Dev accuracy: {dev_pct:.2f}% -> below 70.50% threshold, 0 points")

    if dev_pct < 70.50:
        print("[WARN] Dev accuracy is below the 70.50% minimum threshold.")
    elif dev_pct < 85.0:
        print(f"[INFO] After the due date, accuracy >= 85.00% is needed for 15 points.")
    print()

    # --- Summary ---
    print("=" * 64)
    print(f"  RESULTS:  train_acc = {train_acc:.4f}   dev_acc = {dev_acc:.4f}")
    if structure_ok and eval_ok:
        print("  ALL CHECKS PASSED")
    else:
        print("  SOME CHECKS FAILED — please review the output above.")
    print("=" * 64)


if __name__ == "__main__":
    main()
