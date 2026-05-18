#!/usr/bin/env python3
"""
07_code_check.py
Validation script for the Tagger competition task.
Checks that a saved model can be loaded and evaluated on train, dev,
and test sets, and that prediction files are generated correctly.
"""

import argparse
import sys
import os
import tempfile

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchmetrics
import time

from morpho_dataset import MorphoDataset

# ==================================================================
# TODO: SET YOUR MODEL PATH AND PARAMETERS HERE
#
# MODEL_PATH        – saved .tar model weights file produced by
#                     torch.save(model.state_dict(), ...)
#                     in your training script.
#
#
# BATCH_SIZE        – batch size for the evaluation data loader.
#                     Does not affect results, only speed.
#
# ==================================================================
MODEL_PATH         = "model_weights.tar"       # <-- CHANGE to your saved model path
BATCH_SIZE         = 64
SEED               = 42
# ==================================================================
# TODO: If you need any other batch data processing above MorphoDataset, 
# make it here and adjust the build_vocabularies, make_transform, 
# prepare_batch, eval_model, predict_model, write_predictions, 
# write_predictions_full procedures appropriately.


def build_vocabularies(morpho):
    """Build form and tag lookup dictionaries from the training data.

    This must match exactly what your training script does.
    The default implementation reserves index 0 for PAD and 1 for UNK.

    Returns
    -------
    forms_to_index : dict
    tags_to_index  : dict
    forms_count    : int
    tags_count     : int
    id_to_tag      : np.ndarray
    """
    forms_to_index = {s: i + 2 for i, s in enumerate(morpho.train.forms.word_mapping_vocabulary)}
    forms_count = len(forms_to_index) + 2

    tags_to_index = {t: i + 2 for i, t in enumerate(morpho.train.tags.word_mapping_vocabulary)}
    tags_count = len(tags_to_index) + 2

    id_to_tag = ["<PAD>", "<UNK>", *morpho.train.tags.word_mapping_vocabulary]
    id_to_tag = np.asarray(id_to_tag)

    return forms_to_index, tags_to_index, forms_count, tags_count, id_to_tag



def load_the_model(args):
    """Load a previously saved Tagger model for evaluation.

    Modify this function to match how you saved your model.

    Returns
    -------
    torch.nn.Module
        The loaded model in eval mode, moved to the requested device.
    """

    if not os.path.isfile(args.model_weights):
        print(f"[FAIL] Model weights file '{args.model_weights}' not found.")
        sys.exit(1)

    import tagger_we

    model = tagger_we.SimpleNN(66274, 128, 512, 19)


    checkpoint = torch.load(args.model_weights, map_location="cpu")
    model.load_state_dict(checkpoint)

    model.to(args.device)
    model.eval()
    return model


def make_transform(forms_to_index, tags_to_index):
    """Return a transform function compatible with MorphoDataset.with_transform()."""
    def prepare_tagging_data_transform(sentence):
        form_ids = [forms_to_index.get(w, MorphoDataset.UNK) for w in sentence["forms"]]
        form_ids = torch.as_tensor(form_ids)
        tag_ids = [tags_to_index.get(t, MorphoDataset.UNK) for t in sentence["tags"]]
        tag_ids = torch.as_tensor(tag_ids)
        return form_ids, tag_ids
    return prepare_tagging_data_transform


def prepare_batch(data):
    """Collate function: pad form and tag sequences in a batch."""
    form_ids, tag_ids = zip(*data)
    form_ids_padded = torch.nn.utils.rnn.pad_sequence(
        list(form_ids), batch_first=True, padding_value=MorphoDataset.PAD)
    tag_ids_padded = torch.nn.utils.rnn.pad_sequence(
        list(tag_ids), batch_first=True, padding_value=MorphoDataset.PAD)
    return form_ids_padded, tag_ids_padded


def eval_model(model, device, data_loader, metric, criterion):
    """Evaluate model accuracy and loss on a given data loader."""
    metric = metric.to(device)
    model.eval()
    start_time = time.time()
    total_loss = 0
    metric.reset()
    num_batches = len(data_loader)

    with torch.no_grad():
        for forms, tags in data_loader:
            forms, tags = forms.to(device), tags.to(device)
            outputs = model(forms)
            loss = criterion(outputs, tags)
            total_loss += loss.item()
            metric.update(outputs.detach(), tags)

    acc = metric.compute().cpu().item()
    avg_loss = total_loss / num_batches
    elapsed = time.time() - start_time
    return acc, avg_loss, elapsed


def predict_model(model, device, data_loader):
    """Run prediction and return a list of per-sentence tag-index tensors."""
    model.eval()
    all_outputs = []
    start_time = time.time()

    with torch.no_grad():
        for forms, _ in data_loader:
            forms = forms.to(device)
            outputs = model(forms)
            predicted = torch.argmax(outputs.detach(), 1).cpu()
            for form_seq, preds in zip(forms, predicted):
                real_len = (form_seq != MorphoDataset.PAD).sum().item()
                all_outputs.append(preds[:real_len])

    elapsed = time.time() - start_time
    print(f"  predict_time: {elapsed:.4f} s")
    return all_outputs


def write_predictions(filepath, morpho_dataset, predictions, id_to_tag):
    """Write predictions to a file in the expected vertical format."""
    with open(filepath, "w", encoding="utf-8") as f:
        for sentence, pred_ids in zip(morpho_dataset, predictions):
            forms = sentence["forms"]
            pred_ids_np = np.asarray(pred_ids, dtype=np.int64)
            predicted_tags = id_to_tag[pred_ids_np]
            for form, tag in zip(forms, predicted_tags):
                print(tag, file=f)
            print(file=f)
    return filepath


def write_predictions_full(filepath, morpho_dataset, predictions, id_to_tag):
    """Write predictions with original words and gold labels (tab-separated)."""
    with open(filepath, "w", encoding="utf-8") as f:
        print("word\tpredicted_tag\toriginal_label", file=f)
        for sentence, pred_ids in zip(morpho_dataset, predictions):
            forms = sentence["forms"]
            tags = sentence["tags"]
            pred_ids_np = np.asarray(pred_ids, dtype=np.int64)
            predicted_tags = id_to_tag[pred_ids_np]
            for form, predicted_tag, orig_label in zip(forms, predicted_tags, tags):
                print(form + "\t" + predicted_tag + "\t" + orig_label, file=f)
            print(file=f)
    return filepath


def check_predictions_file(filepath, morpho_dataset):
    """Validate the format of a predictions file against gold data."""
    ok = True
    if not os.path.isfile(filepath):
        print(f"[FAIL] Predictions file '{filepath}' not found.")
        return False
    print(f"[PASS] Predictions file '{filepath}' exists.")

    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    predicted_sentences, in_sentence = [], False
    for line in lines:
        line = line.rstrip("\n")
        if not line:
            in_sentence = False
        else:
            if not in_sentence:
                predicted_sentences.append([])
                in_sentence = True
            predicted_sentences[-1].append(line)

    gold_count = len(morpho_dataset)
    pred_count = len(predicted_sentences)
    if pred_count != gold_count:
        print(f"[FAIL] Number of predicted sentences ({pred_count}) "
              f"does not match gold ({gold_count}).")
        ok = False
    else:
        print(f"[PASS] Sentence count matches ({pred_count}).")

    if ok:
        length_mismatches = 0
        for i, (pred_sent, gold_sent) in enumerate(
                zip(predicted_sentences, morpho_dataset)):
            gold_len = len(gold_sent["forms"])
            pred_len = len(pred_sent)
            if pred_len != gold_len:
                if length_mismatches < 3:
                    print(f"[FAIL] Sentence {i+1}: predicted {pred_len} tags, "
                          f"gold has {gold_len} words.")
                length_mismatches += 1
        if length_mismatches == 0:
            print("[PASS] All sentence lengths match.")
        else:
            print(f"[FAIL] {length_mismatches} sentence(s) have mismatched lengths.")
            ok = False

    return ok


def check_model_structure(model, device, tags_count):
    """Print model info and run a sanity forward pass."""
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
        # Dummy batch: 2 sentences of length 5
        dummy = torch.zeros(2, 5, dtype=torch.long).to(device)
        with torch.no_grad():
            output = model(dummy)
        print(f"[PASS] Forward pass succeeded. Output shape: {tuple(output.shape)}")

        # Output should be [batch, num_tags, seq]
        if output.shape[0] != 2:
            print(f"[WARN] Expected batch dim 2, got {output.shape[0]}.")
        if output.shape[1] != tags_count:
            print(f"[WARN] Expected {tags_count} tag classes in dim 1, got {output.shape[1]}.")
        else:
            print(f"[PASS] Tag dimension ({output.shape[1]}) matches tags_count.")
        if output.shape[2] != 5:
            print(f"[WARN] Expected seq length 5 in dim 2, got {output.shape[2]}.")
    except Exception as e:
        print(f"[FAIL] Forward pass raised an exception: {e}")
        import traceback; traceback.print_exc()
        ok = False

    return ok


def main():
    print("=" * 64)
    print("  Code Check for Tagger Competition – Model Evaluation")
    print("=" * 64)

    args = argparse.Namespace(
        model_weights=MODEL_PATH,
        batch_size=BATCH_SIZE,
        seed=SEED,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print(f"Using device: {args.device}")
    print()

    # --- Step 1: Load data and build vocabularies ---
    print("--- Loading data ---")
    try:
        morpho = MorphoDataset("czech_cac")
        print(f"[PASS] MorphoDataset loaded.")
        print(f"       Train sentences: {len(morpho.train)}")
        print(f"       Dev sentences  : {len(morpho.dev)}")
        print(f"       Test sentences : {len(morpho.test)}")
    except Exception as e:
        print(f"[FAIL] Could not load MorphoDataset: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)

    forms_to_index, tags_to_index, forms_count, tags_count, id_to_tag = \
        build_vocabularies(morpho)
    print(f"       Vocabulary: {forms_count} forms, {tags_count} tags")
    print()

    # --- Step 2: Load model ---
    print("--- Loading model ---")
    model = load_the_model(args)
    print(f"[PASS] Model weights loaded from '{args.model_weights}'.")
    print()

    # --- Step 3: Structural checks ---
    print("--- Checking model structure ---")
    structure_ok = check_model_structure(model, args.device, tags_count)
    print()

    # --- Step 4: Prepare data loaders ---
    print("--- Preparing data loaders ---")
    transform = make_transform(forms_to_index, tags_to_index)
    train_loader = DataLoader(
        morpho.train.with_transform(transform),
        batch_size=args.batch_size, collate_fn=prepare_batch, shuffle=False)
    dev_loader = DataLoader(
        morpho.dev.with_transform(transform),
        batch_size=args.batch_size, collate_fn=prepare_batch, shuffle=False)
    test_loader = DataLoader(
        morpho.test.with_transform(transform),
        batch_size=args.batch_size, collate_fn=prepare_batch, shuffle=False)
    print("[PASS] Data loaders created.")
    print()

    # --- Step 5: Evaluate on train, dev, test ---
    criterion = nn.CrossEntropyLoss(ignore_index=MorphoDataset.PAD)
    metric = torchmetrics.Accuracy(
        "multiclass", num_classes=tags_count, ignore_index=MorphoDataset.PAD)
    eval_ok = True
    dev_acc_pct = 0.0

    for split_name, loader in [("TRAIN", train_loader),
                                ("DEV", dev_loader),
                                ("TEST", test_loader)]:
        print(f"--- Evaluating on {split_name} set ---")
        try:
            acc, loss, elapsed = eval_model(model, args.device, loader, metric, criterion)
            pct = acc * 100
            print(f"  loss: {loss:.4f}  accuracy: {pct:.2f}%  time: {elapsed:.2f} s")
            if split_name == "DEV":
                dev_acc_pct = pct
        except Exception as e:
            print(f"[FAIL] Evaluation on {split_name} set raised an exception: {e}")
            import traceback; traceback.print_exc()
            eval_ok = False
            if split_name == "DEV":
                dev_acc_pct = 0.0
        print()

    # --- Step 6: Generate and validate predictions ---
    predict_ok = True
    for split_name, loader, dataset in [("dev", dev_loader, morpho.dev),
                                         ("test", test_loader, morpho.test)]:
        pred_file = f"predictions_{split_name}.txt"
        print(f"--- Generating predictions on {split_name.upper()} set -> {pred_file} ---")
        try:
            predictions = predict_model(model, args.device, loader)
            write_predictions(pred_file, dataset, predictions, id_to_tag)
            print(f"[PASS] Predictions written to '{pred_file}'.")

            # Also write the full version with words and gold labels
            pred_file_full = f"predictions_{split_name}_full.txt"
            write_predictions_full(pred_file_full, dataset, predictions, id_to_tag)
            print(f"[PASS] Full predictions written to '{pred_file_full}'.")
        except Exception as e:
            print(f"[FAIL] Prediction on {split_name} raised an exception: {e}")
            import traceback; traceback.print_exc()
            predict_ok = False
            continue

        print(f"--- Validating {pred_file} format ---")
        file_ok = check_predictions_file(pred_file, dataset)
        if not file_ok:
            predict_ok = False

        # Cross-check with MorphoDataset.evaluate
        try:
            with open(pred_file, "r", encoding="utf-8") as f:
                eval_acc = MorphoDataset.evaluate(dataset.tags, f.readlines())
            print(f"[INFO] MorphoDataset.evaluate accuracy on {split_name}: {eval_acc:.2f}%")
        except Exception as e:
            print(f"[WARN] MorphoDataset.evaluate failed: {e}")
        print()

    # --- Step 7: Points estimation ---
    print("--- Points estimation ---")
    if dev_acc_pct >= 95.00:
        print(f"  Dev accuracy: {dev_acc_pct:.2f}% -> 20 points (before due date)")
    else:
        print(f"  Dev accuracy: {dev_acc_pct:.2f}% -> below 95.00% threshold")
    print()

    # --- Summary ---
    print("=" * 64)
    all_ok = structure_ok and eval_ok and predict_ok
    if all_ok:
        print("  ALL CHECKS PASSED")
    else:
        print("  SOME CHECKS FAILED — please review the output above.")
    print("=" * 64)


if __name__ == "__main__":
    main()
