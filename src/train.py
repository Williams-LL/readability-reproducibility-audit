import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
import transformers
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, root_mean_squared_error
from scipy.stats import pearsonr

from dataset import prepare_fold_datasets
from model import HybridReadabilityModel


def set_deterministic_seed(seed: int):
    """Fija todas las semillas y desactiva optimizaciones no deterministas de CUDA."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, is_regression: bool) -> dict:
    if is_regression:
        rmse = float(root_mean_squared_error(y_true, y_pred))
        # Evitar NaN si el modelo predice una constante en épocas tempranas
        if np.std(y_pred) < 1e-8:
            r = 0.0
        else:
            r = float(pearsonr(y_true, y_pred)[0])
        return {"rmse": rmse, "pearson_r": r}
    else:
        acc = float(accuracy_score(y_true, y_pred) * 100.0)
        macro_f1 = float(f1_score(y_true, y_pred, average="macro") * 100.0)
        qwk = float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))
        return {"accuracy": acc, "macro_f1": macro_f1, "qwk": qwk}


def evaluate(model, dataloader, criterion, device, is_regression: bool):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            ling_features = batch["ling_features"].to(device)
            targets = batch["target"].to(device)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                outputs = model(input_ids, attention_mask, ling_features)
                loss = criterion(outputs, targets)

            total_loss += loss.item() * len(targets)

            if is_regression:
                preds = outputs.detach().cpu().numpy()
            else:
                preds = torch.argmax(outputs, dim=-1).detach().cpu().numpy()

            all_preds.extend(preds)
            all_targets.extend(targets.cpu().numpy())

    avg_loss = total_loss / len(all_targets)
    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    metrics = compute_metrics(y_true, y_pred, is_regression)
    metrics["val_loss"] = float(avg_loss)
    return metrics, y_true.tolist(), y_pred.tolist()


def run_experiment(args):
    set_deterministic_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    tokenizer = AutoTokenizer.from_pretrained(args.backbone)

    train_ds, val_ds, num_ling_feats, is_regression = prepare_fold_datasets(
        dataset_name=args.dataset,
        fold=args.fold,
        tokenizer=tokenizer,
        split_type=args.split_type,
        feature_family=args.feature_family,
        train_fraction=args.train_fraction,
        seed=args.seed,
        max_len=args.max_len,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False)

    num_outputs = 1 if is_regression else 3
    model = HybridReadabilityModel(
        backbone_name=args.backbone,
        num_ling_features=num_ling_feats,
        num_outputs=num_outputs,
        dropout_rate=args.dropout,
        use_layernorm=not args.no_layernorm,
    ).to(device)

    criterion = nn.MSELoss() if is_regression else nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    total_steps = (len(train_loader) // args.grad_accum) * args.epochs
    warmup_steps = int(total_steps * 0.10)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    start_time = time.time()
    history = []
    best_metric_val = float("inf") if is_regression else -float("inf")
    best_epoch_metrics = None
    best_preds_dump = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            ling_features = batch["ling_features"].to(device)
            targets = batch["target"].to(device)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                outputs = model(input_ids, attention_mask, ling_features)
                loss = criterion(outputs, targets) / args.grad_accum

            scaler.scale(loss).backward()
            train_loss += loss.item() * args.grad_accum * len(targets)

            if (step + 1) % args.grad_accum == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if scaler.get_scale() >= scale_before:
                    scheduler.step()

        train_loss /= len(train_ds)
        val_metrics, y_true, y_pred = evaluate(model, val_loader, criterion, device, is_regression)
        val_metrics["epoch"] = epoch
        val_metrics["train_loss"] = float(train_loss)
        history.append(val_metrics)

        primary_score = val_metrics["rmse"] if is_regression else val_metrics["accuracy"]
        is_better = (primary_score < best_metric_val) if is_regression else (primary_score > best_metric_val)

        if is_better:
            best_metric_val = primary_score
            best_epoch_metrics = val_metrics.copy()
            best_preds_dump = {"y_true": y_true, "y_pred": y_pred}

        print(f"Epoch {epoch}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Metrics: {val_metrics}")

    elapsed_sec = round(time.time() - start_time, 2)
    peak_vram_mb = round(torch.cuda.max_memory_allocated() / (1024 ** 2), 1) if device.type == "cuda" else 0.0

    # Guardar registro completo en JSON
    out_dir = Path("results/logs") / args.exp_name
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = (
        f"{args.dataset}_{args.split_type}_{args.backbone.split('/')[-1]}_"
        f"feat-{args.feature_family}_frac-{args.train_fraction}_f{args.fold}_s{args.seed}"
    )

    log_payload = {
        "run_id": run_id,
        "config": vars(args),
        "environment": {
            "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "peak_vram_mb": peak_vram_mb,
            "runtime_seconds": elapsed_sec,
        },
        "final_epoch": history[-1],
        "best_epoch": best_epoch_metrics,
        "history": history,
        "predictions": best_preds_dump,
    }

    log_path = out_dir / f"{run_id}.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_payload, f, indent=2)

    print(f"\n[OK] Corrida finalizada en {elapsed_sec}s | Peak VRAM: {peak_vram_mb} MB")
    print(f"[OK] Log guardado en -> {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Readability Reproducibility & Stress-Test Trainer")
    parser.add_argument("--exp_name", type=str, default="sanity_check")
    parser.add_argument("--dataset", type=str, choices=["onestop", "clear"], default="onestop")
    parser.add_argument("--split_type", type=str, choices=["stratified", "grouped"], default="grouped")
    parser.add_argument("--backbone", type=str, default="roberta-base")
    parser.add_argument("--feature_family", type=str, choices=["none", "surface", "lexical", "syntactic", "all"], default="all")
    parser.add_argument("--train_fraction", type=float, default=1.0)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--no_layernorm", action="store_true")

    args = parser.parse_args()
    run_experiment(args)