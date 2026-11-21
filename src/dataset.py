import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

# Índices exactos de las 3 familias en las 22 columnas
FEATURE_FAMILIES = {
    "none": [],
    "surface": list(range(0, 8)),       # 8 features
    "lexical": list(range(8, 12)),      # 4 features
    "syntactic": list(range(12, 22)),   # 10 features
    "all": list(range(0, 22)),          # 22 features
}


class ReadabilityDataset(Dataset):
    def __init__(
        self,
        texts: list[str],
        features: np.ndarray,
        targets: np.ndarray,
        tokenizer,
        max_len: int = 512,
        is_regression: bool = False,
    ):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors="pt",
        )
        self.features = torch.tensor(features, dtype=torch.float32)
        target_dtype = torch.float32 if is_regression else torch.long
        self.targets = torch.tensor(targets, dtype=target_dtype)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "ling_features": self.features[idx],
            "target": self.targets[idx],
        }


def prepare_fold_datasets(
    dataset_name: str,
    fold: int,
    tokenizer,
    split_type: str = "stratified",
    feature_family: str = "all",
    train_fraction: float = 1.0,
    seed: int = 42,
    max_len: int = 512,
):
    """Prepara train y val datasets para un fold sin fuga de datos."""
    if dataset_name == "onestop":
        df = pd.read_csv("data/processed/onestop_manifest.csv")
        raw_feats = np.load("data/processed/onestop_features.npy")
        target_col = "level"
        fold_col = "fold_grouped" if split_type == "grouped" else "fold_stratified"
        is_regression = False
    elif dataset_name == "clear":
        df = pd.read_csv("data/processed/clear_manifest.csv")
        raw_feats = np.load("data/processed/clear_features.npy")
        target_col = "target"
        fold_col = "fold_stratified"
        is_regression = True
    else:
        raise ValueError(f"Dataset desconocido: {dataset_name}")

    train_mask = (df[fold_col] != fold).values
    val_mask = (df[fold_col] == fold).values

    train_idx = np.where(train_mask)[0]
    val_idx = np.where(val_mask)[0]

    # Submuestreo controlado para Experimento C (Escasez de datos: 0.25, 0.50, 1.0)
    if train_fraction < 1.0:
        rng = np.random.default_rng(seed + fold)
        n_sub = max(10, int(len(train_idx) * train_fraction))
        train_idx = rng.choice(train_idx, size=n_sub, replace=False)

    # Selección de columnas según familia lingüística (Experimento B)
    col_indices = FEATURE_FAMILIES[feature_family]
    if len(col_indices) > 0:
        scaler = StandardScaler()
        X_train_feats = scaler.fit_transform(raw_feats[train_idx][:, col_indices])
        X_val_feats = scaler.transform(raw_feats[val_idx][:, col_indices])
    else:
        # Matriz vacía (N, 0) cuando feature_family == "none"
        X_train_feats = np.zeros((len(train_idx), 0), dtype=np.float32)
        X_val_feats = np.zeros((len(val_idx), 0), dtype=np.float32)

    train_ds = ReadabilityDataset(
        texts=df.iloc[train_idx]["text"].tolist(),
        features=X_train_feats,
        targets=df.iloc[train_idx][target_col].values,
        tokenizer=tokenizer,
        max_len=max_len,
        is_regression=is_regression,
    )
    val_ds = ReadabilityDataset(
        texts=df.iloc[val_idx]["text"].tolist(),
        features=X_val_feats,
        targets=df.iloc[val_idx][target_col].values,
        tokenizer=tokenizer,
        max_len=max_len,
        is_regression=is_regression,
    )

    return train_ds, val_ds, len(col_indices), is_regression