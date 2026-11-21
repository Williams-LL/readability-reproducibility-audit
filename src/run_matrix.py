import argparse
from pathlib import Path
from types import SimpleNamespace
from train import run_experiment

SEEDS_5 = [42, 100, 2024, 777, 999]
SEEDS_3 = [42, 100, 2024]
FOLDS = [0, 1, 2, 3, 4]


def build_job_list(block: str) -> list[dict]:
    jobs = []

    if block == "exp_A_onestop":
        # Replicación base + auditoría de fuga (Stratified vs Grouped)
        for split in ["stratified", "grouped"]:
            for feat in ["none", "all"]:
                for fold in FOLDS:
                    for seed in SEEDS_5:
                        jobs.append({
                            "exp_name": "exp_A_onestop",
                            "dataset": "onestop",
                            "split_type": split,
                            "backbone": "roberta-base",
                            "feature_family": feat,
                            "train_fraction": 1.0,
                            "fold": fold,
                            "seed": seed,
                            "epochs": 5,
                        })

    elif block == "exp_A_clear":
        # Segundo benchmark continuo (CLEAR Corpus)
        for feat in ["none", "all"]:
            for fold in FOLDS:
                for seed in SEEDS_3:
                    jobs.append({
                        "exp_name": "exp_A_clear",
                        "dataset": "clear",
                        "split_type": "stratified",
                        "backbone": "roberta-base",
                        "feature_family": feat,
                        "train_fraction": 1.0,
                        "fold": fold,
                        "seed": seed,
                        "epochs": 4,
                    })

    elif block == "exp_B_ablation":
        # Ablación por familia lingüística en OneStopEnglish (sin fuga)
        for feat in ["surface", "lexical", "syntactic"]:
            for fold in FOLDS:
                for seed in SEEDS_3:
                    jobs.append({
                        "exp_name": "exp_B_ablation",
                        "dataset": "onestop",
                        "split_type": "grouped",
                        "backbone": "roberta-base",
                        "feature_family": feat,
                        "train_fraction": 1.0,
                        "fold": fold,
                        "seed": seed,
                        "epochs": 5,
                    })

    elif block == "exp_C_scarcity":
        # Prueba de escasez de datos (25% y 50% de entrenamiento)
        for frac in [0.25, 0.50]:
            for feat in ["none", "all"]:
                for fold in FOLDS:
                    for seed in SEEDS_3:
                        jobs.append({
                            "exp_name": "exp_C_scarcity",
                            "dataset": "onestop",
                            "split_type": "grouped",
                            "backbone": "roberta-base",
                            "feature_family": feat,
                            "train_fraction": frac,
                            "fold": fold,
                            "seed": seed,
                            "epochs": 5,
                        })

    elif block == "exp_D_distilbert":
        # Prueba de despliegue ligero en entornos educativos de bajos recursos
        for feat in ["none", "all"]:
            for fold in FOLDS:
                for seed in SEEDS_3:
                    jobs.append({
                        "exp_name": "exp_D_distilbert",
                        "dataset": "onestop",
                        "split_type": "grouped",
                        "backbone": "distilbert-base-uncased",
                        "feature_family": feat,
                        "train_fraction": 1.0,
                        "fold": fold,
                        "seed": seed,
                        "epochs": 5,
                    })
    else:
        raise ValueError(f"Bloque desconocido: {block}")

    return jobs


def main():
    parser = argparse.ArgumentParser(description="Orquestador de Matriz Experimental")
    parser.add_argument(
        "--block",
        type=str,
        required=True,
        choices=["exp_A_onestop", "exp_A_clear", "exp_B_ablation", "exp_C_scarcity", "exp_D_distilbert"],
    )
    args = parser.parse_args()

    jobs = build_job_list(args.block)
    total = len(jobs)
    print(f"=== Iniciando bloque '{args.block}' ({total} corridas totales) ===\n")

    for idx, job in enumerate(jobs, start=1):
        run_id = (
            f"{job['dataset']}_{job['split_type']}_{job['backbone'].split('/')[-1]}_"
            f"feat-{job['feature_family']}_frac-{job['train_fraction']}_f{job['fold']}_s{job['seed']}"
        )
        expected_log = Path("results/logs") / job["exp_name"] / f"{run_id}.json"

        if expected_log.exists():
            print(f"[{idx}/{total}] SKIP (ya existe): {run_id}")
            continue

        print(f"\n[{idx}/{total}] EJECUTANDO: {run_id}")
        run_args = SimpleNamespace(
            exp_name=job["exp_name"],
            dataset=job["dataset"],
            split_type=job["split_type"],
            backbone=job["backbone"],
            feature_family=job["feature_family"],
            train_fraction=job["train_fraction"],
            fold=job["fold"],
            seed=job["seed"],
            max_len=512,
            batch_size=8,
            grad_accum=2,
            epochs=job["epochs"],
            lr=2e-5,
            weight_decay=0.01,
            dropout=0.1,
            no_layernorm=False,
        )
        run_experiment(run_args)


if __name__ == "__main__":
    main()