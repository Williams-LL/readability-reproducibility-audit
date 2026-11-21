import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, pearsonr


def load_all_logs(logs_root: str = "results/logs") -> pd.DataFrame:
    rows = []
    for json_file in Path(logs_root).rglob("*.json"):
        if "sanity_check" in json_file.parts:
            continue
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        cfg = data["config"]
        env = data["environment"]
        # Usamos final_epoch para evitar sesgo de selección optimista en validación
        metrics = data["final_epoch"]

        row = {
            "exp_name": cfg["exp_name"],
            "dataset": cfg["dataset"],
            "split_type": cfg["split_type"],
            "backbone": cfg["backbone"].split("/")[-1],
            "feature_family": cfg["feature_family"],
            "train_fraction": cfg["train_fraction"],
            "fold": cfg["fold"],
            "seed": cfg["seed"],
            "runtime_sec": env["runtime_seconds"],
            "peak_vram_mb": env["peak_vram_mb"],
            **metrics,
        }
        rows.append(row)

    return pd.DataFrame(rows)


def summarize_and_test(df: pd.DataFrame):
    if df.empty:
        print("No se encontraron logs de experimentos aún.")
        return

    group_cols = ["exp_name", "dataset", "split_type", "backbone", "feature_family", "train_fraction"]
    metric_cols = [c for c in ["accuracy", "macro_f1", "qwk", "rmse", "pearson_r"] if c in df.columns]

    print("\n" + "=" * 80)
    print("RESUMEN ESTADÍSTICO (MEDIA ± DESVIACIÓN ESTÁNDAR)")
    print("=" * 80)

    summary_rows = []
    for keys, grp in df.groupby(group_cols):
        row_dict = dict(zip(group_cols, keys))
        row_dict["n_runs"] = len(grp)
        for m in metric_cols:
            if grp[m].notna().any():
                row_dict[f"{m}_mean"] = round(grp[m].mean(), 3)
                row_dict[f"{m}_std"] = round(grp[m].std(), 3)
        summary_rows.append(row_dict)

    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))

    out_dir = Path("results/tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_dir / "summary_metrics.csv", index=False)

    # --- PRUEBAS DE SIGNIFICANCIA PAREADA (M1 'none' vs M2 'all') ---
    print("\n" + "=" * 80)
    print("AUDITORÍA DE SIGNIFICANCIA ESTADÍSTICA (PAIRED T-TEST: M1 vs M2)")
    print("=" * 80)

    pair_keys = ["exp_name", "dataset", "split_type", "backbone", "train_fraction"]
    for keys, grp in df.groupby(pair_keys):
        m1 = grp[grp["feature_family"] == "none"].sort_values(["fold", "seed"])
        m2 = grp[grp["feature_family"] == "all"].sort_values(["fold", "seed"])

        if len(m1) > 1 and len(m1) == len(m2):
            primary_m = "rmse" if keys[1] == "clear" else "accuracy"
            t_stat, p_val = ttest_rel(m2[primary_m].values, m1[primary_m].values)
            diff = m2[primary_m].mean() - m1[primary_m].mean()
            sig = "SÍ (p < 0.05)" if p_val < 0.05 else "NO (ruido/empate)"
            print(
                f"{keys} | Δ {primary_m} (M2 - M1) = {diff:+.3f} | "
                f"t = {t_stat:.3f}, p = {p_val:.4f} -> Significativo: {sig}"
            )


def analyze_clear_human_uncertainty(logs_root: str = "results/logs/exp_A_clear"):
    """Audita si los errores de M2 en CLEAR ocurren en textos con alta incertidumbre humana (target_se)."""
    clear_df = pd.read_csv("data/processed/clear_manifest.csv")
    log_files = list(Path(logs_root).glob("*feat-all*_s42.json"))
    if len(log_files) < 5:
        return

    all_errors = []
    all_se = []
    for lf in sorted(log_files):
        with open(lf, "r", encoding="utf-8") as f:
            data = json.load(f)
        fold = data["config"]["fold"]
        val_sub = clear_df[clear_df["fold_stratified"] == fold].reset_index(drop=True)
        y_true = np.array(data["predictions"]["y_true"])
        y_pred = np.array(data["predictions"]["y_pred"])
        abs_err = np.abs(y_true - y_pred)
        all_errors.extend(abs_err)
        all_se.extend(val_sub["target_se"].values)

    r_val, p_val = pearsonr(all_errors, all_se)
    print("\n" + "=" * 80)
    print("AUDITORÍA DE INCERTIDUMBRE HUMANA EN CLEAR (Error Absoluto vs. Human s.e.)")
    print("=" * 80)
    print(f"Correlación Pearson r = {r_val:.3f} (p = {p_val:.4e})")


if __name__ == "__main__":
    df_logs = load_all_logs()
    summarize_and_test(df_logs)
    analyze_clear_human_uncertainty()