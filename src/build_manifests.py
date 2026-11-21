import re
from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GroupKFold

# ==========================================
# 1. CONFIGURACIÓN DE RUTAS
# ==========================================
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Ajusta estos nombres según cómo estén exactamente en tu carpeta data/raw/
ONESTOP_DIRS = {
    0: RAW_DIR / "Ele-Txt",
    1: RAW_DIR / "Int-Txt" / "Int-Txt",
    2: RAW_DIR / "Adv-Txt",
}
CLEAR_CSV_PATH = RAW_DIR / "clear_corpus.csv"
SEED = 42


# ==========================================
# 2. FUNCIONES DE LIMPIEZA Y EMPAREJAMIENTO
# ==========================================
def read_clean_file(file_path: Path) -> str:
    """Lee el archivo manejando BOM de UTF-8/UTF-16 y codificaciones Windows-1252/Latin-1."""
    raw_bytes = file_path.read_bytes()

    # 1. Detectar y decodificar BOM explícito (UTF-8-SIG, UTF-16 LE/BE)
    if raw_bytes.startswith(b"\xef\xbb\xbf"):
        text = raw_bytes.decode("utf-8-sig")
    elif raw_bytes.startswith((b"\xff\xfe", b"\xfe\xff")):
        text = raw_bytes.decode("utf-16")
    else:
        # 2. Intentar UTF-8 estricto; si falla, usar Windows-1252 (superset de Latin-1)
        content = read_clean_file(file_path)
        clean_text = clean_onestop_text(content)

    return text


def clean_onestop_text(raw_text: str) -> str:
    """Elimina caracteres de control, BOM residuales, símbolos  y encabezados filtrados."""
    # Eliminar BOM (\ufeff), caracteres de reemplazo (\ufffd) y espacios no separables (\xa0)
    text = raw_text.replace("\ufeff", "").replace("\ufffd", "").replace("\xa0", " ")

    lines = text.strip().splitlines()
    cleaned_lines = []
    header_pattern = re.compile(r"^\s*(elementary|intermediate|advanced)\s*:?\s*$", re.IGNORECASE)

    for idx, line in enumerate(lines):
        # Limpiar cualquier basura no alfanumérica suelta al inicio de la primera línea
        line_clean = line.strip()
        if idx < 3 and header_pattern.match(line_clean):
            continue
        cleaned_lines.append(line_clean)

    joined = " ".join([l for l in cleaned_lines if l])
    return re.sub(r"\s+", " ", joined).strip()


def extract_article_slug(filename: str) -> str:
    """Extrae el identificador base del artículo quitando el sufijo de nivel (-ele, -int, -adv)."""
    stem = Path(filename).stem
    # Elimina sufijos comunes al final del nombre del archivo
    slug = re.sub(r"-(ele|int|adv|elementary|intermediate|advanced)$", "", stem, flags=re.IGNORECASE)
    return slug.lower().strip()


# ==========================================
# 3. CONSTRUCCIÓN DE ONESTOP_MANIFEST.CSV
# ==========================================
def build_onestop_manifest():
    records = []

    for level, folder in ONESTOP_DIRS.items():
        if not folder.exists():
            raise FileNotFoundError(f"No se encontró la carpeta: {folder}. Verifica la ruta.")

        # Ignora .DS_Store y cualquier archivo oculto o que no sea .txt
        txt_files = sorted([
            f for f in folder.iterdir()
            if f.is_file() and f.suffix.lower() == ".txt" and not f.name.startswith(".")
        ])

        for file_path in txt_files:
            # Algunos archivos de OneStopEnglish vienen en UTF-8 y otros en Latin-1
            content = read_clean_file(file_path)
            clean_text = clean_onestop_text(content)
            article_slug = extract_article_slug(file_path.name)

            records.append({
                "article_slug": article_slug,
                "level": level,
                "level_name": {0: "Elementary", 1: "Intermediate", 2: "Advanced"}[level],
                "filename": file_path.name,
                "text": clean_text,
            })

    df = pd.DataFrame(records)

    # --- AUDITORÍA ESTRICTA (ASSERTS) ---
    assert len(df) == 567, f"Error: Se esperaban 567 textos, pero se encontraron {len(df)}."
    assert df["text"].str.len().gt(50).all(), "Error: Hay textos vacíos o demasiado cortos."
    assert not df["text"].str.contains("\ufffd", regex=False).any(), "Error: Aún quedan caracteres corruptos () en el texto."


    # Verificar que ningún texto empiece con palabras clave filtradas
    leak_check = df["text"].str.match(r"^(elementary|intermediate|advanced)\b", case=False)
    assert not leak_check.any(), f"Fuga detectada en {leak_check.sum()} archivos."

    # Verificar que cada artículo tenga exactamente 3 versiones (1 por nivel)
    counts_per_slug = df.groupby("article_slug")["level"].nunique()
    bad_slugs = counts_per_slug[counts_per_slug != 3]
    assert len(bad_slugs) == 0, f"Artículos sin triplete completo: {bad_slugs.to_dict()}"

    # Asignar ID numérico limpio (0 a 188) y orden determinista
    df = df.sort_values(["article_slug", "level"]).reset_index(drop=True)
    slug_to_id = {slug: idx for idx, slug in enumerate(df["article_slug"].unique())}
    df["article_id"] = df["article_slug"].map(slug_to_id)
    df["doc_id"] = [f"ose_{i:03d}" for i in range(len(df))]

    # --- GENERACIÓN DE FOLDS DUALES (k=5) ---
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    df["fold_stratified"] = -1
    for fold, (_, val_idx) in enumerate(skf.split(df, df["level"])):
        df.loc[val_idx, "fold_stratified"] = fold

    gkf = GroupKFold(n_splits=5)
    df["fold_grouped"] = -1
    for fold, (_, val_idx) in enumerate(gkf.split(df, df["level"], groups=df["article_id"])):
        df.loc[val_idx, "fold_grouped"] = fold

    assert (df["fold_stratified"] != -1).all() and (df["fold_grouped"] != -1).all()

    out_path = PROCESSED_DIR / "onestop_manifest.csv"
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"[OK] onestop_manifest.csv generado ({len(df)} filas) -> {out_path}")


# ==========================================
# 4. CONSTRUCCIÓN DE CLEAR_MANIFEST.CSV
# ==========================================
def build_clear_manifest():
    if not CLEAR_CSV_PATH.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {CLEAR_CSV_PATH}")

    # Lectura con separador punto y coma (;)
    df_raw = pd.read_csv(CLEAR_CSV_PATH, sep=";")

    # Validar columnas requeridas
    required_cols = {"ID", "Excerpt", "BT_easiness", "s.e."}
    missing = required_cols - set(df_raw.columns)
    assert not missing, f"Faltan columnas en clear_corpus.csv: {missing}"

    df = df_raw[["ID", "Excerpt", "BT_easiness", "s.e."]].copy()
    df = df.rename(columns={
        "ID": "doc_id",
        "Excerpt": "text",
        "BT_easiness": "target",
        "s.e.": "target_se",
    })

    # Asegurar tipos numéricos (por si algún decimal viene con coma en formato europeo)
    for col in ["target", "target_se"]:
        if df[col].dtype == object:
            df[col] = df[col].str.replace(",", ".", regex=False).astype(float)

    # Limpieza de texto
    df["text"] = df["text"].astype(str).apply(lambda t: re.sub(r"\s+", " ", t).strip())

    # --- AUDITORÍA ESTRICTA ---
    assert df["text"].notna().all() and df["text"].str.len().gt(20).all(), "Textos inválidos en CLEAR."
    assert df["target"].notna().all() and df["target_se"].notna().all(), "Valores nulos en targets de CLEAR."
    assert df["doc_id"].is_unique, "Hay IDs duplicados en CLEAR."

    # Estratificación por 5 cuantiles de dificultad para regresión balanceada
    df = df.sort_values("doc_id").reset_index(drop=True)
    difficulty_bins = pd.qcut(df["target"], q=5, labels=False)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    df["fold_stratified"] = -1
    for fold, (_, val_idx) in enumerate(skf.split(df, difficulty_bins)):
        df.loc[val_idx, "fold_stratified"] = fold

    assert (df["fold_stratified"] != -1).all()

    out_path = PROCESSED_DIR / "clear_manifest.csv"
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"[OK] clear_manifest.csv generado ({len(df)} filas) -> {out_path}")


if __name__ == "__main__":
    build_onestop_manifest()
    build_clear_manifest()