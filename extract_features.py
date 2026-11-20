import re
from pathlib import Path
import numpy as np
import pandas as pd
import spacy

PROCESSED_DIR = Path("data/processed")

# Cargamos spaCy deshabilitando NER (no lo necesitamos y acelera x3 la extracción)
nlp = spacy.load("en_core_web_sm", disable=["ner"])


def count_syllables(word: str) -> int:
    """Heurística determinista estándar para conteo de sílabas en inglés."""
    word = re.sub(r"[^a-z]", "", word.lower())
    if not word:
        return 0
    if len(word) <= 3:
        return 1
    word = re.sub(r"(?:[^laeiouy]es|ed|[^laeiouy]e)$", "", word)
    word = re.sub(r"^y", "", word)
    vowels = re.findall(r"[aeiouy]{1,2}", word)
    return max(1, len(vowels))


def get_tree_depth(token) -> int:
    """Calcula la profundidad máxima del árbol de dependencias desde un token raíz."""
    children = list(token.children)
    if not children:
        return 1
    return 1 + max(get_tree_depth(child) for child in children)


def extract_single_text_features(doc) -> list[float]:
    """Extrae las 22 características lingüísticas organizadas en 3 familias."""
    sents = list(doc.sents)
    words = [t for t in doc if t.is_alpha]

    num_chars = sum(len(t.text) for t in words)
    num_words = max(1, len(words))
    num_sents = max(1, len(sents))

    syllables = [count_syllables(t.text) for t in words]
    total_syllables = sum(syllables)
    long_words = sum(1 for t in words if len(t.text) > 6)

    # --- FAMILIA 1: SUPERFICIALES (8 features) ---
    avg_word_len = num_chars / num_words
    avg_sent_len = num_words / num_sents
    syllables_per_word = total_syllables / num_words
    long_word_ratio = long_words / num_words
    fkgl = 0.39 * avg_sent_len + 11.8 * syllables_per_word - 15.59

    surface_feats = [
        float(num_chars),
        float(num_words),
        float(num_sents),
        avg_word_len,
        avg_sent_len,
        syllables_per_word,
        long_word_ratio,
        fkgl,
    ]

    # --- FAMILIA 2: DIVERSIDAD LÉXICA (4 features) ---
    lemmas = [t.lemma_.lower() for t in words]
    vocab_size = len(set(lemmas))
    freq_map = pd.Series(lemmas).value_counts()
    hapax_count = int((freq_map == 1).sum())

    ttr = vocab_size / num_words
    root_ttr = vocab_size / np.sqrt(num_words)
    corrected_ttr = vocab_size / np.sqrt(2.0 * num_words)
    hapax_ratio = hapax_count / num_words

    lexical_feats = [ttr, root_ttr, corrected_ttr, hapax_ratio]

    # --- FAMILIA 3: SINTÁCTICAS / POS (10 features) ---
    pos_counts = {pos: 0 for pos in ["NOUN", "VERB", "ADJ", "ADV", "PRON", "ADP"]}
    conj_count = 0
    sub_clauses = 0

    for t in words:
        if t.pos_ in pos_counts:
            pos_counts[t.pos_] += 1
        elif t.pos_ in ("CCONJ", "SCONJ"):
            conj_count += 1
        if t.dep_ in ("advcl", "relcl", "ccomp", "xcomp"):
            sub_clauses += 1

    sent_depths = [get_tree_depth(s.root) for s in sents] if sents else [1]
    max_tree_depth = float(max(sent_depths))
    mean_tree_depth = float(np.mean(sent_depths))
    sub_clauses_per_sent = sub_clauses / num_sents

    syntactic_feats = [
        pos_counts["NOUN"] / num_words,
        pos_counts["VERB"] / num_words,
        pos_counts["ADJ"] / num_words,
        pos_counts["ADV"] / num_words,
        pos_counts["PRON"] / num_words,
        pos_counts["ADP"] / num_words,
        conj_count / num_words,
        max_tree_depth,
        mean_tree_depth,
        sub_clauses_per_sent,
    ]

    return surface_feats + lexical_feats + syntactic_feats


def process_dataset(manifest_name: str, output_npy_name: str, target_col: str):
    manifest_path = PROCESSED_DIR / manifest_name
    df = pd.read_csv(manifest_path)

    print(f"Extrayendo features para {manifest_name} ({len(df)} textos)...")
    feature_rows = []
    for doc in nlp.pipe(df["text"].tolist(), batch_size=64):
        feature_rows.append(extract_single_text_features(doc))

    X_feats = np.array(feature_rows, dtype=np.float32)

    # --- AUDITORÍA ESTRICTA ---
    assert X_feats.shape == (len(df), 22), f"Dimensión inesperada: {X_feats.shape}"
    assert np.isfinite(X_feats).all(), "Error: Se detectaron valores NaN o Inf en las features."

    # Prueba de sanidad rápida: correlación de longitud media de oración (col 4) y FKGL (col 7) con target
    corr_sent = np.corrcoef(X_feats[:, 4], df[target_col])[0, 1]
    corr_fkgl = np.corrcoef(X_feats[:, 7], df[target_col])[0, 1]
    print(f"  -> Correlación con '{target_col}': AvgSentLen = {corr_sent:.3f} | FKGL = {corr_fkgl:.3f}")

    out_path = PROCESSED_DIR / output_npy_name
    np.save(out_path, X_feats)
    print(f"[OK] Guardado {out_path} con shape {X_feats.shape}\n")

    
if __name__ == "__main__":
    process_dataset("onestop_manifest.csv", "onestop_features.npy", target_col="level")
    process_dataset("clear_manifest.csv", "clear_features.npy", target_col="target")