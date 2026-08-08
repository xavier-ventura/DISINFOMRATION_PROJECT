"""
generar_muestras_eval.py — Genera muestras_eval.json desde el Excel.

Ejecutar ANTES de evaluar_chatbot.py cuando quieras cambiar el numero de muestras.

Uso:
    python generar_muestras_eval.py
    python generar_muestras_eval.py --n-muestras 30
"""

import argparse
import json
import random
from pathlib import Path

import pandas as pd

_DIR       = Path(__file__).parent
EXCEL_PATH = _DIR / "datos" / "dataset_inmigracion_unificado_limpio_tema.xlsx"
OUT_PATH   = _DIR / "muestras_eval.json"

MAPA_ETIQUETA = {
    "VERDADERO": "VERDADERO",
    "FALSO":     "FALSO",
    "CONTEXTO":  "CONTEXTO",
    "ALERTA":    "FALSO",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-muestras", type=int, default=20,
                        help="Ejemplos por clase (default: 20)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Leyendo {EXCEL_PATH.name}...")
    df = pd.read_excel(EXCEL_PATH)
    df["etiqueta_norm"] = df["etiqueta"].map(MAPA_ETIQUETA)
    df = df[df["etiqueta_norm"].notna()]
    print(f"  {len(df)} filas validas")

    muestras = []
    for etiqueta in ["VERDADERO", "FALSO", "CONTEXTO"]:
        subset = df[df["etiqueta_norm"] == etiqueta]
        n      = min(args.n_muestras, len(subset))
        sample = subset.sample(n=n, random_state=args.seed)
        for _, row in sample.iterrows():
            titulo = str(row["titulo"]).strip()
            if titulo and len(titulo) > 10:
                muestras.append([titulo, etiqueta])

    random.seed(args.seed)
    random.shuffle(muestras)

    OUT_PATH.write_text(
        json.dumps(muestras, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    from collections import Counter
    dist = Counter(e for _, e in muestras)
    print(f"\nGuardadas {len(muestras)} muestras en {OUT_PATH.name}")
    print(f"  {dict(dist)}")
    print(f"\nAhora puedes ejecutar:")
    print(f"  python evaluar_chatbot.py --api-key TU_CLAVE --modelo-local modelo_lnr/modelo_final --verbose")


if __name__ == "__main__":
    main()
