"""
evaluar_chatbot.py - Evaluacion del clasificador LNR sobre un conjunto de prueba.

Mide precision, recall, F1 y accuracy del modelo prediciendo
VERDADERO / FALSO / CONTEXTO usando el mismo pipeline que el chatbot
(busqueda semantica en ChromaDB + ensemble LNR).

Uso:
    python evaluar_chatbot.py --api-key TU_KEY
    python evaluar_chatbot.py --api-key TU_KEY --n-muestras 60
    python evaluar_chatbot.py --api-key TU_KEY --solo-curado
    python evaluar_chatbot.py --api-key TU_KEY --verbose
"""

import argparse
import io
import os
import sys
import time
import json
import random
from collections import defaultdict
from pathlib import Path

import chromadb
from chromadb.config import Settings
from google import genai
from google.genai import types as genai_types

_DIR = Path(__file__).parent
if str(_DIR) not in sys.path:
    sys.path.insert(0, str(_DIR))

from clasificador_lnr import cargar_clasificador

# ── Config ────────────────────────────────────────────────────────────────────
COLLECTION_NAME = "bulos_inmigracion_v3"
EMBEDDING_MODEL = "gemini-embedding-2-preview"
CHROMA_DIR      = "./chroma_db_v3"
MODELO_LNR      = "./modelo_lnr/modelo_final"
DATASET_EXCEL = "datos/dataset_inmigracion_unificado_limpio_tema.xlsx"

MAPA_ETIQUETA = {
    "VERDADERO": "VERDADERO",
    "FALSO":     "FALSO",
    "CONTEXTO":  "CONTEXTO",
    "ALERTA":    "FALSO",   # ALERTA ~ FALSO a efectos de evaluacion
}

# ── Conjunto curado de prueba (ground truth conocido) ─────────────────────────
# Afirmaciones con etiqueta esperada verificada manualmente.
# Formato: (afirmacion, etiqueta_esperada)
TEST_CURADO = [
    # --- FALSO (bulos conocidos) ---
    ("Los menores migrantes no acompañados reciben 4.700 euros al mes del Estado",             "FALSO"),
    ("Los inmigrantes ilegales cobran más ayudas sociales que los jubilados españoles",        "FALSO"),
    ("España es el país de Europa que más inmigrantes irregulares recibe",                     "FALSO"),
    ("Los menas cometen el 80% de los delitos en las ciudades donde viven",                    "FALSO"),
    ("Los africanos en situación ilegal cobran 2.000 euros de ayudas del Estado",              "FALSO"),
    ("Los inmigrantes no pagan a la Seguridad Social pero tienen derecho a pensión",           "FALSO"),
    ("Hay más de 100.000 inmigrantes que cobran pensión sin haber cotizado nunca",             "FALSO"),
    ("Los inmigrantes ilegales tienen prioridad en las listas de espera de la sanidad",        "FALSO"),

    # --- CONTEXTO (engañoso o parcialmente verdadero) ---
    ("Los inmigrantes no pagan impuestos pero sí reciben sanidad y educación gratis",          "CONTEXTO"),
    ("Las llegadas de migrantes a Canarias se han disparado este año",                         "CONTEXTO"),
    ("España expulsa a muy pocos inmigrantes de los que entran irregularmente",                "CONTEXTO"),
    ("Los centros de menores migrantes cuestan más que un hotel de cinco estrellas",           "CONTEXTO"),
    ("La inmigración dispara el gasto en servicios sociales en España",                        "CONTEXTO"),

    # --- VERDADERO (noticias contrastadas) ---
    ("El Gobierno ha vetado la enmienda del PP que endurecía los requisitos para regularizar migrantes", "VERDADERO"),
    ("Miles de menores migrantes han sido trasladados desde Canarias a otras comunidades autónomas",     "VERDADERO"),
    ("La reforma de la Ley de Extranjería obliga a las comunidades a acoger menores no acompañados",    "VERDADERO"),
    ("Los inmigrantes contribuyen positivamente a las arcas de la Seguridad Social en España",          "VERDADERO"),
    ("Frontex detectó un aumento de llegadas por la ruta del Mediterráneo occidental",                  "VERDADERO"),
]


# ── Busqueda semantica (igual que el chatbot) ─────────────────────────────────

def buscar_docs(query: str, collection, client_gemini, top_k: int = 6) -> list[dict]:
    result = client_gemini.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=[query],
        config=genai_types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )
    embedding = result.embeddings[0].values
    res = collection.query(
        query_embeddings=[embedding],
        n_results=top_k,
        include=["metadatas", "documents", "distances"],
    )
    docs = []
    for meta, doc, dist in zip(res["metadatas"][0], res["documents"][0], res["distances"][0]):
        docs.append({
            "titular":     meta.get("titular", ""),
            "texto":       doc,
            "fuente":      meta.get("fuente", ""),
            "tipo_fuente": meta.get("tipo_fuente", ""),
            "fecha_iso":   meta.get("fecha_iso", ""),
            "distancia":   round(dist, 4),
        })
    return docs


# ── Metricas ──────────────────────────────────────────────────────────────────

def calcular_metricas(resultados: list[dict]) -> dict:
    clases = ["VERDADERO", "FALSO", "CONTEXTO"]
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    for r in resultados:
        real = r["real"]
        pred = r["pred"]
        for c in clases:
            if real == c and pred == c:
                tp[c] += 1
            elif pred == c and real != c:
                fp[c] += 1
            elif real == c and pred != c:
                fn[c] += 1

    metricas = {}
    for c in clases:
        prec = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) > 0 else 0.0
        rec  = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        metricas[c] = {"precision": prec, "recall": rec, "f1": f1,
                       "tp": tp[c], "fp": fp[c], "fn": fn[c]}

    total    = len(resultados)
    correctos = sum(1 for r in resultados if r["real"] == r["pred"])
    inciertos = sum(1 for r in resultados if r["pred"] == "INCIERTO")

    metricas["__global__"] = {
        "accuracy":       correctos / total if total else 0.0,
        "accuracy_str":   f"{correctos}/{total}",
        "inciertos":      inciertos,
        "total":          total,
    }
    return metricas


def matriz_confusion(resultados: list[dict]) -> None:
    clases = ["VERDADERO", "FALSO", "CONTEXTO", "INCIERTO"]
    clases_presentes = sorted({r["pred"] for r in resultados} | {r["real"] for r in resultados})
    clases_presentes = [c for c in clases if c in clases_presentes]

    ancho = 12
    print("\n  MATRIZ DE CONFUSION (filas=real, columnas=prediccion)")
    print("  " + " " * 12 + "".join(f"{c:>{ancho}}" for c in clases_presentes))
    print("  " + "-" * (12 + ancho * len(clases_presentes)))

    for real in clases_presentes:
        fila = f"  {real:<12}"
        for pred in clases_presentes:
            count = sum(1 for r in resultados if r["real"] == real and r["pred"] == pred)
            marca = f"[{count}]" if (real == pred and count > 0) else str(count)
            fila += f"{marca:>{ancho}}"
        print(fila)


def imprimir_metricas(metricas: dict) -> None:
    g = metricas["__global__"]
    print(f"\n  ACCURACY GLOBAL: {g['accuracy']:.1%}  ({g['accuracy_str']})")
    print(f"  Predicciones INCIERTO: {g['inciertos']} de {g['total']}")
    print()
    print(f"  {'Clase':<12} {'Precision':>10} {'Recall':>10} {'F1':>10}  {'TP':>4} {'FP':>4} {'FN':>4}")
    print("  " + "-" * 60)
    for c in ["VERDADERO", "FALSO", "CONTEXTO"]:
        m = metricas[c]
        print(f"  {c:<12} {m['precision']:>10.1%} {m['recall']:>10.1%} {m['f1']:>10.1%}"
              f"  {m['tp']:>4} {m['fp']:>4} {m['fn']:>4}")

    # Macro F1
    f1s = [metricas[c]["f1"] for c in ["VERDADERO", "FALSO", "CONTEXTO"]]
    print(f"\n  Macro F1: {sum(f1s)/len(f1s):.1%}")


# ── Cargar muestra del dataset ────────────────────────────────────────────────

def cargar_muestra_dataset(n_por_clase: int, seed: int = 42) -> list[tuple[str, str]]:
    """
    Lee las muestras desde muestras_eval.json (generado previamente).
    Esto evita cargar pandas + Excel con el modelo ya en memoria.
    """
    json_path = Path("muestras_eval.json")
    if not json_path.exists():
        print("[AVISO] muestras_eval.json no encontrado.", flush=True)
        print("  Generalo con: python generar_muestras_eval.py", flush=True)
        return []

    try:
        datos = json.loads(json_path.read_text(encoding="utf-8"))
        muestras = [(titulo, etiqueta) for titulo, etiqueta in datos
                    if etiqueta in MAPA_ETIQUETA.values() and len(titulo) > 10]
        print(f"[Dataset] {len(muestras)} muestras cargadas desde muestras_eval.json", flush=True)
        return muestras
    except Exception as e:
        print(f"[ERROR] leyendo muestras_eval.json: {e}", flush=True)
        return []


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Evaluacion del clasificador LNR del chatbot")
    parser.add_argument("--api-key",      default=os.getenv("GOOGLE_API_KEY"))
    parser.add_argument("--chroma-dir",   default=CHROMA_DIR)
    parser.add_argument("--modelo-local", default=MODELO_LNR)
    parser.add_argument("--n-muestras",   type=int, default=20,
                        help="Ejemplos por clase extraidos del dataset Excel (default: 20)")
    parser.add_argument("--solo-curado",  action="store_true",
                        help="Usar solo el conjunto curado (sin muestrear el dataset)")
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--verbose",      action="store_true",
                        help="Mostrar resultado de cada consulta")
    parser.add_argument("--pausa",        type=float, default=1.5,
                        help="Segundos entre consultas (default 1.5, evita rate limit)")
    args = parser.parse_args()

    if not args.api_key:
        sys.exit("ERROR: necesitas --api-key o exportar GOOGLE_API_KEY")

    # ── Inicializar servicios ─────────────────────────────────────────────────
    print("Cargando ChromaDB...")
    client_chroma = chromadb.PersistentClient(
        path=args.chroma_dir, settings=Settings(anonymized_telemetry=False)
    )
    collection = client_chroma.get_collection(COLLECTION_NAME)
    print(f"  {collection.count()} chunks disponibles")

    client_gemini = genai.Client(api_key=args.api_key)

    print("Cargando clasificador LNR...")
    clasificador = cargar_clasificador(args.modelo_local)
    if not clasificador:
        sys.exit("ERROR: no se pudo cargar el modelo LNR")
    print(f"  {clasificador}")

    # ── Construir conjunto de prueba ──────────────────────────────────────────
    test_set = list(TEST_CURADO)

    if not args.solo_curado:
        n_por_clase = args.n_muestras
        muestras    = cargar_muestra_dataset(n_por_clase, seed=args.seed)
        # Evitar duplicados con el conjunto curado
        curados_lower = {t.lower() for t, _ in TEST_CURADO}
        muestras = [(t, e) for t, e in muestras if t.lower() not in curados_lower]
        test_set.extend(muestras)

    random.seed(args.seed)
    random.shuffle(test_set)

    total = len(test_set)
    dist  = defaultdict(int)
    for _, e in test_set:
        dist[e] += 1
    print(f"\nConjunto de prueba: {total} ejemplos")
    print(f"  VERDADERO: {dist['VERDADERO']}  FALSO: {dist['FALSO']}  CONTEXTO: {dist['CONTEXTO']}")

    # ── Evaluar ───────────────────────────────────────────────────────────────
    print(f"\nEvaluando {total} consultas (pausa {args.pausa}s entre llamadas)...\n")

    resultados = []
    EMOJIS = {"VERDADERO": "V", "FALSO": "F", "CONTEXTO": "C", "INCIERTO": "?"}

    for i, (query, real) in enumerate(test_set, 1):
        try:
            docs = buscar_docs(query, collection, client_gemini)
            res  = clasificador.clasificar_con_docs(query, docs)
            pred = res["etiqueta"]
            conf = res["confianza"]
        except Exception as e:
            print(f"  [{i:>3}/{total}] ERROR: {e}")
            pred = "ERROR"
            conf = 0.0
            docs = []

        correcto = pred == real
        marca    = "OK" if correcto else "XX"

        if args.verbose:
            top_fuente = docs[0]["fuente"] if docs else "-"
            top_tipo   = docs[0]["tipo_fuente"] if docs else "-"
            top_sim    = round((1 - docs[0]["distancia"]) * 100, 1) if docs else 0
            print(f"  [{i:>3}/{total}] {marca}  real={real:<10} pred={pred:<10} conf={conf:.0%}"
                  f"  | doc1: {top_fuente} [{top_tipo}] {top_sim}%")
            if not correcto:
                print(f"          Query: {query[:80]}")
        else:
            simbolo = "." if correcto else "X"
            print(simbolo, end="", flush=True)
            if i % 50 == 0:
                print(f" {i}/{total}")

        resultados.append({"query": query, "real": real, "pred": pred,
                           "confianza": conf, "correcto": correcto})

        if i < total:
            time.sleep(args.pausa)

    if not args.verbose:
        print()

    # ── Resultados ────────────────────────────────────────────────────────────
    sep = "=" * 60
    print(f"\n{sep}")
    print("  RESULTADOS DE EVALUACION")
    print(sep)

    metricas = calcular_metricas(resultados)
    imprimir_metricas(metricas)
    matriz_confusion(resultados)

    # ── Errores mas destacados ────────────────────────────────────────────────
    errores = [r for r in resultados if not r["correcto"] and r["pred"] != "INCIERTO"]
    if errores:
        print(f"\n  PREDICCIONES INCORRECTAS ({len(errores)}):")
        print("  " + "-" * 58)
        for r in errores[:10]:
            print(f"  real={r['real']:<10} pred={r['pred']:<10} conf={r['confianza']:.0%}")
            print(f"    {r['query'][:75]}")

    inciertos = [r for r in resultados if r["pred"] == "INCIERTO"]
    if inciertos:
        print(f"\n  INCIERTOS ({len(inciertos)}):")
        print("  " + "-" * 58)
        for r in inciertos[:5]:
            print(f"  real={r['real']:<10}  {r['query'][:75]}")

    print(f"\n{sep}\n")


if __name__ == "__main__":
    main()
