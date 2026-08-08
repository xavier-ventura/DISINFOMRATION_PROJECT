"""
actualizar_rag.py - Lee los JSON del scraper e ingesta articulos nuevos en ChromaDB.

Reutiliza todas las funciones de ingest_to_chromadb3.py (chunking, embeddings, dedup).
"""

import json
import sys
from pathlib import Path

# Asegurar que el directorio del proyecto esta en el path
_DIR = Path(__file__).parent
if str(_DIR) not in sys.path:
    sys.path.insert(0, str(_DIR))

import ingest_to_chromadb3 as ingest

MAPA_ETIQUETA = {
    "VERDADERO": "verdad",
    "FALSO":     "bulo",
    "CONTEXTO":  "contexto",
    "ALERTA":    "alerta",
}


def json_a_docs(json_path: Path) -> list[dict]:
    """Convierte un archivo JSON del scraper al formato interno de ingesta."""
    try:
        registros = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"  [ERROR] No se pudo leer {json_path.name}: {e}")
        return []

    docs = []
    for r in registros:
        titular = ingest.limpiar_texto(str(r.get("titulo", "") or "").strip())
        cuerpo  = ingest.limpiar_texto(str(r.get("texto",  "") or "").strip())
        if not titular or not cuerpo:
            continue
        fuente    = str(r.get("medio", "") or "").strip()
        etiqueta  = str(r.get("etiqueta", "") or "").strip().upper()
        fecha_raw = str(r.get("fecha_publicacion", "") or "").strip()
        fecha_iso, fecha_ts = ingest.parse_fecha(fecha_raw)

        docs.append({
            "doc_id":         ingest.doc_id_deterministico(fuente, titular, cuerpo),
            "titular":        titular,
            "url":            str(r.get("url", "") or "").strip(),
            "fuente":         fuente,
            "tipo_fuente":    MAPA_ETIQUETA.get(etiqueta, etiqueta.lower() or "revisar"),
            "fecha_iso":      fecha_iso,
            "fecha_ts":       fecha_ts,
            "hash_contenido": ingest.hash_contenido(cuerpo),
            "idioma":         "es",
            "longitud":       len(cuerpo),
            "cuerpo":         cuerpo,
            "tema":           str(r.get("tema", "") or "").strip(),
            "tecnica":        "",
            "modificacion":   "",
        })
    return docs


def ejecutar_actualizacion(
    json_dir:        str,
    chroma_dir:      str,
    api_key:         str,
    skip_contextual: bool = True,
    dry_run:         bool = False,
):
    """
    Punto de entrada llamado por pipeline_completo.py.
    Lee todos los JSON de json_dir e ingesta los articulos nuevos en ChromaDB.
    """
    json_path = Path(json_dir)
    archivos  = sorted(json_path.glob("noticias_*.json"))

    if not archivos:
        print(f"[RAG] No hay archivos JSON en {json_dir}")
        return

    print(f"[RAG] {len(archivos)} archivo(s) JSON encontrados")

    todos_docs = []
    for f in archivos:
        docs = json_a_docs(f)
        print(f"  {f.name}: {len(docs)} articulos validos")
        todos_docs.extend(docs)

    if not todos_docs:
        print("[RAG] Ningun articulo valido. Abortando.")
        return

    print(f"\nTotal antes de dedup exacto: {len(todos_docs)}")
    todos_docs = ingest.deduplicar_exacto(todos_docs)
    print(f"Total tras dedup exacto:     {len(todos_docs)}")

    chunks = ingest.expandir_a_chunks(todos_docs)

    if dry_run:
        print(f"\n[DRY RUN] {len(chunks)} chunks generados (sin subir a ChromaDB)")
        for c in chunks[:3]:
            print(f"  - {c['titular'][:70]} [{c['tipo_fuente']}]")
        return

    if not api_key:
        print("[ERROR] Necesitas api_key para generar embeddings")
        return

    # Filtrar chunks ya en ChromaDB (reanudacion automatica)
    chunks = ingest.filtrar_chunks_nuevos(chunks, chroma_dir)
    if not chunks:
        print("\n[RAG] Todos los articulos ya estaban en ChromaDB. Nada que hacer.")
        return

    # Contextual chunking (opcional, consume mucha cuota)
    if not skip_contextual:
        from google import genai
        client_gemini = genai.Client(api_key=api_key)
        chunks = ingest.enriquecer_chunks_con_contexto(chunks, client_gemini)
    else:
        print("[Contextual] Saltado (skip_contextual=True)")

    # Embeddings
    batches = (len(chunks) + ingest.EMBED_BATCH_SIZE - 1) // ingest.EMBED_BATCH_SIZE
    tiempo_est = batches * ingest.EMBED_INTERVALO_SEG // 60
    print(f"\nGenerando embeddings para {len(chunks)} chunks ({batches} batches, ~{tiempo_est} min)...")
    embeddings = ingest.generar_embeddings(chunks, api_key)

    # Deduplicacion semantica
    print("\nDeduplicacion semantica...")
    chunks, embeddings = ingest.deduplicar_semantico(chunks, embeddings)

    # Subir a ChromaDB
    print(f"\nSubiendo {len(chunks)} chunks a ChromaDB en '{chroma_dir}'...")
    ingest.ingestar(chunks, embeddings, chroma_dir)
    print("\n[RAG] Actualizacion completada.")


if __name__ == "__main__":
    import argparse, os
    parser = argparse.ArgumentParser(description="Ingesta de JSONs del scraper en ChromaDB")
    parser.add_argument("--json-dir",       default="./noticias_rag")
    parser.add_argument("--chroma-dir",     default="./chroma_db_v3")
    parser.add_argument("--api-key",        default=os.getenv("GOOGLE_API_KEY"))
    parser.add_argument("--skip-contextual", action="store_true", default=True)
    parser.add_argument("--dry-run",        action="store_true")
    args = parser.parse_args()

    ejecutar_actualizacion(
        json_dir        = args.json_dir,
        chroma_dir      = args.chroma_dir,
        api_key         = args.api_key or "",
        skip_contextual = args.skip_contextual,
        dry_run         = args.dry_run,
    )
