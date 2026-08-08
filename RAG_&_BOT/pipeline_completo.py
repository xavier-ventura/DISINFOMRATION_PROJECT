"""
pipeline_completo.py - Extraccion + Aumentacion + Ingesta RAG + Entrenamiento.

Pasos:
  1. Scraper        -> descarga noticias nuevas a noticias_rag/
  2. Aumentacion    -> genera versiones FALSO/CONTEXTO con Gemini → noticias_aumentadas/
  3. actualizar_rag -> ingesta los articulos nuevos en ChromaDB
  4. Entrenamiento  -> fine-tuning incremental del modelo LNR

Uso rapido:
    python pipeline_completo.py --api-key TU_GOOGLE_API_KEY

Solo scraper + RAG (sin aumentacion ni entrenamiento):
    python pipeline_completo.py --api-key TU_CLAVE --sin-aumentacion --sin-entrenamiento

Solo aumentacion + entrenamiento (sin scraper):
    python pipeline_completo.py --api-key TU_CLAVE --solo-rag --sin-rag

Variables de entorno:
    GOOGLE_API_KEY
"""

import argparse
import io
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# -- Rutas por defecto --------------------------------------------------------
_DIR = Path(__file__).parent

DEFAULT_JSON_DIR      = _DIR / "noticias_rag"
DEFAULT_AUMENTADOS_DIR= _DIR / "noticias_aumentadas"
DEFAULT_CHROMA_DIR    = _DIR / "chroma_db_v3"
DEFAULT_MODELO_BASE   = _DIR / "modelo_lnr" / "modelo_final"


def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def separador(titulo: str = ""):
    linea = "=" * 60
    if titulo:
        print(f"\n{linea}")
        print(f"  {titulo}")
        print(f"{linea}")
    else:
        print(f"\n{linea}\n")


# -- Paso 1: Scraper ----------------------------------------------------------

def ejecutar_scraper(args) -> int:
    separador("PASO 1 - Scraper de noticias")

    if str(_DIR) not in sys.path:
        sys.path.insert(0, str(_DIR))

    try:
        import scraper_inmigracion as scraper
    except ModuleNotFoundError as e:
        log(f"ERROR: falta una dependencia -> {e}")
        sys.exit(1)

    old_argv = sys.argv
    sys.argv = ["scraper_inmigracion.py", "--output-dir", str(args.json_dir)]
    if args.sin_dedup:    sys.argv.append("--sin-dedup")
    if args.sin_elpais:   sys.argv.append("--sin-elpais")
    if args.sin_eldiario: sys.argv.append("--sin-eldiario")
    if args.sin_maldita:  sys.argv.append("--sin-maldita")

    try:
        n_nuevos = scraper.main()
    finally:
        sys.argv = old_argv

    return n_nuevos or 0


# -- Paso 2: Aumentación de datos ---------------------------------------------

def ejecutar_aumentacion(args, n_nuevos_scraper: int) -> int:
    """
    Genera versiones FALSO/CONTEXTO de los artículos nuevos.
    Devuelve el número de artículos generados.
    """
    separador("PASO 2 - Aumentación de datos (Gemini)")

    if str(_DIR) not in sys.path:
        sys.path.insert(0, str(_DIR))

    try:
        import aumentar_datos as aug
    except ImportError:
        log("ERROR: no se encontró aumentar_datos.py en la misma carpeta.")
        sys.exit(1)

    todos = aug.ejecutar_aumentacion(
        json_dir        = str(args.json_dir),
        out_dir         = str(args.aumentados_dir),
        api_key         = args.api_key or "",
        max_articulos   = args.max_aumentar,
        solo_verdaderos = True,
        dry_run         = args.dry_run,
    )

    generados = [r for r in todos if r.get("es_generado")]
    log(f"Versiones generadas: {len(generados)}")
    return len(generados)


# -- Paso 3: Actualizar RAG ---------------------------------------------------

def ejecutar_rag(args):
    separador("PASO 3 - Ingesta en ChromaDB (RAG)")

    if str(_DIR) not in sys.path:
        sys.path.insert(0, str(_DIR))

    try:
        import actualizar_rag as rag
    except ImportError:
        log("ERROR: no se encontró actualizar_rag.py en la misma carpeta.")
        sys.exit(1)

    # Solo ingestar artículos reales (noticias_rag/)
    # Las noticias generadas por IA NO van al RAG
    rag.ejecutar_actualizacion(
        json_dir        = str(args.json_dir),
        chroma_dir      = str(args.chroma_dir),
        api_key         = args.api_key or "",
        skip_contextual = args.skip_contextual,
        dry_run         = args.dry_run,
    )


# -- Paso 4: Entrenamiento incremental ----------------------------------------

def ejecutar_entrenamiento(args):
    separador("PASO 4 - Entrenamiento incremental del modelo LNR")

    if str(_DIR) not in sys.path:
        sys.path.insert(0, str(_DIR))

    try:
        import entrenar_incremental as trainer
    except ImportError:
        log("ERROR: no se encontró entrenar_incremental.py en la misma carpeta.")
        sys.exit(1)

    trainer.ejecutar_entrenamiento(
        modelo_base    = str(args.modelo_base),
        modelo_out     = str(args.modelo_base),
        json_nuevos    = str(args.json_dir),
        json_generados = str(args.aumentados_dir),
        acumulado_path = str(_DIR / "dataset_acumulado.json"),
        n_epocas       = args.epocas,
        dry_run        = args.dry_run,
    )


# -- Main ---------------------------------------------------------------------

def main():
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Pipeline completo FakeNewsBot")

    # General
    parser.add_argument("--api-key",         default=os.getenv("GOOGLE_API_KEY"))
    parser.add_argument("--dry-run",         action="store_true",
                        help="Simular sin llamar a APIs ni modificar el modelo")

    # Rutas
    parser.add_argument("--json-dir",        default=str(DEFAULT_JSON_DIR),    type=Path)
    parser.add_argument("--aumentados-dir",  default=str(DEFAULT_AUMENTADOS_DIR), type=Path)
    parser.add_argument("--chroma-dir",      default=str(DEFAULT_CHROMA_DIR),  type=Path)
    parser.add_argument("--modelo-base",     default=str(DEFAULT_MODELO_BASE), type=Path,
                        help="Ruta al modelo LNR (se actualiza in-place)")

    # RAG
    parser.add_argument("--skip-contextual", action="store_true", default=True)

    # Scraper
    parser.add_argument("--sin-dedup",       action="store_true")
    parser.add_argument("--sin-elpais",      action="store_true")
    parser.add_argument("--sin-eldiario",    action="store_true")
    parser.add_argument("--sin-maldita",     action="store_true")

    # Aumentación
    parser.add_argument("--max-aumentar",    type=int, default=0,
                        help="Máx. artículos a aumentar por ejecución (0 = sin límite)")

    # Entrenamiento
    parser.add_argument("--epocas",          type=int, default=4)

    # Flags de control de pasos
    parser.add_argument("--sin-aumentacion", action="store_true",
                        help="Saltar el paso de aumentación de datos")
    parser.add_argument("--sin-rag",         action="store_true",
                        help="Saltar la ingesta en ChromaDB")
    parser.add_argument("--sin-entrenamiento", action="store_true",
                        help="Saltar el fine-tuning del modelo LNR")
    parser.add_argument("--solo-scraper",    action="store_true")
    parser.add_argument("--solo-rag",        action="store_true")

    args = parser.parse_args()

    if not args.api_key and not args.dry_run:
        sys.exit("ERROR: necesitas --api-key o exportar GOOGLE_API_KEY")

    print()
    print("=" * 60)
    print("   Pipeline FakeNewsBot")
    print("   Scraper → Aumentación → RAG → Entrenamiento LNR")
    print(f"   {datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}")
    print("=" * 60)

    t_inicio = time.time()

    # Paso 1: Scraper
    n_nuevos = 0
    if not args.solo_rag:
        n_nuevos = ejecutar_scraper(args)

    # Paso 2: Aumentación
    n_generados = 0
    if not args.solo_scraper and not args.sin_aumentacion:
        if n_nuevos > 0 or args.solo_rag:
            n_generados = ejecutar_aumentacion(args, n_nuevos)
        else:
            separador()
            log("Sin artículos nuevos → saltando aumentación.")

    # Paso 3: RAG
    if not args.solo_scraper and not args.sin_rag:
        if n_nuevos > 0 or n_generados > 0 or args.solo_rag:
            ejecutar_rag(args)
        else:
            separador()
            log("Sin artículos nuevos → saltando ingesta RAG.")

    # Paso 4: Entrenamiento
    if not args.solo_scraper and not args.sin_entrenamiento:
        if n_nuevos > 0 or n_generados > 0 or args.solo_rag:
            ejecutar_entrenamiento(args)
        else:
            separador()
            log("Sin artículos nuevos → saltando entrenamiento.")

    # Resumen
    separador()
    elapsed = time.time() - t_inicio
    mins, secs = divmod(int(elapsed), 60)
    log(f"Pipeline completado en {mins}m {secs}s")
    if n_nuevos:    log(f"   Artículos scrapeados : {n_nuevos}")
    if n_generados: log(f"   Versiones generadas  : {n_generados}")
    log(f"   JSON dir      : {args.json_dir.resolve()}")
    log(f"   Aumentados dir: {args.aumentados_dir.resolve()}")
    log(f"   Chroma dir    : {args.chroma_dir.resolve()}")
    print()


if __name__ == "__main__":
    main()
