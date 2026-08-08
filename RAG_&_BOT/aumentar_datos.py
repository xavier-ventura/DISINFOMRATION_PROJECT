"""
aumentar_datos.py — Generación automática de noticias FALSO/CONTEXTO
a partir de los artículos VERDADERO extraídos por el scraper.

Equivalente al notebook proyiii_data_augmentation.ipynb pero integrado
en el pipeline y adaptado al schema del scraper.

Flujo:
  1. Lee todos los JSON de noticias_rag/ (salida del scraper).
  2. Filtra solo artículos VERDADERO no procesados antes.
  3. Para cada uno, llama a Gemini para generar una versión FALSO o CONTEXTO.
  4. Combina (originales + generados) y guarda en noticias_aumentadas/.
  5. Registra los artículos procesados en .ids_aumentados.json (evita repetir).

Uso:
    python aumentar_datos.py --api-key TU_GOOGLE_API_KEY
    python aumentar_datos.py --api-key TU_CLAVE --solo-verdaderos
    python aumentar_datos.py --api-key TU_CLAVE --max-articulos 50
    python aumentar_datos.py --api-key TU_CLAVE --dry-run

Variables de entorno:
    GOOGLE_API_KEY
"""

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from google import genai
from google.genai import types as genai_types

# ── Configuración ──────────────────────────────────────────────────────────────
_DIR = Path(__file__).parent

DEFAULT_JSON_DIR  = _DIR / "noticias_rag"
DEFAULT_OUT_DIR   = _DIR / "noticias_aumentadas"
DEDUP_FILE        = ".ids_aumentados.json"  # dentro de out_dir

GENERATION_MODEL  = "gemma-3-27b-it"
MAX_RETRIES       = 2
PAUSA_ENTRE_CALLS = 2.0   # segundos entre llamadas a Gemini
GUARDADO_CADA     = 10    # guardar progreso cada N artículos generados

TECNICAS = [
    "exageración de cifras",
    "cambio de contexto temporal",
    "cambio de localización",
    "generalización indebida",
    "uso de términos ambiguos o emocionales",
    "titular engañoso",
    "mezcla de dato real con conclusión falsa",
    "reformulación sesgada de titular",
    "reinterpretación narrativa de datos reales",
    "eliminación de contexto clave",
    "cherry-picking estadístico",
]

PROMPT_AUGMENTATION = """\
Eres un experto en análisis de desinformación sobre inmigración en España.
Transforma esta noticia REAL en una versión {etiqueta_forzada}.

INSTRUCCIONES:
1. Modifica el titulo y el cuerpo (mínimo 800 caracteres) manteniendo el tema de inmigración.
2. Usa exactamente UNA de estas técnicas: {tecnicas}.
3. La etiqueta DEBE ser exactamente "{etiqueta_forzada}":
   - FALSO: dato inventado, sin base real, completamente erróneo.
   - CONTEXTO: el dato base es real pero se presenta de forma engañosa, sesgada o sin contexto.
4. El resultado debe parecer creíble pero ser incorrecto o manipulador.
5. Escribe siempre en español.

NOTICIA ORIGINAL:
titulo: {titulo}
texto: {texto}

Responde ÚNICAMENTE con este JSON (sin bloques de código, sin texto extra):
{{
  "titulo": "...",
  "texto": "...",
  "etiqueta": "{etiqueta_forzada}",
  "tecnica": "nombre exacto de la técnica usada",
  "modificacion": "1-2 frases explicando qué cambiaste y por qué es engañoso"
}}
"""


# ── Deduplicación ──────────────────────────────────────────────────────────────

def uid_articulo(url: str, titulo: str) -> str:
    """ID único reproducible para un artículo."""
    clave = f"{url}||{titulo}"
    return hashlib.md5(clave.encode()).hexdigest()[:16]


def cargar_ids_procesados(out_dir: Path) -> set:
    p = out_dir / DEDUP_FILE
    if p.exists():
        return set(json.loads(p.read_text(encoding="utf-8")))
    return set()


def guardar_ids_procesados(ids: set, out_dir: Path):
    p = out_dir / DEDUP_FILE
    p.write_text(json.dumps(list(ids)), encoding="utf-8")


# ── Lectura de JSONs del scraper ───────────────────────────────────────────────

def leer_articulos_scraper(json_dir: Path) -> list[dict]:
    """Lee todos los JSON de noticias_rag/ y devuelve lista plana."""
    archivos = sorted(json_dir.glob("noticias_*.json"))
    if not archivos:
        print(f"[AVISO] No hay archivos noticias_*.json en {json_dir}")
        return []

    todos = []
    for f in archivos:
        try:
            registros = json.loads(f.read_text(encoding="utf-8"))
            todos.extend(registros)
        except Exception as e:
            print(f"  [ERROR] {f.name}: {e}")

    print(f"[Leer] {len(todos)} artículos en {len(archivos)} archivo(s)")
    return todos


# ── Generación con Gemini ──────────────────────────────────────────────────────

def generar_version_falsa_con_etiqueta(
    articulo: dict,
    client: genai.Client,
    tecnica: str,
    etiqueta_forzada: str = "FALSO",
) -> dict | None:
    """
    Llama a Gemini para generar una versión FALSO/CONTEXTO del artículo.
    Devuelve el dict con los campos generados, o None si falla.
    """
    prompt = PROMPT_AUGMENTATION.format(
        tecnicas=", ".join(TECNICAS),
        titulo=articulo["titulo"][:500],
        texto=articulo["texto"][:3000],
        etiqueta_forzada=etiqueta_forzada,
    )

    for intento in range(MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model=GENERATION_MODEL,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    temperature=0.85,
                    max_output_tokens=2048,
                ),
            )
            texto_json = response.text.strip()
            # Limpiar por si Gemini añade bloques de código
            if "```" in texto_json:
                texto_json = texto_json.split("```")[-2] if texto_json.count("```") >= 2 else texto_json
                texto_json = texto_json.replace("json", "", 1).strip()

            generado = json.loads(texto_json)

            # Validar campos mínimos
            for campo in ("titulo", "texto", "etiqueta", "tecnica", "modificacion"):
                if campo not in generado:
                    raise ValueError(f"Campo '{campo}' ausente en respuesta")

            # Normalizar etiqueta
            etiqueta = generado["etiqueta"].strip().upper()
            if etiqueta not in ("FALSO", "CONTEXTO"):
                etiqueta = "FALSO"
            generado["etiqueta"] = etiqueta

            return generado

        except Exception as e:
            espera = min((intento + 1) * 5, 60)
            err = str(e)
            if "429" in err or "RESOURCE_EXHAUSTED" in err:
                espera = max(espera, 60)
            if intento < MAX_RETRIES - 1:
                print(f"    [retry {intento+1}/{MAX_RETRIES}] {err[:80]} — esperando {espera}s")
                time.sleep(espera)
            else:
                print(f"    [FALLO] Tras {MAX_RETRIES} intentos: {err[:120]}")

    return None


# ── Guardado incremental ───────────────────────────────────────────────────────

def guardar_lote(registros: list[dict], out_dir: Path, prefijo: str = "aumentados"):
    """Guarda un JSON con todos los registros acumulados (sobreescribe)."""
    ts = datetime.now().strftime("%Y%m%d")
    path = out_dir / f"{prefijo}_{ts}.json"
    path.write_text(
        json.dumps(registros, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


# ── Main ───────────────────────────────────────────────────────────────────────

def ejecutar_aumentacion(
    json_dir: str,
    out_dir: str,
    api_key: str,
    max_articulos: int = 0,
    solo_verdaderos: bool = True,
    dry_run: bool = False,
) -> list[dict]:
    """
    Punto de entrada. Devuelve la lista combinada (originales + generados).
    Llamado por pipeline_completo.py o directamente desde CLI.
    """
    json_path = Path(json_dir)
    out_path  = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Leer artículos scrapeados
    articulos = leer_articulos_scraper(json_path)
    if not articulos:
        return []

    # 2. Filtrar los que queremos augmentar (VERDADERO por defecto)
    candidatos = [
        a for a in articulos
        if not solo_verdaderos or a.get("etiqueta", "").upper() == "VERDADERO"
    ]
    print(f"[Augment] {len(candidatos)} artículos candidatos para generar versión falsa")

    # 3. Excluir ya procesados
    ids_procesados = cargar_ids_procesados(out_path)
    pendientes = [
        a for a in candidatos
        if uid_articulo(a.get("url", ""), a.get("titulo", "")) not in ids_procesados
    ]
    print(f"[Augment] {len(pendientes)} pendientes (sin contar {len(ids_procesados)} ya procesados)")

    if max_articulos > 0:
        pendientes = pendientes[:max_articulos]
        print(f"[Augment] Limitado a {max_articulos} artículos por --max-articulos")

    if not pendientes:
        print("[Augment] Nada nuevo que procesar.")
        return articulos  # devolver solo los originales

    if dry_run:
        print(f"\n[DRY RUN] Se generarían {len(pendientes)} versiones falsas.")
        for a in pendientes[:3]:
            print(f"  - [{a['etiqueta']}] {a['titulo'][:70]}")
        return articulos

    # 4. Generar versiones falsas
    client = genai.Client(api_key=api_key)
    generados = []
    import random

    print(f"\nGenerando {len(pendientes)} versiones falsas con {GENERATION_MODEL}...\n")

    for i, original in enumerate(pendientes, 1):
        tecnica = random.choice(TECNICAS)
        etiqueta_forzada = "CONTEXTO" if i % 2 == 0 else "FALSO"
        print(f"  [{i:>4}/{len(pendientes)}] {original['titulo'][:65]}... [{etiqueta_forzada}]")
        resultado = generar_version_falsa_con_etiqueta(original, client, tecnica, etiqueta_forzada)

        if resultado:
            # Construir registro en el mismo schema que el scraper
            registro_falso = {
                "titulo":            resultado["titulo"],
                "etiqueta":          resultado["etiqueta"],
                "texto":             resultado["texto"],
                "tema":              original.get("tema", "inmigracion"),
                "url":               original.get("url", ""),   # misma URL de referencia
                "fecha_publicacion": original.get("fecha_publicacion", ""),
                "medio":             original.get("medio", ""),
                "fecha_extraccion":  datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                # campos extra para el entrenamiento
                "tecnica":           resultado.get("tecnica", tecnica),
                "modificacion":      resultado.get("modificacion", ""),
                "es_generado":       True,
                "url_original":      original.get("url", ""),
            }
            generados.append(registro_falso)
            ids_procesados.add(uid_articulo(
                original.get("url", ""), original.get("titulo", "")
            ))
            print(f"         → {resultado['etiqueta']} [{resultado.get('tecnica','?')[:45]}]")
        else:
            print(f"         → SALTADO (falló)")

        # Guardado incremental
        if i % GUARDADO_CADA == 0 or i == len(pendientes):
            ruta = guardar_lote(generados, out_path)
            guardar_ids_procesados(ids_procesados, out_path)
            print(f"\n  >> Progreso guardado: {len(generados)} generados → {ruta.name}\n")

        time.sleep(PAUSA_ENTRE_CALLS)

    # 5. Resumen
    print(f"\n{'='*55}")
    print(f"  Aumentación completada")
    print(f"  Artículos originales : {len(articulos)}")
    print(f"  Versiones generadas  : {len(generados)}")
    from collections import Counter
    dist = Counter(g["etiqueta"] for g in generados)
    print(f"  Distribución        : {dict(dist)}")
    print(f"  Salida              : {out_path.resolve()}")
    print(f"{'='*55}\n")

    return articulos + generados


def main():
    import io
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Genera versiones FALSO/CONTEXTO de noticias VERDADERO scrapeadas"
    )
    parser.add_argument("--api-key",         default=os.getenv("GOOGLE_API_KEY"))
    parser.add_argument("--json-dir",        default=str(DEFAULT_JSON_DIR),
                        help="Carpeta con los JSON del scraper")
    parser.add_argument("--out-dir",         default=str(DEFAULT_OUT_DIR),
                        help="Carpeta donde guardar las noticias generadas")
    parser.add_argument("--max-articulos",   type=int, default=0,
                        help="Máx. artículos a procesar (0 = sin límite)")
    parser.add_argument("--solo-verdaderos", action="store_true", default=True,
                        help="Solo augmentar artículos etiquetados VERDADERO (default)")
    parser.add_argument("--todos",           action="store_true",
                        help="Augmentar todos los artículos (ignora etiqueta)")
    parser.add_argument("--dry-run",         action="store_true",
                        help="Simular sin llamar a Gemini")
    args = parser.parse_args()

    if not args.api_key and not args.dry_run:
        sys.exit("ERROR: necesitas --api-key o exportar GOOGLE_API_KEY")

    solo_verdaderos = not args.todos

    ejecutar_aumentacion(
        json_dir        = args.json_dir,
        out_dir         = args.out_dir,
        api_key         = args.api_key or "",
        max_articulos   = args.max_articulos,
        solo_verdaderos = solo_verdaderos,
        dry_run         = args.dry_run,
    )


if __name__ == "__main__":
    main()
