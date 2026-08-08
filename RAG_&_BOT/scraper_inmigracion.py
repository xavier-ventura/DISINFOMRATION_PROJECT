"""
scraper_inmigracion.py - Descarga noticias de inmigracion para el RAG.

Fuentes: El Pais / Cadena SER / Maldita Migracion
Schema:  titulo | etiqueta | texto | tema | url | fecha_publicacion | medio | fecha_extraccion

Uso:
    python scraper_inmigracion.py
    python scraper_inmigracion.py --output-dir "ruta/noticias_rag"
    python scraper_inmigracion.py --sin-elpais --sin-ser   # solo Maldita
    python scraper_inmigracion.py --sin-dedup              # redownload todo
"""

import argparse
import csv
import hashlib
import io
import json
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cloudscraper
import feedparser
from bs4 import BeautifulSoup
from dateutil import parser as dateparser

# -- Configuracion por defecto ------------------------------------------------
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "noticias_rag"
DELAY = 2.0  # segundos entre peticiones

SCRAPER = cloudscraper.create_scraper(
    browser={"browser": "firefox", "platform": "windows", "mobile": False}
)

# -- Keywords de relevancia ---------------------------------------------------
KEYWORDS = {
    "inmigracion", "inmigrante", "inmigrantes", "migracion", "migrante",
    "migrantes", "refugiado", "refugiados", "asilo", "solicitante",
    "frontex", "patera", "pateras", "cayuco", "cayucos",
    "ceuta", "melilla", "mena", "deportacion", "expulsion",
    "extranjero", "extranjeros", "irregular", "irregulares",
    "sin papeles", "flujo migratorio", "llegadas", "acogida",
    "xenofobia", "canarias", "lampedusa", "devolucion en caliente",
    # con tildes (por si el texto ya viene limpio)
    "inmigración", "inmmigrante", "migración", "deportación", "expulsión",
}


def es_relevante(texto: str) -> bool:
    t = texto.lower()
    return any(kw in t for kw in KEYWORDS)


# -- Deduplicacion ------------------------------------------------------------

def cargar_ids(dedup_file: Path) -> set:
    return set(json.loads(dedup_file.read_text(encoding="utf-8"))) if dedup_file.exists() else set()


def guardar_ids(ids: set, dedup_file: Path):
    dedup_file.write_text(json.dumps(list(ids)), encoding="utf-8")


def uid(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()[:12]


# -- Limpieza de texto --------------------------------------------------------

def limpiar(texto: str) -> str:
    if not texto:
        return ""
    texto = BeautifulSoup(texto, "html.parser").get_text(separator=" ")
    return re.sub(r"\s+", " ", texto).strip()


# -- Normalizacion de fechas --------------------------------------------------

def normalizar_fecha(raw) -> str:
    if not raw:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if isinstance(raw, str):
        try:
            return dateparser.parse(raw).strftime("%Y-%m-%d")
        except Exception:
            pass
    try:
        return datetime(*raw[:6]).strftime("%Y-%m-%d")
    except Exception:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")


# -- HTTP con bypass de Cloudflare --------------------------------------------

def get_soup(url: str) -> BeautifulSoup | None:
    time.sleep(DELAY)
    try:
        r = SCRAPER.get(url, timeout=20)
        r.raise_for_status()
        return BeautifulSoup(r.text, "html.parser")
    except Exception as e:
        print(f"    x  {url[:80]}: {e}")
        return None


def get_feed(url: str):
    time.sleep(DELAY)
    try:
        r = SCRAPER.get(url, timeout=20)
        r.raise_for_status()
        return feedparser.parse(r.content)
    except Exception as e:
        print(f"    x  Feed {url[:60]}: {e}")
        return feedparser.parse("")


# -- Constructor de registro --------------------------------------------------

def registro(medio, etiqueta, titulo, texto, url, fecha_raw, tema="inmigracion") -> dict:
    return {
        "titulo":            limpiar(titulo),
        "etiqueta":          etiqueta,
        "texto":             limpiar(texto),
        "tema":              tema,
        "url":               url,
        "fecha_publicacion": normalizar_fecha(fecha_raw),
        "medio":             medio,
        "fecha_extraccion":  datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    }


# -- Fuente 1: El Pais --------------------------------------------------------

ELPAIS_FEEDS = [
    "https://feeds.elpais.com/mrss-s/pages/ep/site/elpais.com/section/sociedad/portada",
    "https://feeds.elpais.com/mrss-s/pages/ep/site/elpais.com/section/espana/portada",
    "https://feeds.elpais.com/mrss-s/pages/ep/site/elpais.com/section/internacional/portada",
]

ELPAIS_SELECTORES = [
    "div[data-dtm-region='articulo_cuerpo']",
    "article.a_c",
    "div.a_b_d",
    "div.a_e_txt",
    "div.article_body",
]


def _texto_elpais(url: str) -> str:
    soup = get_soup(url)
    if not soup:
        return ""
    for sel in ELPAIS_SELECTORES:
        bloque = soup.select_one(sel)
        if bloque:
            return limpiar(bloque.get_text(" "))
    return limpiar(" ".join(p.get_text() for p in soup.select("article p")))


def scrape_elpais(ids: set) -> list[dict]:
    resultados = []
    print("--- El Pais ---------------------------------------------------")
    for feed_url in ELPAIS_FEEDS:
        print(f"  Feed: {feed_url}")
        feed = get_feed(feed_url)
        for e in feed.entries:
            titulo = e.get("title", "")
            desc   = e.get("summary", "")
            url    = e.get("link", "")
            if not url or not es_relevante(titulo + " " + desc):
                continue
            art_id = uid(url)
            if art_id in ids:
                print(f"  (dup) {titulo[:60]}")
                continue
            print(f"  + {titulo[:75]}")
            texto = _texto_elpais(url) or desc
            resultados.append(registro(
                "elpais", "VERDADERO", titulo, texto, url,
                e.get("published_parsed") or e.get("published")
            ))
            ids.add(art_id)
    print(f"  => {len(resultados)} nuevos de El Pais")
    return resultados


# -- Fuente 2: elDiario.es ----------------------------------------------------

ELDIARIO_TAG  = "https://www.eldiario.es/temas/inmigracion"
ELDIARIO_BASE = "https://www.eldiario.es"

ELDIARIO_SELECTORES = [
    "div.article-body",
    "div[class*='article-body']",
    "div[itemprop='articleBody']",
    "div.body",
    "div[class*='ArticleBody']",
    "section.article-body",
]


def _urls_desde_eldiario() -> list[str]:
    soup = get_soup(ELDIARIO_TAG)
    if not soup:
        return []
    enlaces = soup.select("h2 a[href], h3 a[href], article a[href], a.title")
    urls = set()
    for a in enlaces:
        href = a.get("href", "")
        if not href:
            continue
        if href.startswith("/"):
            href = ELDIARIO_BASE + href
        if "eldiario.es" in href and "/temas/" not in href and href != ELDIARIO_TAG:
            urls.add(href.split("?")[0])
    return list(urls)[:30]


def _texto_eldiario(url: str) -> str:
    soup = get_soup(url)
    if not soup:
        return ""
    for sel in ELDIARIO_SELECTORES:
        bloque = soup.select_one(sel)
        if bloque:
            return limpiar(bloque.get_text(" "))
    return limpiar(" ".join(p.get_text() for p in soup.select("article p")))


def scrape_eldiario(ids: set) -> list[dict]:
    resultados = []
    print("--- elDiario.es -----------------------------------------------")
    print(f"  Tag: {ELDIARIO_TAG}")
    urls = _urls_desde_eldiario()
    if not urls:
        print("  x  No se encontraron articulos en la pagina de tag")
        print("  => 0 nuevos de elDiario.es")
        return resultados
    print(f"  {len(urls)} articulos encontrados")
    for url in urls:
        art_id = uid(url)
        if art_id in ids:
            continue
        soup = get_soup(url)
        if not soup:
            continue
        titulo = ""
        for sel in ["h1.title", "h1[class*='title']", "h1"]:
            tag = soup.select_one(sel)
            if tag:
                titulo = tag.get_text(strip=True)
                break
        if not titulo or not es_relevante(titulo):
            continue
        print(f"  + {titulo[:75]}")
        texto = _texto_eldiario(url)
        fecha_raw = ""
        for sel in ["time[datetime]", "span[class*='date']", "time"]:
            tag = soup.select_one(sel)
            if tag:
                fecha_raw = tag.get("datetime") or tag.get_text(strip=True)
                break
        resultados.append(registro("eldiario", "VERDADERO", titulo, texto, url, fecha_raw))
        ids.add(art_id)
    print(f"  => {len(resultados)} nuevos de elDiario.es")
    return resultados


# -- Fuente 3: Maldita Migracion ----------------------------------------------

MALDITA_FEED    = "https://maldita.es/migracion/feed/"
MALDITA_PORTADA = "https://maldita.es/migracion/"

MALDITA_SELECTORES = [
    "div.entry-content",
    "div.article-body",
    "div[class*='post-content']",
    "div[class*='content-article']",
    "article section",
]

VEREDICTOS_MAP = {
    "falso":              "FALSO",
    "bulo":               "FALSO",
    "sin pruebas":        "FALSO",
    "enganoso":           "CONTEXTO",
    "engañoso":           "CONTEXTO",
    "descontextualizado": "CONTEXTO",
    "contexto":           "CONTEXTO",
    "misleading":         "CONTEXTO",
    "verdadero":          "VERDADERO",
    "verdad":             "VERDADERO",
}


def _detectar_etiqueta(soup: BeautifulSoup) -> str:
    candidatos = soup.select(
        "span.verdict, div.verdict, span[class*='verdict'], "
        "div[class*='verdict'], span[class*='label'], "
        "div.article-tag, span[class*='tag'], "
        "p[class*='result'], span.etiqueta"
    )
    for c in candidatos:
        t = c.get_text(strip=True).lower()
        for clave, etiqueta in VEREDICTOS_MAP.items():
            if clave in t:
                return etiqueta
    return "FALSO"


def _texto_y_etiqueta_maldita(url: str) -> tuple[str, str]:
    soup = get_soup(url)
    if not soup:
        return "", "FALSO"
    etiqueta = _detectar_etiqueta(soup)
    for sel in MALDITA_SELECTORES:
        bloque = soup.select_one(sel)
        if bloque:
            return limpiar(bloque.get_text(" ")), etiqueta
    texto = limpiar(" ".join(p.get_text() for p in soup.select("article p")))
    return texto, etiqueta


def _urls_desde_portada() -> list[str]:
    soup = get_soup(MALDITA_PORTADA)
    if not soup:
        return []
    enlaces = soup.select("article a[href], h2 a[href], h3 a[href], .entry-title a")
    return list({
        a["href"] for a in enlaces
        if a.get("href", "").startswith("https://maldita.es/migracion/")
           and "/page/" not in a["href"]
    })[:30]


def scrape_maldita(ids: set) -> list[dict]:
    resultados = []
    print("--- Maldita Migracion -----------------------------------------")
    feed     = get_feed(MALDITA_FEED)
    entradas = feed.entries

    if not entradas:
        print("  RSS vacio -> scraping de portada")
        urls = _urls_desde_portada()
        entradas = [
            {"link": u, "title": "", "summary": "", "published_parsed": None}
            for u in urls
        ]

    for e in entradas:
        url    = e.get("link", "")    if isinstance(e, dict) else e.get("link", "")
        titulo = e.get("title", "")   if isinstance(e, dict) else e.get("title", "")
        desc   = e.get("summary", "") if isinstance(e, dict) else e.get("summary", "")
        if not url:
            continue
        art_id = uid(url)
        if art_id in ids:
            print(f"  (dup) {titulo[:60]}")
            continue
        print(f"  + {(titulo or url)[:75]}")
        texto, etiqueta = _texto_y_etiqueta_maldita(url)
        fecha_raw = (
            e.get("published_parsed") or e.get("published")
            if not isinstance(e, dict) else e.get("published_parsed")
        )
        resultados.append(registro(
            "maldita migracion", etiqueta,
            titulo or url, texto or desc, url, fecha_raw
        ))
        ids.add(art_id)

    print(f"  => {len(resultados)} nuevos de Maldita")
    return resultados


# -- Guardado -----------------------------------------------------------------

SCHEMA = ["titulo", "etiqueta", "texto", "tema", "url",
          "fecha_publicacion", "medio", "fecha_extraccion"]


def guardar(registros: list[dict], output_dir: Path, prefijo: str = "noticias"):
    if not registros:
        print("  Sin registros nuevos.")
        return None, None

    ts = datetime.now().strftime("%Y%m%d_%H%M")

    json_path = output_dir / f"{prefijo}_{ts}.json"
    json_path.write_text(
        json.dumps(registros, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"  [JSON] -> {json_path}  ({len(registros)} articulos)")

    csv_path = output_dir / f"{prefijo}_{ts}.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SCHEMA)
        writer.writeheader()
        writer.writerows(registros)
    print(f"  [CSV]  -> {csv_path}")

    dist = Counter(r["etiqueta"] for r in registros)
    print(f"  Etiquetas : {dict(dist)}")
    print(f"  Por medio : {dict(Counter(r['medio'] for r in registros))}")
    return json_path, csv_path


# -- Main ---------------------------------------------------------------------

def main():
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Scraper de noticias de inmigracion")
    parser.add_argument("--output-dir",   default=str(DEFAULT_OUTPUT_DIR),
                        help="Carpeta donde guardar los JSON/CSV")
    parser.add_argument("--sin-dedup",    action="store_true",
                        help="Desactivar deduplicacion (redownload todo)")
    parser.add_argument("--sin-elpais",   action="store_true")
    parser.add_argument("--sin-eldiario", action="store_true")
    parser.add_argument("--sin-maldita",  action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dedup_file = output_dir / ".ids_vistos.json"

    ids = set() if args.sin_dedup else cargar_ids(dedup_file)
    print(f"Carpeta de salida : {output_dir.resolve()}")
    print(f"Deduplicacion     : {'Desactivada' if args.sin_dedup else 'Activada'}")
    print(f"IDs en cache: {len(ids)}\n")

    todos = []

    if not args.sin_elpais:
        todos.extend(scrape_elpais(ids))
        print()

    if not args.sin_eldiario:
        todos.extend(scrape_eldiario(ids))
        print()

    if not args.sin_maldita:
        todos.extend(scrape_maldita(ids))
        print()

    print("-" * 60)
    guardar(todos, output_dir)

    if not args.sin_dedup:
        guardar_ids(ids, dedup_file)

    print(f"\nTotal nuevos: {len(todos)} articulos -> {output_dir}/")
    return len(todos)


if __name__ == "__main__":
    main()
