"""
entrenar_incremental.py — Acumula artículos nuevos al dataset total y reentrena.

Lógica:
  - Mantiene un archivo maestro: dataset_acumulado.json
  - Cada ejecución añade los artículos nuevos (reales + generados) que aún
    no estén en el acumulado.
  - Reentrena el modelo LNR sobre el dataset completo acumulado.
  - Guarda el modelo actualizado en --modelo-out (por defecto sobreescribe el base).

Uso:
    python entrenar_incremental.py --modelo-base ./modelo_lnr/modelo_final
    python entrenar_incremental.py --modelo-base ./modelo_lnr/modelo_final --dry-run
    python entrenar_incremental.py --modelo-base ./modelo_lnr/modelo_final --epocas 3
"""

import argparse
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

# ── Configuración ──────────────────────────────────────────────────────────────
_DIR = Path(__file__).parent

DEFAULT_MODELO_BASE    = _DIR / "modelo_lnr" / "modelo_final"
DEFAULT_JSON_NUEVOS    = _DIR / "noticias_rag"
DEFAULT_JSON_GENERADOS = _DIR / "noticias_aumentadas"
DEFAULT_ACUMULADO      = _DIR / "dataset_acumulado.json"

MAX_LEN         = 512
BATCH_SIZE      = 4
GRAD_ACCUM      = 16
EPOCAS          = 4
LR              = 1e-5
WARMUP_RATIO    = 0.10
WEIGHT_DECAY    = 0.01
LABEL_SMOOTHING = 0.10

LABEL2ID = {"VERDADERO": 0, "CONTEXTO": 1, "FALSO": 2}
ID2LABEL  = {v: k for k, v in LABEL2ID.items()}
ETIQUETAS_VALIDAS = set(LABEL2ID.keys())


# ── Dataset acumulado ──────────────────────────────────────────────────────────

def uid_articulo(r: dict) -> str:
    import hashlib
    clave = f"{r.get('url', '')}||{r.get('titulo', '')}"
    return hashlib.md5(clave.encode()).hexdigest()[:16]


def cargar_acumulado(path: Path) -> list[dict]:
    if path.exists():
        datos = json.loads(path.read_text(encoding="utf-8"))
        print(f"[Acumulado] {len(datos)} artículos cargados desde {path.name}")
        return datos
    print(f"[Acumulado] {path.name} no existe aún — se creará nuevo")
    return []


def guardar_acumulado(datos: list[dict], path: Path):
    path.write_text(json.dumps(datos, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[Acumulado] {len(datos)} artículos guardados en {path.name}")


def leer_json_dir(carpeta: Path) -> list[dict]:
    registros = []
    for f in sorted(carpeta.glob("*.json")):
        if f.name.startswith("."):
            continue
        try:
            datos = json.loads(f.read_text(encoding="utf-8"))
            if isinstance(datos, list):
                registros.extend(datos)
        except Exception as e:
            print(f"  [AVISO] {f.name}: {e}")
    return registros


def añadir_nuevos_al_acumulado(acumulado: list[dict], carpetas: list[Path]) -> int:
    """
    Lee artículos de las carpetas indicadas, añade los que no estén
    ya en el acumulado. Devuelve el número de artículos añadidos.
    """
    ids_existentes = {uid_articulo(r) for r in acumulado}
    n_añadidos = 0

    for carpeta in carpetas:
        if not carpeta.exists():
            print(f"  [INFO] {carpeta} no existe — saltando")
            continue

        candidatos = leer_json_dir(carpeta)
        for r in candidatos:
            etiqueta = r.get("etiqueta", "").strip().upper()
            if etiqueta not in ETIQUETAS_VALIDAS:
                continue
            uid = uid_articulo(r)
            if uid in ids_existentes:
                continue
            r["etiqueta"] = etiqueta
            acumulado.append(r)
            ids_existentes.add(uid)
            n_añadidos += 1

    return n_añadidos


# ── Dataset y entrenamiento ────────────────────────────────────────────────────

def articulo_a_texto(r: dict) -> str:
    titulo = str(r.get("titulo", "") or "").strip()
    texto  = str(r.get("texto",  "") or "").strip()
    return f"{titulo} {texto}".strip()


def construir_dataset(registros: list[dict], tokenizer):
    from datasets import Dataset
    textos    = [articulo_a_texto(r) for r in registros]
    etiquetas = [LABEL2ID[r["etiqueta"]] for r in registros]
    enc = tokenizer(
        textos, max_length=MAX_LEN, truncation=True,
        padding="max_length", return_tensors=None,
    )
    enc["labels"] = etiquetas
    return Dataset.from_dict(enc)


def calcular_pesos_clase(registros: list[dict]) -> list[float]:
    conteo = Counter(LABEL2ID[r["etiqueta"]] for r in registros)
    total  = sum(conteo.values())
    pesos  = [total / (len(LABEL2ID) * max(conteo.get(i, 1), 1)) for i in range(len(LABEL2ID))]
    norm   = [p / max(pesos) for p in pesos]
    print(f"  Distribución: {dict(Counter(r['etiqueta'] for r in registros))}")
    print(f"  Pesos clase : { {k: f'{norm[v]:.2f}' for k, v in LABEL2ID.items()} }")
    return norm


def crear_trainer(model, tokenizer, train_ds, val_ds, pesos_clase, output_dir, n_epocas):
    import torch
    import torch.nn as nn
    import numpy as np
    from transformers import Trainer, TrainingArguments
    from sklearn.metrics import f1_score, accuracy_score

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pesos_t = torch.tensor(pesos_clase, dtype=torch.float).to(device)

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            loss = nn.CrossEntropyLoss(
                weight=pesos_t, label_smoothing=LABEL_SMOOTHING
            )(outputs.logits, labels)
            return (loss, outputs) if return_outputs else loss

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy":    accuracy_score(labels, preds),
            "f1_macro":    f1_score(labels, preds, average="macro",    zero_division=0),
            "f1_weighted": f1_score(labels, preds, average="weighted", zero_division=0),
        }

    total_steps  = (len(train_ds) // (BATCH_SIZE * GRAD_ACCUM)) * n_epocas
    warmup_steps = max(1, int(total_steps * WARMUP_RATIO))

    training_args = TrainingArguments(
        output_dir                  = str(output_dir / "checkpoints"),
        num_train_epochs            = n_epocas,
        per_device_train_batch_size = BATCH_SIZE,
        per_device_eval_batch_size  = BATCH_SIZE * 2,
        gradient_accumulation_steps = GRAD_ACCUM,
        learning_rate               = LR,
        weight_decay                = WEIGHT_DECAY,
        warmup_steps                = warmup_steps,
        lr_scheduler_type           = "cosine",
        eval_strategy               = "epoch",
        save_strategy               = "epoch",
        load_best_model_at_end      = True,
        metric_for_best_model       = "f1_macro",
        greater_is_better           = True,
        save_total_limit            = 1,
        fp16                        = torch.cuda.is_available(),
        report_to                   = "none",
        logging_steps               = 20,
    )

    return WeightedTrainer(
        model=model, args=training_args,
        train_dataset=train_ds, eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )


# ── Pipeline principal ─────────────────────────────────────────────────────────

def ejecutar_entrenamiento(
    modelo_base:    str,
    modelo_out:     str,
    json_nuevos:    str,
    json_generados: str,
    acumulado_path: str,
    n_epocas:       int  = EPOCAS,
    dry_run:        bool = False,
):
    base_path = Path(modelo_base)
    out_path  = Path(modelo_out)
    acum_path = Path(acumulado_path)

    # 1. Cargar dataset acumulado
    acumulado = cargar_acumulado(acum_path)

    # 2. Añadir artículos nuevos (reales del scraper + generados por Gemini)
    n_añadidos = añadir_nuevos_al_acumulado(
        acumulado,
        carpetas=[Path(json_nuevos), Path(json_generados)],
    )

    print(f"\n[Train] Artículos añadidos esta ejecución : {n_añadidos}")
    print(f"[Train] Dataset total acumulado          : {len(acumulado)}")

    if n_añadidos == 0:
        print("[Train] No hay artículos nuevos — el dataset no ha cambiado. Saltando entrenamiento.")
        return

    if len(acumulado) < 80:
        print(f"[Train] Dataset demasiado pequeño ({len(acumulado)}). Espera a tener más datos.")
        return

    if dry_run:
        dist = Counter(r["etiqueta"] for r in acumulado)
        print(f"\n[DRY RUN] Entrenarías con {len(acumulado)} artículos — {dict(dist)}")
        print(f"  Modelo base  : {base_path}")
        print(f"  Modelo salida: {out_path}")
        return

    # 3. Guardar acumulado actualizado
    guardar_acumulado(acumulado, acum_path)

    # 4. Split estratificado 90/10
    por_clase = defaultdict(list)
    for r in acumulado:
        por_clase[r["etiqueta"]].append(r)

    train_split, val_split = [], []
    for items in por_clase.values():
        random.shuffle(items)
        corte = max(1, int(len(items) * 0.9))
        train_split.extend(items[:corte])
        val_split.extend(items[corte:])

    random.shuffle(train_split)
    print(f"[Train] Train: {len(train_split)} | Val: {len(val_split)}")

    # 5. Cargar modelo
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
    except ImportError:
        sys.exit("ERROR: pip install transformers torch scikit-learn datasets")

    print(f"[Train] Cargando modelo desde {base_path}...")
    tokenizer = AutoTokenizer.from_pretrained(str(base_path))
    model     = AutoModelForSequenceClassification.from_pretrained(
        str(base_path),
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    # 6. Tokenizar y entrenar
    print("[Train] Tokenizando...")
    train_ds = construir_dataset(train_split, tokenizer)
    val_ds   = construir_dataset(val_split,   tokenizer)

    pesos   = calcular_pesos_clase(train_split)
    trainer = crear_trainer(model, tokenizer, train_ds, val_ds, pesos, out_path, n_epocas)

    print(f"\n[Train] Entrenando con {len(train_split)} artículos ({n_epocas} épocas)...")
    t0 = time.time()
    trainer.train()
    mins, secs = divmod(int(time.time() - t0), 60)
    print(f"[Train] Completado en {mins}m {secs}s")

    # 7. Guardar modelo
    out_path.mkdir(parents=True, exist_ok=True)
    import shutil
    tmp_path = out_path.parent / "modelo_temp"
    trainer.save_model(str(tmp_path))
    tokenizer.save_pretrained(str(tmp_path))
    if out_path.exists():
        shutil.rmtree(out_path)
    shutil.copytree(str(tmp_path), str(out_path))
    shutil.rmtree(str(tmp_path))
    print(f"[Train] Modelo guardado en {out_path}")

    # 8. Métricas
    m = trainer.evaluate()
    print(f"\n[Train] Métricas en validación:")
    print(f"  Accuracy   : {m.get('eval_accuracy',    0):.1%}")
    print(f"  F1 Macro   : {m.get('eval_f1_macro',    0):.1%}")
    print(f"  F1 Weighted: {m.get('eval_f1_weighted', 0):.1%}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    import io
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Entrenamiento incremental — acumula datos y reentrena sobre todo"
    )
    parser.add_argument("--modelo-base",    default=str(DEFAULT_MODELO_BASE))
    parser.add_argument("--modelo-out",     default=None,
                        help="Dónde guardar el modelo (por defecto sobreescribe --modelo-base)")
    parser.add_argument("--json-nuevos",    default=str(DEFAULT_JSON_NUEVOS),
                        help="Carpeta noticias_rag/ (artículos reales del scraper)")
    parser.add_argument("--json-generados", default=str(DEFAULT_JSON_GENERADOS),
                        help="Carpeta noticias_aumentadas/ (generados por Gemini)")
    parser.add_argument("--acumulado",      default=str(DEFAULT_ACUMULADO),
                        help="Archivo maestro donde se acumulan todos los artículos")
    parser.add_argument("--epocas",         type=int, default=EPOCAS)
    parser.add_argument("--dry-run",        action="store_true")
    args = parser.parse_args()

    ejecutar_entrenamiento(
        modelo_base    = args.modelo_base,
        modelo_out     = args.modelo_out or args.modelo_base,
        json_nuevos    = args.json_nuevos,
        json_generados = args.json_generados,
        acumulado_path = args.acumulado,
        n_epocas       = args.epocas,
        dry_run        = args.dry_run,
    )


if __name__ == "__main__":
    main()
