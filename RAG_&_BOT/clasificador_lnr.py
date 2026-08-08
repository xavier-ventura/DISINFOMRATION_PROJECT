"""
Módulo de inferencia local — Clasificador LNR de bulos sobre inmigración.

Mejoras respecto a v1:
  - Ensemble multi-documento: en lugar de usar un solo documento de contexto,
    combina los top-K más relevantes ponderando por similitud semántica.
    Esto reduce el error cuando el documento más cercano no es el mejor proxy.
  - Umbral de confianza: si la probabilidad del ganador es < UMBRAL, devuelve
    INCIERTO en lugar de una predicción equivocada con alta confianza aparente.
  - Devuelve distribución completa de probabilidades por clase.
  - Compatible con Python 3.14 mediante cargador puro sin extensiones nativas
    (safetensors.torch y torch.frombuffer sobre mmap).
"""

from __future__ import annotations

import sys
import json
import mmap
import struct
import warnings
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

_pipeline_fn = None


def _get_pipeline():
    global _pipeline_fn
    if _pipeline_fn is None:
        from transformers import pipeline as hf_pipeline
        _pipeline_fn = hf_pipeline
    return _pipeline_fn


# ── Presentación visual ────────────────────────────────────────────────────────
LABEL_INFO: dict[str, tuple[str, str]] = {
    'VERDADERO': ('🟢', 'Verdadero'),
    'CONTEXTO':  ('🟡', 'Contexto / Engañoso'),
    'FALSO':     ('🔴', 'Falso / Bulo'),
    'INCIERTO':  ('⚪', 'Incierto'),
}

UMBRAL_CONFIANZA_DEFAULT   = 0.50   # umbral general (se reduce si top doc es fact-checker)
UMBRAL_FACTCHECK_TOP       = 0.42   # umbral reducido cuando el doc más similar es fact-checker
TOP_K_DOCS_DEFAULT         = 6      # cuántos documentos usar en el ensemble
PESO_FACTCHECK             = 1.65   # multiplicador de peso para docs fact-checker (2-6)

SESGO_VERDADERO_DEFAULT    = 1.0    # sin boost a VERDADERO

# Señal de bulo: boost proporcional a FALSO según fracción de docs fact-checker.
# Proporcional (no umbral fijo): con 1/6 docs fact-checker ya aplica 33% del boost.
SEÑAL_BULO_BOOST           = 1.90   # boost máximo a FALSO cuando todos los docs son fact-check
SEÑAL_BULO_UMBRAL          = 0.50   # fracción de normalización (frac/umbral, capped at 1.0)
# Penalización a VERDADERO cuando el doc más similar es un fact-checker
PENALIZACION_VERDADERO_TOP = 0.82   # reducir VERDADERO al 82% si top doc es fact-checker

# Nombres de clave guardados con gamma/beta → weight/bias del modelo en memoria
_KEY_REMAP = {
    'LayerNorm.gamma': 'LayerNorm.weight',
    'LayerNorm.beta':  'LayerNorm.bias',
}

# Mapa dtype safetensors → torch
_SF_DTYPE: dict[str, torch.dtype] = {
    'F32':  torch.float32,
    'F16':  torch.float16,
    'BF16': torch.bfloat16,
    'I64':  torch.int64,
    'I32':  torch.int32,
    'I16':  torch.int16,
    'I8':   torch.int8,
    'U8':   torch.uint8,
    'BOOL': torch.bool,
}

# Mapa dtype safetensors → numpy (para BF16 necesitamos un paso extra)
try:
    import numpy as np
    _NP_DTYPE = {
        'F32':  np.float32,
        'F16':  np.float16,
        'I64':  np.int64,
        'I32':  np.int32,
        'I16':  np.int16,
        'I8':   np.int8,
        'U8':   np.uint8,
        'BOOL': np.bool_,
    }
except ImportError:
    np = None
    _NP_DTYPE = {}


def _es_python_314_o_superior() -> bool:
    return sys.version_info >= (3, 14)


def _cargar_pesos_safetensors_puro(model_path: Path, model) -> None:
    """
    Carga model.safetensors directamente en los parámetros del modelo
    sin usar la extensión nativa de safetensors (que hace segfault en Python 3.14+).

    Usa mmap + torch.frombuffer para leer tensores sin copiar datos en exceso.
    """
    import gc

    sf_path = model_path / 'model.safetensors'
    if not sf_path.exists():
        raise FileNotFoundError(f'No se encontro {sf_path}')

    # Construir mapas de parámetros y buffers del modelo
    param_map  = {n: p for n, p in model.named_parameters()}
    buffer_map = {n: b for n, b in model.named_buffers()}

    with open(sf_path, 'rb') as f:
        header_len = struct.unpack('<Q', f.read(8))[0]
        header     = json.loads(f.read(header_len).decode('utf-8'))
        data_start = 8 + header_len

        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        mv = memoryview(mm)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')   # suprimir "non-writable buffer"

            for raw_name, info in header.items():
                if raw_name == '__metadata__':
                    continue

                # Traducir nombre gamma/beta → weight/bias si hace falta
                name = raw_name
                for old, new in _KEY_REMAP.items():
                    name = name.replace(old, new)

                dtype_str          = info['dtype']
                shape              = info['shape']
                off_start, off_end = info['data_offsets']
                seg                = mv[data_start + off_start : data_start + off_end]

                tdtype = _SF_DTYPE.get(dtype_str)
                if tdtype is None:
                    continue

                if dtype_str == 'BF16' and np is not None:
                    arr = np.frombuffer(seg, dtype=np.uint16)
                    t   = torch.from_numpy(arr.copy()).view(torch.bfloat16)
                else:
                    t = torch.frombuffer(seg, dtype=tdtype)

                if shape:
                    t = t.reshape(shape)

                # Copiar en el parámetro / buffer del modelo y liberar referencia
                if name in param_map:
                    param_map[name].data.copy_(t)
                elif name in buffer_map:
                    buffer_map[name].data.copy_(t)

                del t   # liberar referencia al mmap antes de la siguiente iteración

        # Liberar referencias al mmap (el cierre lo gestiona el GC)
        del mv, seg
        gc.collect()
        # No llamar mm.close() explícitamente: puede fallar si torch aún
        # referencia el buffer. El GC lo cerrará cuando ya no haya referencias.


def _crear_inferencia_manual(model, tokenizer, device_id: int, max_length: int):
    """
    Crea una función de inferencia que imita el comportamiento de
    pipeline('text-classification', top_k=None).

    Devuelve list[list[{'label': str, 'score': float}]].
    """
    torch_device = torch.device('cpu' if device_id < 0 else f'cuda:{device_id}')
    model.to(torch_device)
    model.eval()

    id2label = model.config.id2label

    @torch.no_grad()
    def infer(textos: list[str]) -> list[list[dict]]:
        enc = tokenizer(
            textos,
            max_length=max_length,
            truncation=True,
            padding=True,
            return_tensors='pt',
        )
        enc = {k: v.to(torch_device) for k, v in enc.items()}
        logits = model(**enc).logits                        # (B, n_labels)
        probs  = F.softmax(logits, dim=-1).cpu().tolist()  # (B, n_labels)

        resultados = []
        for prob_row in probs:
            resultados.append([
                {'label': id2label[i], 'score': p}
                for i, p in enumerate(prob_row)
            ])
        return resultados

    return infer


class ClasificadorLNR:
    """
    Clasificador de noticias sobre inmigración con ensemble multi-documento.

    Parámetros
    ----------
    model_path : str | Path
        Carpeta con el modelo guardado (config.json + model.safetensors…).
    device : str | int
        'auto' detecta GPU; 'cpu' fuerza CPU; 0 usa la primera GPU.
    max_length : int
        Máximo tokens (debe coincidir con MAX_LEN del entrenamiento: 512).
    umbral_confianza : float
        Confianza mínima para emitir un veredicto. Por debajo → INCIERTO.
    """

    def __init__(
        self,
        model_path: str | Path,
        device: str | int = 'auto',
        max_length: int = 512,
        umbral_confianza: float = UMBRAL_CONFIANZA_DEFAULT,
        sesgo_verdadero: float = SESGO_VERDADERO_DEFAULT,
    ) -> None:
        self.model_path       = Path(model_path)
        self.max_length       = max_length
        self.umbral_confianza = umbral_confianza
        self.sesgo_verdadero  = sesgo_verdadero

        if not self.model_path.exists():
            raise FileNotFoundError(
                f'No se encontró el modelo en: {self.model_path}\n'
                'Entrena el modelo primero ejecutando entrenar_modelo.ipynb'
            )

        if device == 'auto':
            device = 0 if torch.cuda.is_available() else -1

        if _es_python_314_o_superior():
            # Python 3.14+: safetensors.torch hace segfault con esta combinación
            # de versiones. Cargamos pesos directamente con mmap + torch.frombuffer.
            print(f'  [Python {sys.version_info.major}.{sys.version_info.minor}] '
                  f'usando cargador nativo (sin extension safetensors)')
            from transformers import (
                AutoModelForSequenceClassification,
                AutoConfig,
                AutoTokenizer,
            )
            tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
            config    = AutoConfig.from_pretrained(str(self.model_path))
            model     = AutoModelForSequenceClassification.from_config(config)
            _cargar_pesos_safetensors_puro(self.model_path, model)
            self._pipe = _crear_inferencia_manual(model, tokenizer, device, max_length)
        else:
            pipeline = _get_pipeline()
            self._pipe = pipeline(
                'text-classification',
                model=str(self.model_path),
                tokenizer=str(self.model_path),
                device=device,
                truncation=True,
                max_length=self.max_length,
                top_k=None,
            )

        print(f'[OK] ClasificadorLNR cargado desde {self.model_path.name}')

    # ── Método principal ───────────────────────────────────────────────────────

    def clasificar_con_docs(
        self,
        query: str,
        docs: list[dict],
        top_k: int = TOP_K_DOCS_DEFAULT,
    ) -> dict:
        """
        Clasifica la query usando un ensemble de los top-K documentos más
        relevantes recuperados por ChromaDB.

        Algoritmo:
          1. Seleccionar los top_k docs con menor distancia semántica.
          2. Para cada doc, construir el input: query + titular + texto.
          3. Obtener distribución de probabilidad completa [p_V, p_C, p_F].
          4. Promediar ponderando por relevancia (1 − distancia).
          5. Si max(prob) < umbral → devolver INCIERTO.

        Parámetros
        ----------
        query : str
            Consulta o afirmación del usuario.
        docs : list[dict]
            Documentos recuperados por ChromaDB (cada uno con 'titular',
            'texto', 'distancia', 'fuente', 'tipo_fuente').
        top_k : int
            Número máximo de documentos a usar en el ensemble.
        """
        if not docs:
            return self._clasificar_sin_contexto(query)

        # Usar todos los docs ordenados por similitud (sin filtrar por tipo_fuente).
        # Filtrar solo por tipo_fuente='verdad' excluía los fact-checks de Maldita,
        # que son la señal más fuerte de que una afirmación es un bulo.
        docs_sel = sorted(docs, key=lambda d: d['distancia'])[:top_k]

        entradas = []
        pesos    = []
        # 'alerta' es otra etiqueta de fact-check de Maldita
        tipos_factcheck  = {'bulo', 'fact-check', 'alerta'}
        doc_mas_similar  = docs_sel[0]
        top_es_factcheck = doc_mas_similar.get('tipo_fuente') in tipos_factcheck

        for i, doc in enumerate(docs_sel):
            entrada = query + ' ' + doc.get('titular', '') + ' ' + doc.get('texto', '')
            entradas.append(entrada)
            peso_base = max(0.01, 1.0 - doc['distancia'])
            if doc.get('tipo_fuente') in tipos_factcheck:
                multiplicador = 2.0 if i == 0 and top_es_factcheck else PESO_FACTCHECK
                peso_base *= multiplicador
            pesos.append(peso_base)

        resultados_batch = self._pipe(entradas)

        label_set  = {item['label'] for res in resultados_batch for item in res}
        prob_acum  = {lbl: 0.0 for lbl in label_set}
        peso_total = sum(pesos)

        for resultado_doc, peso in zip(resultados_batch, pesos):
            w = peso / peso_total
            for item in resultado_doc:
                prob_acum[item['label']] += item['score'] * w

        # Señal de bulo proporcional: boost a FALSO en funcion de la fraccion de
        # docs fact-checker. Sin umbral fijo: 1/6 docs ya aplica 33% del boost maximo.
        frac_bulo = sum(1 for d in docs_sel if d.get('tipo_fuente') in tipos_factcheck) / len(docs_sel)
        prob_acum = self._aplicar_señal_bulo(prob_acum, frac_bulo, top_es_factcheck)

        prob_decision = self._aplicar_sesgo(prob_acum)
        etiqueta  = max(prob_decision, key=prob_decision.get)
        confianza = prob_decision[etiqueta]

        # Umbral dinamico: mas permisivo cuando el doc mas similar es fact-checker
        umbral = UMBRAL_FACTCHECK_TOP if top_es_factcheck else self.umbral_confianza

        # Override: el doc mas relevante es un fact-checker y el modelo tiende a VERDADERO.
        # El fact-checker contradice la afirmacion -> redirigir a FALSO o CONTEXTO.
        if top_es_factcheck and etiqueta == 'VERDADERO':
            candidatos = {k: v for k, v in prob_acum.items() if k in ('FALSO', 'CONTEXTO')}
            if candidatos:
                etiqueta  = max(candidatos, key=candidatos.get)
                confianza = candidatos[etiqueta]
                return self._formatear_resultado(etiqueta, confianza, prob_acum, n_docs=len(docs_sel))

        if confianza < umbral:
            # Con doc fact-checker en top: decidir entre FALSO y CONTEXTO
            # en lugar de devolver INCIERTO (el modelo duda pero la señal es clara)
            if top_es_factcheck:
                candidatos = {k: v for k, v in prob_acum.items() if k in ('FALSO', 'CONTEXTO')}
                if candidatos:
                    etiqueta  = max(candidatos, key=candidatos.get)
                    confianza = candidatos[etiqueta]
                    return self._formatear_resultado(etiqueta, confianza, prob_acum, n_docs=len(docs_sel))
            return self._resultado_incierto(prob_acum, n_docs=len(docs_sel))

        return self._formatear_resultado(etiqueta, confianza, prob_acum, n_docs=len(docs_sel))

    # ── Método de compatibilidad (sin docs de contexto) ───────────────────────

    def clasificar(self, titulo: str, texto: str = '') -> dict:
        """
        Clasifica sin documentos de contexto externo.
        Útil para pruebas rápidas; en producción usar clasificar_con_docs().
        """
        entrada = f'{titulo} {texto}'.strip()
        resultado_batch = self._pipe([entrada])[0]
        prob_acum = {item['label']: item['score'] for item in resultado_batch}
        prob_decision = self._aplicar_sesgo(prob_acum)
        etiqueta  = max(prob_decision, key=prob_decision.get)
        confianza = prob_decision[etiqueta]

        if confianza < self.umbral_confianza:
            return self._resultado_incierto(prob_acum, n_docs=0)

        return self._formatear_resultado(etiqueta, confianza, prob_acum, n_docs=0)

    # ── Señal de bulo por tipo de documento ──────────────────────────────────

    def _aplicar_señal_bulo(
        self, prob_acum: dict[str, float], frac_bulo: float, top_es_factcheck: bool = False
    ) -> dict[str, float]:
        """
        Boost proporcional a FALSO segun fraccion de docs fact-checker.
        Sin umbral fijo: cualquier fraccion > 0 activa la señal, escalada
        linealmente hasta SEÑAL_BULO_BOOST cuando frac_bulo >= SEÑAL_BULO_UMBRAL.
        Ademas, si el doc mas similar es un fact-checker, penaliza VERDADERO.
        """
        if frac_bulo == 0 or 'FALSO' not in prob_acum:
            return prob_acum
        ajustado = dict(prob_acum)
        # Boost proporcional: escala lineal entre 1.0 y SEÑAL_BULO_BOOST
        factor_escala = min(frac_bulo / SEÑAL_BULO_UMBRAL, 1.0)
        boost = 1.0 + (SEÑAL_BULO_BOOST - 1.0) * factor_escala
        ajustado['FALSO'] *= boost
        # Penalizar VERDADERO cuando el doc mas relevante es un fact-checker
        if top_es_factcheck and 'VERDADERO' in ajustado:
            ajustado['VERDADERO'] *= PENALIZACION_VERDADERO_TOP
        total = sum(ajustado.values())
        return {k: v / total for k, v in ajustado.items()}

    # ── Corrección de sesgo ───────────────────────────────────────────────────

    def _aplicar_sesgo(self, prob_acum: dict[str, float]) -> dict[str, float]:
        """
        Aplica un boost a VERDADERO para compensar la subestimación sistemática
        causada por el desajuste entre entrenamiento (artículos completos) e
        inferencia RAG (query + documento de fact-check).

        El boost solo cambia QUÉ clase gana; las probabilidades reportadas al
        usuario siguen siendo las originales sin sesgo (más honestas).
        """
        if self.sesgo_verdadero == 1.0 or 'VERDADERO' not in prob_acum:
            return prob_acum
        ajustado = dict(prob_acum)
        ajustado['VERDADERO'] *= self.sesgo_verdadero
        total = sum(ajustado.values())
        return {k: v / total for k, v in ajustado.items()}

    # ── Helpers internos ──────────────────────────────────────────────────────

    def _clasificar_sin_contexto(self, query: str) -> dict:
        return self.clasificar(query)

    def _formatear_resultado(
        self,
        etiqueta: str,
        confianza: float,
        prob_acum: dict[str, float],
        n_docs: int,
    ) -> dict:
        emoji, nombre = LABEL_INFO.get(etiqueta, ('⚪', etiqueta))
        probs_ordenadas = sorted(prob_acum.items(), key=lambda x: x[1], reverse=True)
        detalle = '  '.join(
            f'{LABEL_INFO.get(lbl, ("",""))[0]} {lbl}: {p:.1%}'
            for lbl, p in probs_ordenadas
        )
        return {
            'etiqueta':        etiqueta,
            'confianza':       round(confianza, 4),
            'emoji':           emoji,
            'nombre':          nombre,
            'es_incierto':     False,
            'n_docs_usados':   n_docs,
            'distribucion':    {lbl: round(p, 4) for lbl, p in prob_acum.items()},
            'detalle_probs':   detalle,
            'texto_resultado': f'{emoji} **{nombre}** (confianza: {confianza:.1%})',
        }

    def _resultado_incierto(
        self,
        prob_acum: dict[str, float],
        n_docs: int,
    ) -> dict:
        emoji, nombre = LABEL_INFO['INCIERTO']
        mejor_lbl  = max(prob_acum, key=prob_acum.get)
        mejor_prob = prob_acum[mejor_lbl]
        probs_ord  = sorted(prob_acum.items(), key=lambda x: x[1], reverse=True)
        detalle    = '  '.join(
            f'{LABEL_INFO.get(lbl, ("",""))[0]} {lbl}: {p:.1%}'
            for lbl, p in probs_ord
        )
        return {
            'etiqueta':        'INCIERTO',
            'confianza':       round(mejor_prob, 4),
            'emoji':           emoji,
            'nombre':          nombre,
            'es_incierto':     True,
            'n_docs_usados':   n_docs,
            'distribucion':    {lbl: round(p, 4) for lbl, p in prob_acum.items()},
            'detalle_probs':   detalle,
            'texto_resultado': (
                f'{emoji} **{nombre}** — confianza insuficiente ({mejor_prob:.1%}). '
                f'Tendencia: {LABEL_INFO.get(mejor_lbl, ("",""))[0]} {mejor_lbl}'
            ),
        }

    # ── Utilidades ────────────────────────────────────────────────────────────

    @property
    def disponible(self) -> bool:
        return self._pipe is not None

    def __repr__(self) -> str:
        return (
            f'ClasificadorLNR('
            f'model="{self.model_path.name}", '
            f'umbral={self.umbral_confianza}, '
            f'sesgo_verdadero={self.sesgo_verdadero}, '
            f'max_length={self.max_length})'
        )


# ── Carga con fallback ─────────────────────────────────────────────────────────

def cargar_clasificador(
    model_path: str | Path,
    silencioso: bool = False,
) -> Optional[ClasificadorLNR]:
    """
    Intenta cargar el ClasificadorLNR. Devuelve None si el modelo no existe
    o faltan dependencias (permite arranque degradado en modo Gemini puro).
    """
    try:
        return ClasificadorLNR(model_path)
    except FileNotFoundError as e:
        if not silencioso:
            print(f'[INFO] Clasificador LNR no disponible: {e}')
        return None
    except ImportError as e:
        if not silencioso:
            print(f'[INFO] Dependencias del clasificador no instaladas: {e}')
        return None
