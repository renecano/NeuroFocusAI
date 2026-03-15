"""
NeuroFocus AI - Generador de resumen inteligente
Usa Gemini API para generar un analisis personalizado de la sesion
con recomendaciones practicas para el estudiante.
"""

import os
import urllib.request
import urllib.error
import json


_SYSTEM_PROMPT = """Analiza datos de atencion de un estudiante y responde SOLO con JSON valido.
Idioma: espanol. Segunda persona. Con emojis. Sin tecnicismos. Motivador pero honesto. Breve."""


# Schema con tipos en minusculas (requerido por Gemini 2.5)
_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "titulo":           {"type": "string"},
        "resumen_general":  {"type": "string"},
        "puntuacion_label": {"type": "string"},
        "fortalezas":       {"type": "array", "items": {"type": "string"}, "maxItems": 2},
        "areas_mejora":     {"type": "array", "items": {"type": "string"}, "maxItems": 2},
        "recomendaciones": {
            "type": "array",
            "maxItems": 2,
            "items": {
                "type": "object",
                "properties": {
                    "icono":       {"type": "string"},
                    "titulo":      {"type": "string"},
                    "descripcion": {"type": "string"}
                },
                "required": ["icono", "titulo", "descripcion"]
            }
        },
        "mensaje_final": {"type": "string"}
    },
    "required": [
        "titulo", "resumen_general", "puntuacion_label",
        "fortalezas", "areas_mejora", "recomendaciones", "mensaje_final"
    ]
}


def generate_summary(session_data: dict) -> dict:
    """
    Genera un resumen inteligente de la sesion usando Gemini API.
    """
    prompt = _build_prompt(session_data)

    try:
        result = _call_gemini(prompt)
        # Limpiar posibles bloques markdown que Gemini incluya
        result = result.strip()
        if result.startswith("```"):
            result = result.split("```")[1]
            if result.startswith("json"):
                result = result[4:]
            result = result.strip()
        return json.loads(result)
    except json.JSONDecodeError as e:
        print(f"  [WARN] JSON invalido de Gemini: {e}")
        return _fallback_summary(session_data, str(e))
    except Exception as e:
        print(f"  [WARN] Error llamando Gemini: {e}")
        return _fallback_summary(session_data, str(e))


def _build_prompt(d: dict) -> str:
    mins, secs = divmod(int(d.get("duracion_seg", 0)), 60)
    t_estados  = d.get("tiempo_estados", {})

    lines = [
        "Datos de la sesion de estudio:",
        f"- Duracion total: {mins} minutos {secs} segundos",
        f"- Score de atencion promedio: {d.get('score_promedio', 0)}/100",
        f"- Tiempo en atencion alta: {_fmt(t_estados.get('Atencion alta', 0))}",
        f"- Tiempo en atencion media: {_fmt(t_estados.get('Atencion media', 0))}",
        f"- Tiempo en distraccion: {_fmt(t_estados.get('Distraccion', 0))}",
        f"- Tiempo con fatiga: {_fmt(t_estados.get('Fatiga', 0))}",
        f"- Porcentaje del tiempo atento: {d.get('attention_pct', 0):.0f}%",
        f"- Total de parpadeos detectados: {d.get('total_blinks', 0)}",
        f"- Episodios de fatiga visual: {d.get('fatigue_events', 0)}",
        f"- Episodios de distraccion (dejar de mirar la pantalla): {d.get('distraction_events', 0)}",
        "",
        "Genera un resumen BREVE (max 3 oraciones por campo) usando esos datos.",
    ]
    return "\n".join(lines)


def _call_gemini(prompt: str) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "No se encontro GEMINI_API_KEY. "
            "Configura tu API key antes de ejecutar."
        )

    model = "gemini-2.5-flash"
    url   = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent"
    )

    payload = {
        "system_instruction": {
            "parts": [{"text": _SYSTEM_PROMPT}]
        },
        "contents": [
            {"parts": [{"text": prompt}]}
        ],
        "generationConfig": {
            "response_mime_type": "application/json",
            "response_schema":    _RESPONSE_SCHEMA,
            "temperature":        0.7,
            "maxOutputTokens":    4096,
        }
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type":  "application/json",
            "x-goog-api-key": api_key,
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code}: {body}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"Error de red: {e.reason}")

    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError):
        raise RuntimeError(
            f"Respuesta inesperada de Gemini: {json.dumps(data, ensure_ascii=False)[:300]}"
        )


def _fallback_summary(d: dict, error: str = "") -> dict:
    """Resumen local cuando la API no esta disponible."""
    score = d.get("score_promedio", 0)
    att   = d.get("attention_pct", 0)

    if score >= 80:
        titulo  = "🌟 Sesion excelente"
        resumen = "Tuviste una sesion de estudio muy productiva con alta concentracion."
        label   = "Excelente concentracion"
    elif score >= 60:
        titulo  = "✅ Buena sesion de estudio"
        resumen = "Tu sesion estuvo bien en general, con algunos momentos de distraccion."
        label   = "Buen nivel de atencion"
    else:
        titulo  = "⚠️ Sesion con areas de mejora"
        resumen = "Tu sesion tuvo varios momentos de distraccion o fatiga. Considera descansar mas."
        label   = "Atencion por mejorar"

    return {
        "titulo":           titulo,
        "resumen_general":  resumen,
        "puntuacion_label": label,
        "fortalezas": [
            f"Completaste {_fmt(d.get('duracion_seg', 0))} de sesion",
            f"Mantuviste atencion el {att:.0f}% del tiempo",
        ],
        "areas_mejora": [
            "Reducir episodios de distraccion",
            "Tomar pausas regulares para evitar fatiga visual",
        ],
        "recomendaciones": [
            {
                "icono":       "⏱️",
                "titulo":      "Tecnica Pomodoro",
                "descripcion": "Estudia 25 minutos y descansa 5. Ayuda a sostener la concentracion.",
            },
            {
                "icono":       "👁️",
                "titulo":      "Regla 20-20-20",
                "descripcion": "Cada 20 minutos mira algo lejano 20 segundos para descansar la vista.",
            },
        ],
        "mensaje_final": "Sigue adelante. Cada sesion te ayuda a mejorar.",
    }


def _fmt(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}min {s}s"