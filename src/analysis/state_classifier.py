"""
NeuroFocus AI — Clasificador de estado del usuario
Responsabilidad: combinar señales visuales para producir un estado
comprensible (ATENCIÓN ALTA / MEDIA / DISTRACCIÓN / FATIGA)
y un score de atención 0-100.
"""

from enum import Enum
import time
from collections import deque
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


class UserState(Enum):
    NO_FACE          = "Sin rostro"
    ATTENTION_HIGH   = "Atencion alta"
    ATTENTION_MEDIUM = "Atencion media"
    DISTRACTION      = "Distraccion"
    FATIGUE          = "Fatiga"


class StateClassifier:
    """
    Combina métricas de EyeAnalyzer y HeadPoseAnalyzer para
    producir un estado y un score de atención continuo.
    """

    def __init__(self):
        self._score_history = deque(maxlen=30)   # historial de scores recientes
        self._session_start = time.time()
        self._state_durations = {s: 0.0 for s in UserState}
        self._last_state = UserState.NO_FACE
        self._last_ts    = time.time()

    # ── API principal ────────────────────────────────────────
    def classify(self, eye_metrics: dict, head_metrics: dict) -> dict:
        """
        Recibe métricas de ambos analizadores y devuelve:
        {
            state: UserState,
            score: int (0-100),
            attention_pct: float (porcentaje acumulado de atención),
            session_seconds: float,
            state_label: str,
            state_color: tuple BGR,
        }
        """
        now = time.time()

        # ── Calcular score base ──────────────────────────────
        score = 100

        # Penalización por fatiga ocular
        if eye_metrics["fatigue_signal"]:
            score -= 40
        elif eye_metrics["eyes_closed"]:
            score -= 20

        # Penalización por distracción de cabeza
        if head_metrics["distraction_signal"]:
            score -= 45
        elif not head_metrics["looking_forward"]:
            score -= 15

        # Penalización por tasa de parpadeo alta
        blink_rate = eye_metrics["blink_rate"]
        if blink_rate > config.FATIGUE_BLINK_RATE:
            score -= min(20, (blink_rate - config.FATIGUE_BLINK_RATE) * 2)

        score = max(0, min(100, score))
        self._score_history.append(score)
        smooth_score = int(sum(self._score_history) / len(self._score_history))

        # ── Clasificar estado ────────────────────────────────
        if eye_metrics["fatigue_signal"] or eye_metrics["closed_duration"] > config.CLOSED_EYE_SECONDS:
            state = UserState.FATIGUE
        elif head_metrics["distraction_signal"]:
            state = UserState.DISTRACTION
        elif smooth_score >= config.ATTENTION_HIGH:
            state = UserState.ATTENTION_HIGH
        elif smooth_score >= config.ATTENTION_MEDIUM:
            state = UserState.ATTENTION_MEDIUM
        else:
            state = UserState.DISTRACTION

        # ── Acumular duración de estados ─────────────────────
        elapsed = now - self._last_ts
        self._state_durations[self._last_state] += elapsed
        self._last_state = state
        self._last_ts    = now

        # ── Porcentaje de atención acumulado ─────────────────
        session_secs = now - self._session_start
        attentive_secs = (
            self._state_durations[UserState.ATTENTION_HIGH]
            + self._state_durations[UserState.ATTENTION_MEDIUM]
        )
        attention_pct = (attentive_secs / max(session_secs, 1)) * 100

        return {
            "state":          state,
            "score":          smooth_score,
            "attention_pct":  round(attention_pct, 1),
            "session_seconds": round(session_secs, 1),
            "state_label":    state.value,
            "state_color":    _state_color(state),
            "state_durations": {s.value: round(d, 1) for s, d in self._state_durations.items()},
        }

    def reset(self) -> None:
        self._score_history.clear()
        self._session_start = time.time()
        self._state_durations = {s: 0.0 for s in UserState}
        self._last_state = UserState.NO_FACE
        self._last_ts    = time.time()

    def summary(self) -> dict:
        """Resumen de la sesión para mostrar al finalizar."""
        session_secs = time.time() - self._session_start
        return {
            "duracion_total_seg": round(session_secs, 1),
            "estados": {s.value: round(d, 1) for s, d in self._state_durations.items()},
        }


# ── Helpers ──────────────────────────────────────────────────
def _state_color(state: UserState):
    mapping = {
        UserState.ATTENTION_HIGH:   config.COLOR_GREEN,
        UserState.ATTENTION_MEDIUM: config.COLOR_YELLOW,
        UserState.DISTRACTION:      config.COLOR_RED,
        UserState.FATIGUE:          config.COLOR_RED,
        UserState.NO_FACE:          config.COLOR_GRAY,
    }
    return mapping.get(state, config.COLOR_WHITE)