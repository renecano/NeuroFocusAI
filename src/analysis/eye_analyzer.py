"""
NeuroFocus AI — Análisis ocular
Responsabilidad: calcular EAR (Eye Aspect Ratio), detectar parpadeos
y estimar señales de fatiga basadas en el comportamiento ocular.
"""

import time
import numpy as np
import sys
import os
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


def _euclidean(p1, p2) -> float:
    return np.linalg.norm(np.array(p1) - np.array(p2))


def _ear(eye_points) -> float:
    """
    Eye Aspect Ratio:
        EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    Los 6 puntos deben estar en orden: extremo_izq, sup1, sup2,
    extremo_der, inf1, inf2
    """
    A = _euclidean(eye_points[1], eye_points[5])
    B = _euclidean(eye_points[2], eye_points[4])
    C = _euclidean(eye_points[0], eye_points[3])
    if C < 1e-6:
        return 0.0
    return (A + B) / (2.0 * C)


class EyeAnalyzer:
    """
    Rastrea parpadeos y señales de fatiga ocular en tiempo real.
    """

    def __init__(self):
        self._blink_counter    = 0      # cuadros consecutivos con EAR bajo
        self._total_blinks     = 0
        self._closed_start     = None   # timestamp cuando los ojos se cerraron
        self._blink_timestamps = deque() # timestamps de parpadeos recientes

    # ── API principal ────────────────────────────────────────
    def update(self, detection_result) -> dict:
        """
        Actualiza el estado ocular con el frame actual.
        Devuelve un dict con las métricas calculadas.
        """
        now = time.time()

        left_pts  = detection_result.get_points(config.LEFT_EYE)
        right_pts = detection_result.get_points(config.RIGHT_EYE)

        ear_left  = _ear(left_pts)
        ear_right = _ear(right_pts)
        ear_avg   = (ear_left + ear_right) / 2.0

        eyes_closed = ear_avg < config.EAR_THRESHOLD

        # ── Conteo de parpadeos ──────────────────────────────
        if eyes_closed:
            self._blink_counter += 1
            if self._closed_start is None:
                self._closed_start = now
        else:
            if self._blink_counter >= config.BLINK_CONSEC_FRAMES:
                self._total_blinks += 1
                self._blink_timestamps.append(now)
            self._blink_counter = 0
            self._closed_start  = None

        # ── Limpiar parpadeos fuera de la ventana de 60 s ────
        cutoff = now - 60.0
        while self._blink_timestamps and self._blink_timestamps[0] < cutoff:
            self._blink_timestamps.popleft()

        blink_rate = len(self._blink_timestamps)   # parpadeos/min

        # ── Tiempo con ojos cerrados ─────────────────────────
        closed_duration = 0.0
        if self._closed_start is not None:
            closed_duration = now - self._closed_start

        # ── Señal de fatiga ocular ───────────────────────────
        fatigue_signal = (
            blink_rate > config.FATIGUE_BLINK_RATE
            or closed_duration > config.CLOSED_EYE_SECONDS
        )

        return {
            "ear":             round(ear_avg, 3),
            "ear_left":        round(ear_left, 3),
            "ear_right":       round(ear_right, 3),
            "eyes_closed":     eyes_closed,
            "blink_rate":      blink_rate,
            "total_blinks":    self._total_blinks,
            "closed_duration": round(closed_duration, 2),
            "fatigue_signal":  fatigue_signal,
        }

    def reset(self) -> None:
        self._blink_counter    = 0
        self._total_blinks     = 0
        self._closed_start     = None
        self._blink_timestamps.clear()