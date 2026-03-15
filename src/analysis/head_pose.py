"""
NeuroFocus AI — Análisis de orientación de cabeza
Responsabilidad: estimar yaw (giro lateral) y pitch (inclinación vertical)
usando puntos faciales clave y determinar si el usuario mira a la pantalla.
"""

import time
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


class HeadPoseAnalyzer:
    """
    Calcula yaw y pitch aproximados a partir de landmarks 2D.
    No requiere calibración de cámara — usa relaciones geométricas simples
    que son suficientes para clasificar distracción.
    """

    def __init__(self):
        self._distraction_start = None

    # ── API principal ────────────────────────────────────────
    def update(self, detection_result) -> dict:
        """
        Devuelve dict con yaw, pitch y señal de distracción.
        """
        now = time.time()

        nose    = detection_result.get_point(config.NOSE_TIP)
        chin    = detection_result.get_point(config.CHIN)
        l_ear   = detection_result.get_point(config.LEFT_EAR)
        r_ear   = detection_result.get_point(config.RIGHT_EAR)
        forehead = detection_result.get_point(config.FOREHEAD)

        yaw   = self._calc_yaw(nose, l_ear, r_ear)
        pitch = self._calc_pitch(nose, chin, forehead)

        looking_forward = (
            abs(yaw) < config.YAW_THRESHOLD
            and abs(pitch) < config.PITCH_THRESHOLD
        )

        # ── Tiempo fuera de pantalla ─────────────────────────
        if not looking_forward:
            if self._distraction_start is None:
                self._distraction_start = now
        else:
            self._distraction_start = None

        distraction_duration = 0.0
        if self._distraction_start is not None:
            distraction_duration = now - self._distraction_start

        distraction_signal = distraction_duration > config.DISTRACTION_SECONDS

        return {
            "yaw":                  round(yaw, 1),
            "pitch":                round(pitch, 1),
            "looking_forward":      looking_forward,
            "distraction_duration": round(distraction_duration, 2),
            "distraction_signal":   distraction_signal,
        }

    def reset(self) -> None:
        self._distraction_start = None

    # ── Cálculos geométricos ─────────────────────────────────
    @staticmethod
    def _calc_yaw(nose, l_ear, r_ear) -> float:
        """
        Yaw ≈ asimetría horizontal de la nariz respecto al eje oreja-oreja.
        Positivo → girado a la derecha del usuario.
        """
        mid_x = (l_ear[0] + r_ear[0]) / 2.0
        face_width = abs(r_ear[0] - l_ear[0])
        if face_width < 1:
            return 0.0
        # Normalizar a [-1, 1] y escalar a grados aproximados
        ratio = (nose[0] - mid_x) / (face_width / 2.0)
        return ratio * 45.0   # mapeo empírico a grados

    @staticmethod
    def _calc_pitch(nose, chin, forehead) -> float:
        """
        Pitch ≈ posición vertical de la nariz respecto al eje frente-mentón.
        Positivo → cabeza inclinada hacia abajo.
        """
        face_height = abs(chin[1] - forehead[1])
        if face_height < 1:
            return 0.0
        mid_y = (chin[1] + forehead[1]) / 2.0
        ratio = (nose[1] - mid_y) / (face_height / 2.0)
        return ratio * 40.0   # mapeo empírico a grados