"""
NeuroFocus AI — Módulo de captura de video
Responsabilidad: abrir la webcam, entregar cuadros y liberar recursos.
"""

import cv2
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


class VideoCapture:
    """Wrapper de cv2.VideoCapture con configuración automática."""

    def __init__(self, index: int = config.CAMERA_INDEX):
        self._cap = cv2.VideoCapture(index)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"No se pudo abrir la cámara con índice {index}. "
                "Verifica que la webcam esté conectada y no esté en uso."
            )
        self._configure()

    # ── Configuración ────────────────────────────────────────
    def _configure(self) -> None:
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.FRAME_WIDTH)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        self._cap.set(cv2.CAP_PROP_FPS,          config.FPS_TARGET)

    # ── Lectura de cuadros ───────────────────────────────────
    def read(self):
        """
        Devuelve (ok: bool, frame: np.ndarray | None).
        El frame está voltado horizontalmente (efecto espejo).
        """
        ok, frame = self._cap.read()
        if ok:
            frame = cv2.flip(frame, 1)   # espejo natural para el usuario
        return ok, frame

    # ── Propiedades ──────────────────────────────────────────
    @property
    def width(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def fps(self) -> float:
        return self._cap.get(cv2.CAP_PROP_FPS)

    @property
    def is_open(self) -> bool:
        return self._cap.isOpened()

    # ── Limpieza ─────────────────────────────────────────────
    def release(self) -> None:
        """Libera la cámara y destruye ventanas de OpenCV."""
        self._cap.release()
        cv2.destroyAllWindows()

    # ── Context manager ──────────────────────────────────────
    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.release()