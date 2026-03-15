"""
NeuroFocus AI — Módulo de detección facial
Responsabilidad: detectar el rostro y devolver los 478 landmarks faciales
usando MediaPipe Tasks API (v0.10+).

NOTA: Esta versión usa la nueva API mediapipe.tasks (0.10+).
El modelo face_landmarker.task se descarga automáticamente la primera vez.
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python            import vision as mp_vision
from mediapipe.tasks.python.core       import base_options as mp_base
import urllib.request
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

# ── Ruta del modelo ──────────────────────────────────────────
_MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
_MODEL_PATH = os.path.join(os.path.dirname(__file__), "face_landmarker.task")

# ── Conexiones para dibujar malla facial ────────────────────
_TESSELATION = mp_vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION


def _ensure_model() -> None:
    """Descarga el modelo si no existe localmente."""
    if not os.path.exists(_MODEL_PATH):
        print("  [INFO] Descargando modelo face_landmarker.task (~30 MB)...")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        print("  [INFO] Modelo descargado correctamente.")


class FaceDetector:
    """
    Encapsula MediaPipe FaceLandmarker (Tasks API v0.10+).
    Uso:
        detector = FaceDetector()
        result = detector.process(frame)
        if result.detected:
            landmarks = result.landmarks   # lista de (x_px, y_px, z)
    """

    def __init__(self):
        _ensure_model()

        options = mp_vision.FaceLandmarkerOptions(
            base_options=mp_base.BaseOptions(model_asset_path=_MODEL_PATH),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=config.MAX_FACES,
            min_face_detection_confidence=config.DETECTION_CONFIDENCE,
            min_face_presence_confidence=config.DETECTION_CONFIDENCE,
            min_tracking_confidence=config.TRACKING_CONFIDENCE,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)

    # ── Procesamiento principal ──────────────────────────────
    def process(self, frame: np.ndarray) -> "DetectionResult":
        """
        Procesa un cuadro BGR y devuelve un DetectionResult.
        """
        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        detection = self._landmarker.detect(mp_image)

        if not detection.face_landmarks:
            return DetectionResult(detected=False)

        raw_landmarks = detection.face_landmarks[0]

        # Convertir landmarks normalizados [0-1] → píxeles
        landmarks = [
            (int(lm.x * w), int(lm.y * h), lm.z)
            for lm in raw_landmarks
        ]
        return DetectionResult(detected=True, landmarks=landmarks, raw=raw_landmarks)

    # ── Dibujo de landmarks ──────────────────────────────────
    def draw_landmarks(self, frame: np.ndarray, result: "DetectionResult") -> np.ndarray:
        """Dibuja los puntos faciales y la malla sobre el frame (in-place)."""
        if not result.detected:
            return frame

        h, w = frame.shape[:2]
        # Dibujar conexiones de la malla
        for connection in _TESSELATION:
            pt1 = result.raw[connection.start]
            pt2 = result.raw[connection.end]
            x1, y1 = int(pt1.x * w), int(pt1.y * h)
            x2, y2 = int(pt2.x * w), int(pt2.y * h)
            cv2.line(frame, (x1, y1), (x2, y2), (70, 70, 70), 1, cv2.LINE_AA)

        # Dibujar puntos clave de los ojos
        for idx in config.LEFT_EYE + config.RIGHT_EYE:
            if idx < len(result.landmarks):
                x, y, _ = result.landmarks[idx]
                cv2.circle(frame, (x, y), 2, config.COLOR_GREEN, -1, cv2.LINE_AA)

        return frame

    # ── Limpieza ─────────────────────────────────────────────
    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# ── Objeto resultado ─────────────────────────────────────────
class DetectionResult:
    """Resultado devuelto por FaceDetector.process()."""

    def __init__(self, detected: bool, landmarks=None, raw=None):
        self.detected  = detected
        self.landmarks = landmarks or []   # list[(x_px, y_px, z)]
        self.raw       = raw               # lista de NormalizedLandmark

    def get_point(self, index: int):
        """Devuelve (x, y) en píxeles para un landmark dado su índice."""
        if index >= len(self.landmarks):
            return (0, 0)
        x, y, _ = self.landmarks[index]
        return (x, y)

    def get_points(self, indices: list):
        """Devuelve lista de (x, y) para un grupo de índices."""
        return [self.get_point(i) for i in indices]