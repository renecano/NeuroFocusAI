"""
NeuroFocus AI - Dashboard visual rediseñado
Panel lateral moderno con metricas, barras de progreso y consejos dinamicos.
NOTA: Solo se usan caracteres ASCII para compatibilidad total con OpenCV en Windows.
"""

import cv2
import numpy as np
import time
import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from analysis.state_classifier import UserState

# ── Paleta de colores ────────────────────────────────────────
C_BG_DARK   = (18,  18,  28)    # fondo principal del panel
C_BG_CARD   = (30,  32,  48)    # fondo de tarjeta/sección
C_ACCENT    = (255, 180,  40)    # amarillo dorado — acento
C_GREEN     = (80,  210, 100)
C_YELLOW    = (60,  200, 240)
C_RED       = (70,   80, 230)
C_BLUE      = (210, 140,  60)    # azul acero para detalles
C_WHITE     = (240, 240, 250)
C_GRAY_LT   = (140, 140, 160)
C_GRAY_DK   = (55,   55,  72)
C_BORDER    = (50,   55,  80)

# ── Consejos por estado ───────────────────────────────────────
# Se usan solo ASCII: sin tildes ni enies
TIPS = {
    UserState.ATTENTION_HIGH: [
        "Excelente! Sigue asi.",
        "Concentracion optima.",
        "Muy buen ritmo de estudio.",
        "Estas en modo flujo!",
    ],
    UserState.ATTENTION_MEDIUM: [
        "Intenta eliminar distracciones.",
        "Toma agua si llevas mucho tiempo.",
        "Ajusta tu postura si es necesario.",
        "Divide la tarea en partes.",
    ],
    UserState.DISTRACTION: [
        "Vuelve la vista a la pantalla.",
        "Cierra pestanas innecesarias.",
        "Haz una pausa de 2 min y regresa.",
        "Pon musica instrumental si ayuda.",
    ],
    UserState.FATIGUE: [
        "Toma un descanso de 5-10 min.",
        "Haz parpadeos deliberados.",
        "Aleja la vista 20 seg (regla 20-20-20).",
        "Considera una siesta corta.",
    ],
    UserState.NO_FACE: [
        "Posicionate frente a la camara.",
        "Asegurate de tener buena luz.",
    ],
}

# Etiquetas ASCII para estados (sin tildes)
STATE_LABELS_ASCII = {
    UserState.ATTENTION_HIGH:   "ATENCION ALTA",
    UserState.ATTENTION_MEDIUM: "ATENCION MEDIA",
    UserState.DISTRACTION:      "DISTRACCION",
    UserState.FATIGUE:          "FATIGA",
    UserState.NO_FACE:          "SIN ROSTRO",
}

STATE_COLORS = {
    UserState.ATTENTION_HIGH:   C_GREEN,
    UserState.ATTENTION_MEDIUM: C_YELLOW,
    UserState.DISTRACTION:      C_RED,
    UserState.FATIGUE:          C_RED,
    UserState.NO_FACE:          C_GRAY_LT,
}


class Dashboard:
    """
    Layout:
    ┌──────────────────────┬──────────────────┐
    │   Video en vivo      │  [NF AI titulo]  │
    │   + malla facial     │  [Estado badge]  │
    │   + barra score      │  [Score ring]    │
    │                      │  [Metricas]      │
    │                      │  [Consejo]       │
    │                      │  [Session log]   │
    └──────────────────────┴──────────────────┘
    """

    PANEL_W = 320

    def __init__(self, frame_w: int, frame_h: int):
        self._fw   = frame_w
        self._fh   = frame_h
        self._pw   = self.PANEL_W
        self._font = cv2.FONT_HERSHEY_SIMPLEX
        self._font_b = cv2.FONT_HERSHEY_DUPLEX
        self._start = time.time()
        self._tip_index = 0
        self._tip_timer = time.time()

    # ════════════════════════════════════════════════════════
    #  API PRINCIPAL
    # ════════════════════════════════════════════════════════
    def render(
        self,
        frame: np.ndarray,
        classification: dict | None,
        eye_metrics:    dict | None,
        head_metrics:   dict | None,
        face_detected:  bool,
    ) -> np.ndarray:

        panel = np.full((self._fh, self._pw, 3), C_BG_DARK, dtype=np.uint8)

        state = classification["state"] if classification else UserState.NO_FACE
        color = STATE_COLORS[state]

        # ── Video overlay ──
        if face_detected and classification:
            self._overlay_state_bar(frame, classification)
            self._overlay_score_bar(frame, classification["score"])
        else:
            self._overlay_no_face(frame)

        self._overlay_timer(frame)

        # ── Panel ──
        self._panel_header(panel, state, color)
        if face_detected and classification:
            self._panel_score_ring(panel, classification["score"], color)
            self._panel_metrics(panel, classification, eye_metrics, head_metrics)
            self._panel_tip(panel, state)
            self._panel_session(panel, classification)
        else:
            self._panel_waiting(panel)

        self._panel_footer(panel)

        # ── Separador vertical ──
        sep = np.full((self._fh, 3, 3), C_BORDER, dtype=np.uint8)
        return np.hstack([frame, sep, panel])

    # ════════════════════════════════════════════════════════
    #  OVERLAY SOBRE EL VIDEO
    # ════════════════════════════════════════════════════════
    def _overlay_state_bar(self, frame: np.ndarray, cls: dict) -> None:
        state = cls["state"]
        label = STATE_LABELS_ASCII[state]
        color = STATE_COLORS[state]

        # Gradiente oscuro arriba
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (self._fw, 52), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        # Punto de color + estado
        cv2.circle(frame, (16, 26), 7, color, -1, cv2.LINE_AA)
        cv2.circle(frame, (16, 26), 7, C_WHITE, 1, cv2.LINE_AA)
        cv2.putText(frame, label, (30, 33),
                    self._font_b, 0.72, color, 2, cv2.LINE_AA)

        # Borde de alerta parpadeante
        if state in (UserState.FATIGUE, UserState.DISTRACTION):
            pulse = int(abs(math.sin(time.time() * 3)) * 3) + 2
            cv2.rectangle(frame, (0, 0), (self._fw - 1, self._fh - 1), color, pulse)

    def _overlay_score_bar(self, frame: np.ndarray, score: int) -> None:
        bx, by = 10, self._fh - 28
        bw     = self._fw - 20
        bh     = 14

        # Fondo redondeado
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), C_GRAY_DK, -1)
        # Relleno
        filled = max(0, int(bw * score / 100))
        col    = _score_color(score)
        if filled > 0:
            cv2.rectangle(frame, (bx, by), (bx + filled, by + bh), col, -1)
        # Etiqueta encima
        cv2.putText(frame, f"Atencion: {score}%",
                    (bx, by - 7), self._font, 0.48, C_WHITE, 1, cv2.LINE_AA)

    def _overlay_no_face(self, frame: np.ndarray) -> None:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (self._fw, self._fh), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
        msg = "Rostro no detectado"
        tw, _ = cv2.getTextSize(msg, self._font_b, 0.75, 2)[0]
        cv2.putText(frame, msg,
                    ((self._fw - tw) // 2, self._fh // 2),
                    self._font_b, 0.75, C_RED, 2, cv2.LINE_AA)

    def _overlay_timer(self, frame: np.ndarray) -> None:
        elapsed = int(time.time() - self._start)
        m, s = divmod(elapsed, 60)
        cv2.putText(frame, f"{m:02d}:{s:02d}",
                    (self._fw - 68, self._fh - 38),
                    self._font, 0.48, C_GRAY_LT, 1, cv2.LINE_AA)

    # ════════════════════════════════════════════════════════
    #  SECCIONES DEL PANEL
    # ════════════════════════════════════════════════════════
    def _panel_header(self, panel: np.ndarray, state: UserState, color) -> None:
        # Franja de color en la parte superior
        cv2.rectangle(panel, (0, 0), (self._pw, 48), color, -1)
        # Logo / título
        cv2.putText(panel, "NeuroFocus", (10, 20),
                    self._font_b, 0.62, C_BG_DARK, 2, cv2.LINE_AA)
        cv2.putText(panel, "AI", (172, 20),
                    self._font_b, 0.62, C_BG_DARK, 2, cv2.LINE_AA)
        # Sub-label
        cv2.putText(panel, "Monitor de atencion",
                    (10, 40), self._font, 0.38, C_BG_DARK, 1, cv2.LINE_AA)

    def _panel_score_ring(self, panel: np.ndarray, score: int, color) -> None:
        """Círculo de progreso para el score."""
        cx, cy, r = self._pw // 2, 115, 44
        # Fondo del anillo
        cv2.circle(panel, (cx, cy), r, C_GRAY_DK, 8, cv2.LINE_AA)
        # Arco de progreso (aproximado con líneas)
        angle = int(360 * score / 100)
        for deg in range(-90, -90 + angle, 3):
            rad = math.radians(deg)
            x1  = int(cx + r * math.cos(rad))
            y1  = int(cy + r * math.sin(rad))
            cv2.circle(panel, (x1, y1), 4, color, -1, cv2.LINE_AA)
        # Texto central
        cv2.putText(panel, f"{score}%",
                    (cx - 22, cy + 8), self._font_b, 0.78, C_WHITE, 2, cv2.LINE_AA)
        cv2.putText(panel, "score",
                    (cx - 16, cy + 22), self._font, 0.38, C_GRAY_LT, 1, cv2.LINE_AA)

    def _panel_metrics(
        self, panel: np.ndarray,
        cls: dict, eye: dict, head: dict,
    ) -> None:
        y = 172
        pw = self._pw

        # ── Tarjeta: Ojos ──
        _card(panel, 8, y, pw - 8, y + 94, C_BG_CARD)
        cv2.putText(panel, "OJOS", (16, y + 16),
                    self._font, 0.42, C_ACCENT, 1, cv2.LINE_AA)
        _hline(panel, y + 22, pw)

        ear_pct = min(100, int(eye["ear"] * 300))   # EAR 0.33 -> 100%
        _mini_bar(panel, "EAR", ear_pct, 16, y + 38, pw - 20, C_BLUE)
        _mini_bar(panel, "Parp/min",
                  min(100, int(eye["blink_rate"] * 2)), 16, y + 60, pw - 20, C_YELLOW)

        fat_c = C_RED if eye["fatigue_signal"] else C_GREEN
        fat_t = "Fatiga: SI" if eye["fatigue_signal"] else "Fatiga: No"
        cv2.putText(panel, fat_t, (16, y + 82),
                    self._font, 0.42, fat_c, 1, cv2.LINE_AA)
        blink_t = f"Parpadeos totales: {eye['total_blinks']}"
        cv2.putText(panel, blink_t, (pw // 2 - 10, y + 82),
                    self._font, 0.38, C_GRAY_LT, 1, cv2.LINE_AA)

        # ── Tarjeta: Cabeza ──
        y += 102
        _card(panel, 8, y, pw - 8, y + 90, C_BG_CARD)
        cv2.putText(panel, "CABEZA", (16, y + 16),
                    self._font, 0.42, C_ACCENT, 1, cv2.LINE_AA)
        _hline(panel, y + 22, pw)

        yaw_pct   = min(100, int(abs(head["yaw"])   / config.YAW_THRESHOLD   * 100))
        pitch_pct = min(100, int(abs(head["pitch"]) / config.PITCH_THRESHOLD * 100))
        _mini_bar(panel, f"Yaw  {head['yaw']:+.0f}g", yaw_pct,
                  16, y + 38, pw - 20, C_BLUE)
        _mini_bar(panel, f"Pitch{head['pitch']:+.0f}g", pitch_pct,
                  16, y + 60, pw - 20, C_BLUE)

        fwd_c = C_GREEN if head["looking_forward"] else C_YELLOW
        fwd_t = "Al frente: Si" if head["looking_forward"] else "Al frente: No"
        cv2.putText(panel, fwd_t, (16, y + 80),
                    self._font, 0.42, fwd_c, 1, cv2.LINE_AA)
        dist_c = C_RED if head["distraction_signal"] else C_GRAY_LT
        dist_t = "DISTRACCION" if head["distraction_signal"] else ""
        cv2.putText(panel, dist_t, (pw // 2, y + 80),
                    self._font, 0.42, dist_c, 1, cv2.LINE_AA)

    def _panel_tip(self, panel: np.ndarray, state: UserState) -> None:
        """Consejo dinámico que rota cada 8 segundos."""
        tips = TIPS.get(state, TIPS[UserState.NO_FACE])
        # Rotar cada 8 segundos
        if time.time() - self._tip_timer > 8:
            self._tip_index = (self._tip_index + 1) % len(tips)
            self._tip_timer = time.time()
        tip = tips[self._tip_index % len(tips)]

        y = 372
        _card(panel, 8, y, self._pw - 8, y + 54, (28, 44, 38))
        cv2.putText(panel, "CONSEJO", (16, y + 16),
                    self._font, 0.40, C_GREEN, 1, cv2.LINE_AA)
        _hline(panel, y + 21, self._pw)
        # Dividir en dos líneas si es largo
        words = tip.split()
        line1, line2 = _wrap(words, 28)
        cv2.putText(panel, line1, (14, y + 36),
                    self._font, 0.42, C_WHITE, 1, cv2.LINE_AA)
        if line2:
            cv2.putText(panel, line2, (14, y + 50),
                        self._font, 0.42, C_WHITE, 1, cv2.LINE_AA)

    def _panel_session(self, panel: np.ndarray, cls: dict) -> None:
        """Tiempo por estado en la sesión."""
        y = 434
        cv2.putText(panel, "SESION", (10, y),
                    self._font, 0.40, C_ACCENT, 1, cv2.LINE_AA)
        _hline(panel, y + 4, self._pw)
        y += 18
        colors_map = {
            "Atencion alta":  C_GREEN,
            "Atencion media": C_YELLOW,
            "Distraccion":    C_RED,
            "Fatiga":         C_RED,
        }
        # Colores por etiqueta ASCII
        label_map = {
            "Atencion alta":  ("Atencion alta",  C_GREEN),
            "Atencion media": ("Atencion media", C_YELLOW),
            "Distraccion":    ("Distraccion",    C_RED),
            "Fatiga":         ("Fatiga",         C_RED),
        }
        for orig_label, secs in cls["state_durations"].items():
            if orig_label == UserState.NO_FACE.value:
                continue
            label_ascii, col = label_map.get(orig_label, (orig_label[:14], C_GRAY_LT))
            time_str = _fmt_time(secs)
            # Barra de proporción
            total = max(1, cls["session_seconds"])
            pct   = int(secs / total * (self._pw - 120))
            cv2.rectangle(panel, (10, y - 10), (10 + pct, y - 2), col, -1)
            cv2.putText(panel, f"{label_ascii[:14]}", (10, y + 6),
                        self._font, 0.38, col, 1, cv2.LINE_AA)
            cv2.putText(panel, time_str, (self._pw - 50, y + 6),
                        self._font, 0.38, C_GRAY_LT, 1, cv2.LINE_AA)
            y += 22

        # Porcentaje global de atención
        cv2.putText(panel,
                    f"Atento: {cls['attention_pct']:.0f}%   Total: {_fmt_time(cls['session_seconds'])}",
                    (10, y + 6), self._font, 0.40, C_WHITE, 1, cv2.LINE_AA)

    def _panel_waiting(self, panel: np.ndarray) -> None:
        cy = self._fh // 2
        cv2.putText(panel, "Esperando rostro...",
                    (14, cy), self._font, 0.48, C_GRAY_LT, 1, cv2.LINE_AA)
        tip = TIPS[UserState.NO_FACE][0]
        cv2.putText(panel, tip, (14, cy + 28),
                    self._font, 0.40, C_GRAY_LT, 1, cv2.LINE_AA)

    def _panel_footer(self, panel: np.ndarray) -> None:
        y = self._fh - 14
        cv2.rectangle(panel, (0, self._fh - 24), (self._pw, self._fh), C_BG_CARD, -1)
        cv2.putText(panel, "Q / ESC para salir",
                    (10, y), self._font, 0.38, C_GRAY_LT, 1, cv2.LINE_AA)


# ════════════════════════════════════════════════════════════
#  HELPERS DE DIBUJO
# ════════════════════════════════════════════════════════════
def _card(img, x1, y1, x2, y2, color) -> None:
    """Rectángulo de fondo con borde sutil."""
    cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
    cv2.rectangle(img, (x1, y1), (x2, y2), C_BORDER, 1)


def _hline(img, y, pw) -> None:
    cv2.line(img, (8, y), (pw - 8, y), C_BORDER, 1)


def _mini_bar(img, label: str, pct: int, x, y, max_x, color) -> None:
    """Mini barra de progreso horizontal con etiqueta."""
    bar_x  = x + 82
    bar_w  = max_x - bar_x
    bar_h  = 9
    filled = max(0, int(bar_w * pct / 100))
    cv2.putText(img, label, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, C_GRAY_LT, 1, cv2.LINE_AA)
    cv2.rectangle(img, (bar_x, y - 9), (bar_x + bar_w, y - 9 + bar_h), C_GRAY_DK, -1)
    if filled > 0:
        cv2.rectangle(img, (bar_x, y - 9), (bar_x + filled, y - 9 + bar_h), color, -1)
    cv2.putText(img, f"{pct}%", (bar_x + bar_w + 4, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, C_GRAY_LT, 1, cv2.LINE_AA)


def _wrap(words: list, max_chars: int) -> tuple:
    """Divide una lista de palabras en dos líneas según max_chars."""
    line1 = ""
    for i, w in enumerate(words):
        if len(line1) + len(w) + 1 <= max_chars:
            line1 = (line1 + " " + w).strip()
        else:
            return line1, " ".join(words[i:])
    return line1, ""


def _score_color(score: int):
    if score >= config.ATTENTION_HIGH:
        return C_GREEN
    if score >= config.ATTENTION_MEDIUM:
        return C_YELLOW
    return C_RED


def _fmt_time(seconds: float) -> str:
    s = int(seconds)
    m, s = divmod(s, 60)
    return f"{m:02d}:{s:02d}"