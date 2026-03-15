# ============================================================
#  NeuroFocus AI — Configuración global del sistema
# ============================================================

# ── Cámara ──────────────────────────────────────────────────
CAMERA_INDEX = 0          # índice de la webcam (0 = cámara por defecto)
FRAME_WIDTH  = 640
FRAME_HEIGHT = 480
FPS_TARGET   = 30

# ── MediaPipe Face Mesh ──────────────────────────────────────
MAX_FACES            = 1
DETECTION_CONFIDENCE = 0.7
TRACKING_CONFIDENCE  = 0.7

# ── Índices de landmarks MediaPipe (Face Mesh 468 puntos) ───
#    Ojo izquierdo (desde la perspectiva del usuario)
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
#    Ojo derecho
RIGHT_EYE = [33,  160, 158, 133, 153, 144]

#    Nariz (punta) y mentón para pose de cabeza
NOSE_TIP   = 1
CHIN       = 152
LEFT_EAR   = 234
RIGHT_EAR  = 454
FOREHEAD   = 10

# ── Parpadeo ─────────────────────────────────────────────────
EAR_THRESHOLD       = 0.21   # Eye Aspect Ratio mínimo para considerar ojo abierto
BLINK_CONSEC_FRAMES = 2      # cuadros consecutivos con EAR bajo = parpadeo
FATIGUE_BLINK_RATE  = 25     # parpadeos/min por encima de este valor → fatiga
CLOSED_EYE_SECONDS  = 1.5    # segundos con ojos cerrados → fatiga inmediata

# ── Orientación de cabeza ────────────────────────────────────
YAW_THRESHOLD   = 25    # grados de giro lateral máximo para "mirando al frente"
PITCH_THRESHOLD = 20    # grados de inclinación vertical
DISTRACTION_SECONDS = 2.0   # segundos fuera de pantalla → distracción

# ── Clasificación de estados ─────────────────────────────────
#    score de atención: 0–100
ATTENTION_HIGH   = 70
ATTENTION_MEDIUM = 40
# por debajo de ATTENTION_MEDIUM → distracción / fatiga

# ── Ventana temporal para métricas ───────────────────────────
METRICS_WINDOW_SECONDS = 60   # ventana deslizante para calcular promedios

# ── Colores BGR para OpenCV ──────────────────────────────────
COLOR_GREEN  = (0,   200,  80)
COLOR_YELLOW = (0,   200, 220)
COLOR_RED    = (0,    60, 220)
COLOR_WHITE  = (255, 255, 255)
COLOR_BLACK  = (0,     0,   0)
COLOR_GRAY   = (160, 160, 160)
COLOR_BG     = (25,   25,  35)   # fondo del panel lateral

# ── Interfaz ─────────────────────────────────────────────────
PANEL_WIDTH      = 280    # ancho del panel de métricas lateral
FONT_SCALE_LARGE = 0.7
FONT_SCALE_SMALL = 0.5
FONT_THICKNESS   = 2