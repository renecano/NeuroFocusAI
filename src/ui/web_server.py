"""
NeuroFocus AI - Servidor WebSocket
Transmite metricas en tiempo real al navegador web.
Corre en un hilo separado para no bloquear el bucle de camara.
"""

import asyncio
import json
import threading
import time
import base64
import cv2
import numpy as np
import websockets
from websockets.server import serve


class WebServer:
    """
    Servidor WebSocket en hilo daemon.
    - El bucle principal manda datos con .push(data)
    - El navegador puede mandar {"cmd": "stop"} para detener la sesion
    - main.py consulta .stop_requested para saber si debe parar
    """

    def __init__(self, host: str = "localhost", port: int = 8765):
        self._host   = host
        self._port   = port
        self._clients = set()
        self._latest  = {}
        self._loop    = None
        self._thread  = None
        self._lock    = threading.Lock()
        self.stop_requested = False

    # ── Arrancar servidor ────────────────────────────────────
    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        for _ in range(20):
            if self._loop is not None:
                break
            time.sleep(0.1)

    def _run(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._serve())

    async def _serve(self) -> None:
        async with serve(self._handler, self._host, self._port):
            await asyncio.Future()

    async def _handler(self, websocket) -> None:
        self._clients.add(websocket)
        if self._latest:
            try:
                await websocket.send(json.dumps(self._latest, cls=_SafeEncoder))
            except Exception:
                pass
        try:
            async for raw in websocket:
                try:
                    msg = json.loads(raw)
                    if msg.get("cmd") == "stop":
                        self.stop_requested = True
                except Exception:
                    pass
        except Exception:
            pass
        finally:
            self._clients.discard(websocket)

    # ── Enviar datos ─────────────────────────────────────────
    def push(self, payload: dict) -> None:
        with self._lock:
            self._latest = payload
        if self._loop and self._clients:
            asyncio.run_coroutine_threadsafe(
                self._broadcast(json.dumps(payload, cls=_SafeEncoder)), self._loop
            )

    async def _broadcast(self, message: str) -> None:
        dead = set()
        for ws in list(self._clients):
            try:
                await ws.send(message)
            except Exception:
                dead.add(ws)
        self._clients -= dead

    def push_frame(
        self,
        frame:          np.ndarray,
        classification: dict | None,
        eye_metrics:    dict | None,
        head_metrics:   dict | None,
        face_detected:  bool,
        score_history:  list = None,
    ) -> None:
        h, w = frame.shape[:2]
        scale = min(1.0, 480 / w)
        small = cv2.resize(frame, (int(w * scale), int(h * scale)))
        _, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 75])
        frame_b64 = base64.b64encode(buf).decode("utf-8")

        payload = {
            "frame":         frame_b64,
            "face_detected": face_detected,
            "ts":            round(time.time(), 2),
        }

        if face_detected and classification and eye_metrics and head_metrics:
            payload["classification"] = {
                "state_label":     classification["state_label"],
                "score":           classification["score"],
                "attention_pct":   classification["attention_pct"],
                "session_seconds": classification["session_seconds"],
                "state_durations": classification["state_durations"],
            }
            payload["eye"] = {
                "ear":            eye_metrics["ear"],
                "blink_rate":     eye_metrics["blink_rate"],
                "total_blinks":   eye_metrics["total_blinks"],
                "fatigue_signal": bool(eye_metrics["fatigue_signal"]),
                "eyes_closed":    bool(eye_metrics["eyes_closed"]),
            }
            payload["head"] = {
                "yaw":                head_metrics["yaw"],
                "pitch":              head_metrics["pitch"],
                "looking_forward":    bool(head_metrics["looking_forward"]),
                "distraction_signal": bool(head_metrics["distraction_signal"]),
            }

        if score_history is not None:
            payload["score_history"] = score_history[-120:]

        self.push(payload)

    def push_summary(self, summary_html: str) -> None:
        self.push({"type": "summary", "html": summary_html})


# ── JSON encoder seguro para tipos numpy ─────────────────────
class _SafeEncoder(json.JSONEncoder):
    def default(self, obj):
        if type(obj).__name__ == "bool_":
            return bool(obj)
        if type(obj).__name__ in ("int8","int16","int32","int64",
                                   "uint8","uint16","uint32","uint64"):
            return int(obj)
        if type(obj).__name__ in ("float16","float32","float64"):
            return float(obj)
        if hasattr(obj, "tolist"):
            return obj.tolist()
        return super().default(obj)