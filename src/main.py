"""
NeuroFocus AI - Punto de entrada principal

Modos de uso:
    python src/main.py            -> ventana OpenCV
    python src/main.py --web      -> panel web (recomendado)
    python src/main.py --both     -> ambos simultaneamente
"""

import cv2
import sys
import os
import time
import argparse
import webbrowser
import threading

sys.path.insert(0, os.path.dirname(__file__))

from camera.capture             import VideoCapture
from detection.face_detector    import FaceDetector
from analysis.eye_analyzer      import EyeAnalyzer
from analysis.head_pose         import HeadPoseAnalyzer
from analysis.state_classifier  import StateClassifier, UserState
from analysis.summary_generator import generate_summary
from ui.dashboard               import Dashboard
from ui.web_server              import WebServer
import config


def parse_args():
    p = argparse.ArgumentParser(description="NeuroFocus AI")
    p.add_argument("--web",  action="store_true", help="Activar panel web")
    p.add_argument("--both", action="store_true", help="Ventana OpenCV + panel web")
    p.add_argument("--no-ai-summary", action="store_true",
                   help="Desactivar resumen con IA al finalizar")
    return p.parse_args()


def main():
    args = parse_args()
    use_web = args.web or args.both
    use_cv  = not args.web or args.both
    use_ai_summary = not args.no_ai_summary

    print("=" * 55)
    print("  NeuroFocus AI")
    if use_web:
        print("  Panel web  ->  abre  src/ui/dashboard.html")
    if use_cv:
        print("  Ventana OpenCV activa  (Q / ESC para salir)")
    print("=" * 55)

    try:
        cam = VideoCapture(config.CAMERA_INDEX)
    except RuntimeError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    detector    = FaceDetector()
    eye_ana     = EyeAnalyzer()
    head_ana    = HeadPoseAnalyzer()
    classifier  = StateClassifier()
    dashboard   = Dashboard(cam.width, cam.height) if use_cv else None
    web_server  = None

    if use_web:
        web_server = WebServer()
        web_server.start()
        print(f"  WebSocket escuchando en ws://localhost:8765")
        html_path = os.path.join(os.path.dirname(__file__), "ui", "dashboard.html")
        abs_path  = os.path.abspath(html_path).replace(os.sep, '/')
        threading.Timer(1.2, lambda: webbrowser.open(f"file:///{abs_path}")).start()

    if use_cv:
        cv2.namedWindow("NeuroFocus AI", cv2.WINDOW_NORMAL)

    print(f"  Camara: {cam.width}x{cam.height}")

    classification     = None
    eye_metrics        = None
    head_metrics       = None
    frame_count        = 0
    fatigue_events     = 0
    distraction_events = 0
    last_fatigue       = False
    last_distract      = False
    session_start_ts   = time.time()

    score_history     = []
    SCORE_SAMPLE_EVERY = 2.0
    last_score_sample  = time.time()

    try:
        while True:
            ok, frame = cam.read()
            if not ok:
                break

            frame_count += 1
            result = detector.process(frame)

            if result.detected:
                detector.draw_landmarks(frame, result)
                eye_metrics    = eye_ana.update(result)
                head_metrics   = head_ana.update(result)
                classification = classifier.classify(eye_metrics, head_metrics)

                if eye_metrics["fatigue_signal"] and not last_fatigue:
                    fatigue_events += 1
                if head_metrics["distraction_signal"] and not last_distract:
                    distraction_events += 1
                last_fatigue  = eye_metrics["fatigue_signal"]
                last_distract = head_metrics["distraction_signal"]

                now = time.time()
                if now - last_score_sample >= SCORE_SAMPLE_EVERY:
                    score_history.append({
                        "t": round(now - session_start_ts, 1),
                        "s": classification["score"],
                    })
                    last_score_sample = now
            else:
                eye_ana.reset()
                head_ana.reset()
                classification = None
                eye_metrics    = None
                head_metrics   = None
                last_fatigue   = False
                last_distract  = False

            if use_cv and dashboard:
                canvas = dashboard.render(
                    frame, classification, eye_metrics, head_metrics,
                    face_detected=result.detected,
                )
                cv2.imshow("NeuroFocus AI", canvas)

            if use_web and web_server and frame_count % 3 == 0:
                web_server.push_frame(
                    frame, classification, eye_metrics, head_metrics,
                    face_detected=result.detected,
                    score_history=score_history,
                )

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if use_web and web_server and web_server.stop_requested:
                print("  [WEB] Sesion finalizada desde el navegador.")
                break

    except KeyboardInterrupt:
        pass
    finally:
        summary_data  = classifier.summary()
        session_secs  = summary_data["duracion_total_seg"]
        score_avg     = 0
        attention_pct = 0
        total_blinks  = eye_metrics["total_blinks"] if eye_metrics else 0

        if eye_metrics and head_metrics:
            cls_data      = classifier.classify(eye_metrics, head_metrics)
            score_avg     = cls_data["score"]
            attention_pct = cls_data["attention_pct"]

        if score_history:
            score_avg = int(sum(p["s"] for p in score_history) / len(score_history))

        print("\n" + "=" * 55)
        print("  Resumen de sesion")
        print("=" * 55)
        m, s = divmod(int(session_secs), 60)
        print(f"  Duracion      : {m:02d}:{s:02d}")
        print(f"  Score avg     : {score_avg}%")
        print(f"  Tiempo atento : {attention_pct:.0f}%")
        for estado, dur in summary_data["estados"].items():
            m2, s2 = divmod(int(dur), 60)
            print(f"  {estado:<18}: {m2:02d}:{s2:02d}")

        # ── Resumen con IA ───────────────────────────────────
        if use_ai_summary and session_secs > 10:
            print("\n  Generando analisis con IA...")
            try:
                ai_data = generate_summary({
                    "duracion_seg":       session_secs,
                    "score_promedio":     score_avg,
                    "attention_pct":      attention_pct,
                    "total_blinks":       total_blinks,
                    "fatigue_events":     fatigue_events,
                    "distraction_events": distraction_events,
                    "tiempo_estados":     summary_data["estados"],
                })
                ai_data["score"] = score_avg

                if use_web and web_server:
                    web_server.push_summary(_build_summary_html(ai_data))
                    print("  Resumen enviado al navegador.")
                    time.sleep(2)

                print(f"  {ai_data.get('titulo', '')}")

            except Exception as e:
                print(f"  [WARN] No se pudo generar resumen IA: {e}")

        cam.release()
        detector.close()
        print("\n  Sistema finalizado.")


def _build_summary_html(data: dict) -> str:
    score = data.get("score", 0)
    col   = "#22d68a" if score >= 70 else "#f5c842" if score >= 40 else "#f05a7e"

    fortalezas = "".join(
        f'<div class="summary-item"><span class="si-icon">&#x2705;</span><span>{f}</span></div>'
        for f in data.get("fortalezas", [])
    )
    mejoras = "".join(
        f'<div class="summary-item"><span class="si-icon">&#x1F4CC;</span><span>{m}</span></div>'
        for m in data.get("areas_mejora", [])
    )
    recs = "".join(
        f'<div class="summary-item">'
        f'<span class="si-icon">{r["icono"]}</span>'
        f'<span><strong>{r["titulo"]}</strong> &mdash; {r["descripcion"]}</span>'
        f'</div>'
        for r in data.get("recomendaciones", [])
    )

    return f"""
    <div class="summary-title">{data.get("titulo", "Resumen de sesion")}</div>
    <div class="summary-subtitle">{data.get("resumen_general", "")}</div>
    <div class="summary-score-row">
      <div class="summary-big-score" style="color:{col}">{score}%</div>
      <div>
        <div style="font-weight:700;font-size:1rem">{data.get("puntuacion_label","")}</div>
        <div style="color:var(--muted);font-size:.85rem;margin-top:4px">Score promedio de la sesion</div>
      </div>
    </div>
    <div class="summary-section">
      <h3>&#x1F4AA; Lo que hiciste bien</h3>
      <div class="summary-items">{fortalezas}</div>
    </div>
    <div class="summary-section">
      <h3>&#x1F3AF; Areas de mejora</h3>
      <div class="summary-items">{mejoras}</div>
    </div>
    <div class="summary-section">
      <h3>&#x1F680; Recomendaciones</h3>
      <div class="summary-items">{recs}</div>
    </div>
    <p class="summary-msg">"{data.get("mensaje_final","")}"</p>
    <button class="btn-close"
      onclick="document.getElementById('summary-overlay').style.display='none'">
      Cerrar y volver
    </button>
    """


if __name__ == "__main__":
    main()