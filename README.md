# NeuroFocusAI

Real-time student attention and emotional state analysis using computer vision. The system detects facial landmarks, eye behavior, and head position to classify student focus states and generate session summaries through a web dashboard.

---

## States Detected

| State | Description |
|---|---|
| 🟢 Focused | Eye gaze stable, head aligned, low blink rate |
| 🟡 Distracted | Gaze deviation or frequent head movement |
| 🔴 Fatigued | High blink frequency, drooping eyelids |
| 🟠 Stressed | Irregular blink patterns, tense facial landmarks |

---

## Features

- Real-time webcam analysis at session level
- Eye movement and blink rate detection
- Head pose estimation (pitch, yaw, roll)
- Multi-state student classification
- AI-generated session summary
- Web dashboard with visual metrics

---

## System Architecture

```
src/
│
├── camera/      → webcam capture and frame pipeline
├── detection/   → face detection and MediaPipe landmark extraction
├── analysis/    → eye analysis, head pose estimation, state classification
├── ui/          → web dashboard and server
├── config.py    → thresholds and system configuration
└── main.py      → main execution entry point
```

---

## Tech Stack

- **Python** — core application logic
- **OpenCV** — webcam capture and frame processing
- **MediaPipe** — facial landmark detection (468 points)
- **HTML/CSS/JS** — web dashboard interface

---

## Getting Started

**Clone the repo:**
```bash
git clone https://github.com/renecano/NeuroFocusAI.git
cd NeuroFocusAI
```

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Run the system:**
```bash
python src/main.py
```

**Launch the dashboard** (separate terminal):
```bash
python src/ui/web_server.py
```

Then open your browser at `http://localhost:5000` (or the port shown in the terminal).

---

## Dashboard

The web dashboard displays:
- Live state classification feed
- Blink rate and gaze stability over time
- Per-session timeline of attention states
- AI-generated summary of the learning session

---

## Project Goal

NeuroFocusAI explores how computer vision can support learning environments by giving educators and students objective data about focus, fatigue, and stress — without invasive hardware.

Developed as part of the ITC engineering program at Tecnológico de Monterrey.

---

## License

MIT License
