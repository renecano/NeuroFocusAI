# NeuroFocusAI

NeuroFocusAI is an artificial intelligence system that analyzes student attention and emotional state using computer vision.

The system uses a webcam to detect facial landmarks, eye behavior and head position to estimate whether a student is:

- Focused
- Distracted
- Fatigued
- Stressed

The results are presented through a visual dashboard that summarizes the learning session.

---

## Features

- Real-time webcam analysis
- Eye movement and blink detection
- Head pose estimation
- Student state classification
- Session summary generation using AI
- Web dashboard for visualization

---

## System Architecture

The project is modular and organized into different components.
```
src/
│
├── camera/      → webcam capture
├── detection/   → face detection and landmarks
├── analysis/    → eye analysis, head pose, classification
├── ui/          → dashboard and web interface
├── config.py    → configuration
└── main.py      → main execution script
```

---

## Technologies Used

- Python
- OpenCV
- MediaPipe
- Computer Vision
- HTML Dashboard

---

## How to Run

Clone the repository:
```bash
git clone https://github.com/renecano/NeuroFocusAI.git
cd NeuroFocusAI
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Run the system:
```bash
python src/main.py
```

---

## Dashboard

The system includes a web dashboard that displays the analysis results and session summaries.
```bash
python src/ui/web_server.py
```

Then open the dashboard in your browser.

---

## Project Goal

This project explores how artificial intelligence and computer vision can be used to improve learning environments by detecting student fatigue, stress and lack of focus.

---

## License

MIT License
