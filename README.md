<div align="center">

<img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
<img src="https://img.shields.io/badge/InsightFace-ArcFace-6C3483?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Mistral_AI-F7931E?style=for-the-badge"/>
<img src="https://img.shields.io/badge/SQLite-003B57?style=for-the-badge&logo=sqlite&logoColor=white"/>
<img src="https://img.shields.io/badge/ONNX_Runtime-005CED?style=for-the-badge&logo=onnx&logoColor=white"/>

# 🛡️ FaceGuard Intelligence

### Real-Time Facial Recognition & Smart Reporting System

*A production-grade AI system that identifies faces from live webcam or uploaded images,  
generates intelligent security reports, logs all events to a database, and provides  
an interactive web dashboard — powered by InsightFace ArcFace and Mistral AI.*

---

**[View on GitHub](https://github.com/owaisahmed19/FaceGuard-AI)** · **[Report a Bug](https://github.com/owaisahmed19/FaceGuard-AI/issues)** · **[Request a Feature](https://github.com/owaisahmed19/FaceGuard-AI/issues)**

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [System Architecture](#-system-architecture)
- [Agent Pipeline](#-agent-pipeline)
- [Data Flow & Pipeline](#-data-flow--pipeline)
- [Academic Foundation](#-academic-foundation)
- [Dashboard Screenshots](#-dashboard-screenshots)
- [Features](#-features)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Team](#-team)

---

## 🔍 Overview

**FaceGuard Intelligence** is a modular, real-time facial recognition system built with Python. It identifies individuals from webcam feeds or uploaded images against a custom dataset of known people, generates AI-enhanced security reports, and persists all events to a SQLite database for forensic analysis.

### Core Capabilities

| Capability | Technology |
|---|---|
| Face Detection & Embedding | InsightFace ArcFace (buffalo_l, ONNX) |
| Identity Matching | Averaged centroid embeddings + cosine similarity (512-d) |
| Recognition Accuracy | **85%+** with two-stage matching & augmentation |
| Generative Reports | Mistral AI (`mistral-small-latest`) |
| Web Interface | Streamlit Dashboard |
| Event Persistence | SQLite via SQLAlchemy ORM |
| PDF Export | FPDF2 + Mistral AI executive summary |

---

## 🏗️ System Architecture

<div align="center">

![FaceGuard Architecture](pics/Gemini_Generated_Image_dyu39wdyu39wdyu3.png)

*Three-layer architecture: Streamlit UI → Modular Agent Pipeline → Storage & External Services*

</div>

The system is organized into three distinct layers:

**Layer 1 — User Interface**
A Streamlit web dashboard with five tabs: Live Camera, Upload Image, Dataset Manager, Logs, and System Chat. All user interaction flows through this layer.

**Layer 2 — Agent Pipeline**
Five specialized Python agents handle all processing: `DatasetAgent`, `RecognitionAgent`, `ReportAgent`, `ChatAgent`, and `PdfAgent`. Each agent has a single responsibility and communicates through clean interfaces.

**Layer 3 — Storage & External Services**
- **SQLite Database** — persistent event log with full audit trail
- **Face Embeddings Cache** — pre-computed ArcFace embeddings stored as `.pkl`
- **Mistral AI API** — generative summaries and professional report language

---

## 🤖 Agent Pipeline

<div align="center">

![FaceGuard Agents](pics/Gemini_Generated_Image_a8t4hha8t4hha8t4.png)

*Five modular agents, each with a dedicated responsibility in the recognition pipeline*

</div>

### Agent 1 — DatasetAgent
**File:** `agents/dataset_agent.py`

Builds a high-accuracy embedding database from the custom dataset. For each person it:
1. Loads every photo and applies **3× augmentation** (original + horizontal flip + brightness boost)
2. Extracts a 512-d ArcFace embedding from each augmented variant (quality & size filtered)
3. **Averages all embeddings into one L2-normalised centroid** — far more robust than single-photo embeddings
4. Caches centroids + all individual embeddings to `models/face_embeddings.pkl`

The centroid approach is the primary reason accuracy reaches **85%+** — it represents the person's true average appearance rather than one specific photo.

### Agent 2 — RecognitionAgent
**File:** `agents/recognition_agent.py`

The core inference engine. Uses a **two-stage matching pipeline** to achieve 85%+ accuracy:

**Stage 1 — Centroid match:** Compares the detected face embedding against each person's pre-averaged centroid vector using cosine similarity. Fast and highly accurate.

**Stage 2 — Top-K vote (fallback):** If the centroid score falls within the borderline margin, the system compares against all 15 individual stored embeddings and uses majority voting for the final decision — preventing wrong labels on ambiguous frames.

A quality gate rejects blurry, tiny, or partial faces before matching (`det_score ≥ 0.70`, `face size ≥ 60px`).

```python
# Two-stage matching logic
similarity = dot(centroid, embedding) / (norm(centroid) * norm(embedding))
if similarity >= 0.65:
    identity = person_name          # clear centroid match
elif similarity >= 0.60:
    identity = top_k_vote_result    # borderline → majority vote decides
else:
    identity = "Unknown"
```

### Agent 3 — ReportAgent
**File:** `agents/report_agent.py`

Transforms raw recognition results into structured, human-readable reports. Produces both a **text report** and a **JSON payload**. When a Mistral API key is configured, it calls `mistral-small-latest` to produce a professional security alert — e.g., *"Alert: Owais_Ahmed was identified via Live Camera"* — with no raw confidence metrics exposed.

### Agent 4 — ChatAgent
**File:** `agents/chat_agent.py`

An interactive AI assistant embedded in the System Chat tab. Powered by Mistral AI with a strict system prompt, it guides users through the dashboard features — explaining how to add people to the dataset, what confidence scores mean, and how to use each tab. It explicitly avoids discussing source code internals or security configurations.

### Agent 5 — PdfAgent
**File:** `agents/pdf_agent.py`

Generates downloadable PDF reports from the event log. Fetches the last 50 events from the database, sends them to Mistral AI for an executive summary, and assembles a professional FPDF document with a header, AI-generated analysis paragraph, and a full event table including timestamps, identities, and sources.

---

## 🔄 Data Flow & Pipeline

<div align="center">

![Data Flow Pipeline](pics/Gemini_Generated_Image_14zbr314zbr314zb.png)

*Complete 8-stage pipeline: raw input → preprocessing → embedding → matching → reporting → storage → output*

</div>

```
STAGE 1 — INPUT          Webcam (cv2.VideoCapture) or Uploaded Image
STAGE 2 — PREPROCESSING  BGR → RGB, dimension normalization, largest face selection
STAGE 3 — DETECTION      InsightFace detects all face regions in the frame
STAGE 4 — EMBEDDING      ArcFace extracts 512-d normalized vector per face
STAGE 5 — MATCHING       Centroid cosine similarity (threshold 0.65) + Top-K vote fallback
STAGE 6 — REPORTING      ReportAgent: structured JSON + text + optional Mistral alert
STAGE 7 — STORAGE        SQLite logs: timestamp, name, confidence, source, report
STAGE 8 — OUTPUT         Annotated image + dashboard report + downloadable PDF
```

> **Live Camera Optimization:** Only every 5th frame passes through the full inference pipeline — reducing CPU load 5× while keeping the video display smooth. Events are only written to the database when the set of detected identities changes, avoiding redundant writes.

---

## 🎓 Academic Foundation

<div align="center">

![Course to Project](pics/Gemini_Generated_Image_j5kntaj5kntaj5kn.png)

*FaceGuard was built by applying concepts from the [cours_ia](https://github.com/cguyeux/cours_ia) curriculum by Prof. Cguyeux*

</div>

FaceGuard directly applies techniques taught in the **cours_ia** university AI curriculum:

| Course Module | Concept Learned | Applied in FaceGuard |
|---|---|---|
| Generative TP1 — LangChain + Mistral | LLM text generation & prompt engineering | `ReportAgent` + `ChatAgent` |
| Generative TP2 — Pydantic structured outputs | Reliable schema-constrained LLM outputs | JSON report structure |
| Generative TP4 — LangChain Agents | Agent orchestration & tool integration | Modular 5-agent architecture |
| Predictive TP2bis — PCA / embeddings | High-dimensional vector space reasoning | ArcFace 512-d face embeddings |
| Predictive TP3 — Supervised classification | Threshold-based decision making | Cosine similarity ≥ 0.65 + Top-K vote matching |
| Predictive TP1 — pandas / data pipelines | Data management & feature persistence | Dataset management + SQLite logging |

---

## 📸 Dashboard Screenshots

### Main Dashboard — Live Camera Tab

<div align="center">

![Dashboard Home](pics/Screenshots/Screenshot%202026-04-05%20163444.png)

*FaceGuard Intelligence dashboard — five navigation tabs: Live Camera, Upload Image, Dataset Manager, Logs, System Chat*

</div>

---

### Live Recognition — Known Identity Detected

<div align="center">

![Live Recognition](pics/Screenshots/Screenshot%202026-04-05%20162846.png)

*Real-time recognition: green bounding box with name label confirms identity (Owais_Ahmed)*

</div>

---

### Image Upload — Group Photo with Mixed Results

<div align="center">

![Group Photo Upload](pics/Screenshots/Screenshot%202026-04-05%20162956.png)

*Upload mode: one known identity (green box), multiple unknowns (red boxes), Agent Report displayed alongside*

</div>

---

### Image Upload — Celebrity Recognition

<div align="center">

![Celebrity Recognition](pics/Screenshots/Screenshot%202026-04-05%20163050.png)

*Angelina Jolie identified among two unknown individuals — AI alert: "Angelina Jolie was detected among two unknown individuals"*

</div>

---

### Recognition Event Logs & AI PDF Export

<div align="center">

![Event Logs](pics/Screenshots/Screenshot%202026-04-05%20164050.png)

*Full audit log of all recognition events with timestamps and one-click AI-powered PDF report generation*

</div>

---

### Dataset Embeddings Manager

<div align="center">

![Dataset Manager](pics/Screenshots/Screenshot%202026-04-05%20164110.png)

*Rebuild embeddings after adding new people to the dataset — displays current known identities count*

</div>

---

### System Chat — AI Assistant

<div align="center">

![System Chat](pics/Screenshots/Screenshot%202026-04-05%20164219.png)

*Interactive Mistral-powered assistant guides users through all dashboard features in natural language*

</div>

---

## ✨ Features

- **85%+ Recognition Accuracy** — Averaged centroid embeddings + 3× augmentation + two-stage matching
- **Live Camera Recognition** — Real-time identity detection at 30 FPS with 5× frame-skip optimization
- **Image Upload Analysis** — One-shot recognition on any JPG, JPEG, PNG, or WEBP image
- **Dataset Manager** — Add new people by placing image folders in `data/people_dataset/` then rebuilding
- **Recognition Event Logs** — Full SQLite audit trail with expandable event details
- **AI PDF Reports** — Downloadable PDF with Mistral-generated executive summary and full event table
- **System Chat** — Embedded Mistral AI assistant for real-time user guidance
- **No GPU Required** — Runs entirely on CPU via ONNX runtime
- **Quality Gate** — Blurry, tiny, and partial faces are automatically rejected before matching

---

## 🛠️ Installation

### Prerequisites

- Python 3.10+
- Webcam (for live camera mode)
- Mistral AI API key (optional — enables AI-enhanced reports, chat, and PDF summaries)

### Steps

**1. Clone the repository**
```bash
git clone https://github.com/owaisahmed19/FaceGuard-AI.git
cd FaceGuard-AI
```

**2. Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Download the InsightFace ONNX model**
```bash
python download_models.py
```

**5. Configure environment variables**
```bash
cp .env.example .env
# Open .env and set your MISTRAL_API_KEY
```

**6. Add people to the dataset**
```
data/
└── people_dataset/
    ├── Owais_Ahmed/
    │   ├── photo1.jpg
    │   └── photo2.jpg
    └── Muhammad_Shiraz/
        ├── photo1.jpg
        └── photo2.jpg
```

**7. Launch the dashboard**
```bash
streamlit run dashboard/app.py
```

Open **http://localhost:8501** in your browser.

---

## 🖥️ Usage

| Tab | What to Do |
|---|---|
| **Live Camera** | Click "Start Camera" to begin real-time recognition |
| **Upload Image** | Drag and drop any image to analyze all faces |
| **Dataset Manager** | After adding person folders, click "Rebuild Dataset Embeddings" |
| **Logs** | Browse all past events; click "Generate AI PDF Report" to download |
| **System Chat** | Ask the AI assistant anything about using the system |

### Troubleshooting

| Issue | Solution |
|---|---|
| Empty dataset warning on boot | Ensure person folders exist in `data/people_dataset/` and rebuild embeddings |
| Webcam cannot open | Close any app using the camera (Zoom, Teams), check OS camera permissions |
| Face labeled "Unknown" | Add more reference photos per person and click **Rebuild Embeddings** — the more photos, the better the centroid |
| Slow performance | Ensure you are using CPU mode (ctx_id=-1 in DatasetAgent), close other heavy processes |

---

## 📁 Project Structure

```
faceguard/
├── agents/
│   ├── chat_agent.py          # Mistral-powered AI assistant
│   ├── dataset_agent.py       # Face embedding generation & caching
│   ├── recognition_agent.py   # Face detection & identity matching
│   ├── report_agent.py        # Text/JSON report generation
│   └── pdf_agent.py           # PDF report export with AI summary
├── config/
│   └── settings.py            # Paths, thresholds, and constants
├── dashboard/
│   └── app.py                 # Streamlit web interface (main entry point)
├── utils/
│   └── database.py            # SQLAlchemy ORM models and helpers
├── data/
│   └── people_dataset/        # Custom dataset — one folder per person
├── models/
│   └── face_embeddings.pkl    # Cached ArcFace embeddings (auto-generated)
├── logs/
│   └── faceguard.db           # SQLite event database (auto-generated)
├── pics/
│   └── Screenshots/           # Dashboard screenshots
├── notebooks/                 # Jupyter notebooks for research
├── download_models.py         # InsightFace model downloader
├── requirements.txt
└── .env.example
```

---

## 📦 Dependencies

```
streamlit>=1.30.0        # Web UI framework
insightface==0.7.3       # Face detection & ArcFace embedding model
onnxruntime              # CPU-based inference engine for ONNX models
opencv-python>=4.8.0     # Webcam capture and image processing
numpy>=1.24.0            # Numerical computing
SQLAlchemy>=2.0.25       # ORM database abstraction layer
Pillow>=10.0.0           # Image I/O handling
loguru>=0.7.2            # Structured logging
python-dotenv>=1.0.0     # .env environment variable loading
requests                 # Mistral AI HTTP client
fpdf2                    # PDF document generation
```

---

## 👥 Team

<div align="center">

| | Name | Contributions |
|---|---|---|
| 👤 | **Owais Ahmed Khan** | AI pipeline · InsightFace integration · Agent architecture · Recognition engine |
| 👤 | **Muhammad Shiraz** | Streamlit dashboard · Database design · Report & PDF systems · System Chat |

*MSc Machine Learning — UFR STGI · Second Semester Project · 2026*

*Built upon the [cours_ia](https://github.com/cguyeux/cours_ia) curriculum by Prof. Cguyeux*

</div>

---

## 📄 License

This project was developed as part of the Master's program in Machine Learning at UFR STGI. All rights reserved by the authors.

---

<div align="center">

Built with dedication by **Owais Ahmed Khan** & **Muhammad Shiraz**

*UFR STGI · Machine Learning · 2026*

</div>
