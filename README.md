# Tumor Detection AI — End-to-End Brain & Breast Tumor Detection with XAI

An end-to-end deep-learning project that classifies brain MRI and breast scans and explains each prediction with **Grad-CAM** heatmaps. It includes the training experiments, a FastAPI inference backend and a lightweight web UI.

**Live demo:** <https://end-2-end-tumor-detection-production.up.railway.app/>
(hosted on Railway's free credit, so it may be offline or take about a minute to wake up on the first request.)

## Features

- **5-class classification** — `No Tumor`, `Tumor` (brain) and `Benign`, `Malignant`, `Normal` (breast).
- **Explainable AI** — Grad-CAM overlays highlight the regions that drove the prediction.
- **Custom `TrigConv2D` layer** — convolution with fixed sine/cosine kernels (see `backend/models/loader.py`).
- **REST API** — `/predict` for classification, `/gradcam` for the heatmap overlay.
- **Web UI** — upload a scan and view the result in the browser.
- **Lazy model loading** — the trained model is downloaded from the Hugging Face Hub on the first request.
- **Daily rotating logs** — kept for 30 days.

## Models

Three models were trained and compared in `experiments/trignometric-cnn.ipynb`:

| Model | Description |
|-------|-------------|
| Baseline CNN | 3 × Conv2D + MaxPool, Dense(128), Dropout(0.5) |
| ResNet | Transfer-learning model |
| Trig CNN | CNN using `TrigConv2D` trigonometric kernels (**served by the API**) |

Input size is 128 × 128 RGB, pixel values scaled to [0, 1]. The served model is hosted at
[`Sauravyadav7721/Brain-Breast-Tumor`](https://huggingface.co/Sauravyadav7721/Brain-Breast-Tumor) (`trig_model.keras`).

**Dataset:** the Kaggle "Brain & Breast Tumor" dataset (`khaledhellmy/brain-breast-tumor`). Classes are oversampled to balance them, then split 80/20 (`random_state=42`).

## Project Structure

```
.
├── backend/
│   ├── config.py          # Paths, HF repo, image size
│   ├── main.py            # FastAPI app (/, /predict, /gradcam)
│   ├── models/loader.py   # TrigConv2D layer + model loader
│   └── utils/
│       ├── preprocess.py  # Image preprocessing
│       ├── predict.py     # Inference + class labels
│       ├── gradcam.py     # Grad-CAM heatmap + overlay
│       ├── service.py     # Service layer (lazy model, pipeline)
│       └── logger.py      # Daily rotating logger
├── frontend/              # index.html, style.css, app.js
├── experiments/           # Training notebooks
├── checkpoints/           # Training checkpoints (git-ignored contents)
├── models/                # Exported/trained models (git-ignored contents)
├── logs/                  # Training/run logs (git-ignored contents)
├── artifacts/             # Saved .h5/.keras models (git-ignored)
├── run.py                 # Server entry point (reads the PORT env var)
├── Dockerfile             # Container build for deployment
├── .dockerignore
├── setup.py
└── requirements.txt
```

## Getting Started

### Prerequisites

- Python 3.10+
- Internet access on first run (to download the model from Hugging Face)

### Installation

```bash
git clone https://github.com/sauravyadav7721/END-2-END-Tumor-Detection.git
cd "END 2 END Tumor Detection with XAI"

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### Run the server

```bash
python run.py
```

The app is served at <http://localhost:8000>. Interactive API docs are at <http://localhost:8000/docs>.
The port defaults to 8000 and can be changed with the `PORT` environment variable.

### Run with Docker

```bash
docker build -t tumor-detection-ai .
docker run -p 8000:8000 tumor-detection-ai
```

## Deployment (Railway)

The project is deployed on [Railway](https://railway.com) from the `Dockerfile`:

1. Push the repo to GitHub.
2. In Railway: **New Project → Deploy from GitHub repo** and select this repo.
3. Railway builds the image and sets the `PORT` variable, which `run.py` reads automatically.
4. In the service **Settings → Networking**, click **Generate Domain** to get a public URL.

The first prediction after a restart is slow because the model is downloaded from the Hugging Face Hub and loaded into memory.
`requirements.txt` uses `tensorflow-cpu` to keep the image and memory usage small.

## API

| Method | Endpoint | Body | Response |
|--------|----------|------|----------|
| `GET` | `/` | — | Web UI |
| `POST` | `/predict` | `multipart/form-data`, field `file` (image) | JSON: `filename`, `prediction`, `confidence`, `all_probs` |
| `POST` | `/gradcam` | `multipart/form-data`, field `file` (image) | PNG Grad-CAM overlay |

Example:

```bash
curl -X POST http://localhost:8000/predict -F "file=@scan.jpg"
```

```json
{
  "filename": "scan.jpg",
  "prediction": "Tumor",
  "confidence": "97.42%",
  "all_probs": { "No Tumor": 0.01, "Tumor": 0.97, "Benign": 0.0, "Malignant": 0.01, "Normal": 0.01 }
}
```

## Training

Training lives in `experiments/trignometric-cnn.ipynb` (written for Kaggle, so dataset paths start with `/kaggle/input/...`). To rerun it locally, adjust the paths. Put checkpoints in `checkpoints/`, exported models in `models/` and run logs in `logs/`. Their contents are git-ignored.

## Tech Stack

TensorFlow / Keras · OpenCV · NumPy · FastAPI · Uvicorn · Hugging Face Hub · Vanilla JS frontend · Docker · Railway

## Disclaimer

This project is for research and educational purposes only. It is **not** a medical device and must not be used for clinical diagnosis.

## Author

Saurav Yadav
