<div align="center">
  <h1>👁️ Diabetic Retinopathy & Glaucoma Detection AI</h1>
  <p><i>A complete, Dockerized FastAPI application for automated screening of fundus images</i></p>
  
  <p>
    <img src="https://img.shields.io/badge/Python-3.10-blue.svg" alt="Python Version">
    <img src="https://img.shields.io/badge/PyTorch-2.0.1-EE4C2C.svg?logo=pytorch" alt="PyTorch">
    <img src="https://img.shields.io/badge/FastAPI-0.103.1-009688.svg?logo=fastapi" alt="FastAPI">
    <img src="https://img.shields.io/badge/Docker-Ready-2496ED.svg?logo=docker" alt="Docker">
    <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  </p>
</div>

<br>

This repository contains a full-stack, production-ready AI pipeline designed to assist ophthalmologists in diagnosing **Diabetic Retinopathy (DR)** and screening for **Glaucoma**. It features a robust deep learning classifier, explainable AI (XAI) overlays, and a fallback-secured computer vision algorithm for optic disc analysis.

---

## ✨ Key Features

*   🧠 **Deep Learning Diagnosis:** Powered by **EfficientNet-B3** trained with Transfer Learning on the APTOS dataset. Achieved a highly robust **QWK of 0.88, Accuracy of 77%, and Sensitivity of 67%**.
*   🔍 **Explainable AI (XAI):** Generates high-resolution **Grad-CAM heatmaps** overlaid directly onto the preprocessed fundus images, highlighting microaneurysms, hemorrhages, and exudates.
*   👁️ **Glaucoma Screening:** Features a custom OpenCV-based heuristic pipeline for Cup-to-Disc Ratio (vCDR) calculation. Utilizes Red/Green channel extraction for contour detection, equipped with robust failure-fallback mechanisms to prevent clinical anomalies.
*   🐳 **Production Ready:** Fully containerized using **Docker** with a lightweight PyTorch CPU build, making it extremely fast, portable, and easy to deploy on any server.

## 🔬 Dataset

This project was trained and evaluated on the **APTOS 2019 Blindness Detection Dataset**.
🔗 **[Dataset Link (Kaggle)](https://www.kaggle.com/datasets/mariaherrerot/aptos2019)**

---

## 🔬 Medical Preprocessing Pipeline

To ensure the deep learning model receives the highest quality input, all images undergo a strict medical preprocessing pipeline before inference:
1.  **Dark Border Cropping:** Automatically detects and removes uninformative black borders from the raw fundus photography.
2.  **Ben Graham's Method:** Applies a weighted Gaussian Blur overlay to normalize lighting conditions and vastly improve the visibility of blood vessels and retinal lesions.

---

## 🛠️ Tech Stack

*   **Deep Learning:** PyTorch, Torchvision, PyTorch Grad-CAM
*   **Computer Vision:** OpenCV, Albumentations, PIL
*   **Backend & API:** FastAPI, Uvicorn, Pydantic
*   **Frontend:** Vanilla JavaScript, HTML5, CSS3
*   **Deployment:** Docker

---

## 🚀 Quick Start (Docker)

The fastest and most reliable way to run this application is via Docker. You don't need to manually configure virtual environments or PyTorch dependencies.

**1. Clone the repository:**
```bash
git clone https://github.com/your-username/diabetic-retinopathy-ai.git
cd diabetic-retinopathy-ai/diabetic_retinopathy_project
```

**2. Build the Docker Image:**
```bash
docker build -t diabetic-retinopathy-api .
```

**3. Run the Container:**
```bash
docker run -d -p 8000:8000 --name dr-api diabetic-retinopathy-api
```

The web application and API will now be live at: **[http://localhost:8000](http://localhost:8000)**

---

## 📁 Project Structure

```text
diabetic_retinopathy_project/
├── app/                  # FastAPI Application & Frontend
│   ├── static/           # JS, CSS, and Demo Images
│   ├── templates/        # HTML Views
│   └── main.py           # FastAPI Router & Server
├── src/                  # Core Machine Learning & CV Logic
│   ├── classifier_model.py # EfficientNet-B3 Architecture
│   ├── explain.py        # Grad-CAM implementation
│   ├── inference.py      # Production Inference Engine
│   └── cdr_calculator.py # OpenCV Glaucoma Heuristics
├── models/               # Trained Weights (best_model.pth)
├── scripts/              # Training & Data Prep Utilities
├── Dockerfile            # Lightweight Production Docker Build
├── .dockerignore         # Docker Context Filters
└── requirements.txt      # PyTorch CPU & Backend Dependencies
```

---

## 🎓 Team & Acknowledgements

This system was developed as a graduation project under the honorable supervision of **Dr. Mohamed Elhadad**.

**Development Team:**
*   **Ahmed Shehta Zoghli**
*   **Eslam Tag Elser**
*   **Mohamed Hassan Ahmed**
*   **Osama Mohamed Kamel**
*   **Ahmed Zain Elabiden**


> *"Dedicated to improving accessible healthcare technology through open-source artificial intelligence."*
