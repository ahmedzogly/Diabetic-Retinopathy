# Project Progress

## Phase 1: Baseline Architecture (✅ Completed)
- Established project structure.
- Created EfficientNet-B0 baseline classifier.
- Built initial U-Net segmentation scaffolding.
- Integrated basic training loop and data loading.

## Phase 2: Optimization & High Accuracy (✅ Completed)
- Upgraded model architecture to **EfficientNet-B3**.
- Implemented **Ben Graham's Preprocessing** (cropping black borders, lighting normalization via Gaussian Blur overlay).
- Improved class weighting (smoothed inverse frequencies).
- Achieved Validation QWK of **0.8332**.
- Evaluated on test set with significant improvements in Sensitivity for early/moderate stages.

## Phase 3: Explainability & Glaucoma Screening (⏳ In Progress)
- **Goal 1**: High-Resolution Grad-CAM hooked into EfficientNet-B3, overlaid onto the preprocessed image to highlight microaneurysms and lesions clearly.
- **Goal 2**: Implement OpenCV heuristic-based CDR (Cup-to-Disc Ratio) Calculator for Glaucoma screening.
- **Goal 3**: Integrate DR diagnosis + Grad-CAM + Glaucoma Risk (CDR) into the FastAPI backend and web frontend.

## Phase 4: Full Multi-Disease Pipeline & Deployment (🔜 Next)
- Unify the deep learning models (DR classification + UNet segmentation) and OpenCV algorithms into a single production pipeline.
- Dockerize the application.
- Final clinical reporting dashboard.
