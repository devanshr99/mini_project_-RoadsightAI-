# RoadSight AI — Road Sign Recognition & Explainability System

**B.Tech Computer Science Engineering Mini-Project**  
A production-ready deep-learning road sign recognition system featuring multi-architecture classification, transfer learning backbones, and mathematical **Grad-CAM (Gradient-Weighted Class Activation Mapping)** visual explainability over the German Traffic Sign Recognition Benchmark (GTSRB).

---

## System Overview

Autonomous vehicles and driver-assistance systems (ADAS) require both high classification accuracy and interpretable decision-making. **RoadSight AI** combines high-performance convolutional and residual neural networks with real-time explainability heatmaps, verifying that neural decisions are based on actual sign pictograms rather than spurious background cues.

### Key Features

- **Multi-Model Inference & Selection**:
  - **Custom CNN (Baseline)**: 4-layer convolutional architecture engineered specifically for GTSRB (98.92% test accuracy, 3.82 ms latency).
  - **SignNet-Lite**: Ultra-fast edge variant optimized for low latency (1.97 ms inference on CPU).
  - **MobileNetV2**: Transfer learning using inverted residual blocks (Phase B: unfreezing top residual blocks, 98.40% test accuracy).
  - **ResNet18**: Deep residual network with skip connections (Phase B: unfreezing layer4 + classifier, **99.67% test accuracy**, 0.9972 macro F1).
- **Mathematical Grad-CAM Explainability**:
  - Live gradient computation with respect to target convolutional feature maps.
  - Interactive **Split Comparison Slider**, **Side-by-Side View**, and **Jet Colormap** views.
  - Safe PyTorch hook lifecycle management preventing memory leaks.
- **Strict Preprocessing & Shape Verification**:
  - Strict input tensor assertion: `[1, 3, 64, 64]` with GTSRB mean/std normalization.
  - Rejection of invalid, corrupted, or undersized images.
- **Empirical Evaluation & Dataset Explorer**:
  - Quantitative benchmark matrices from real experimental runs (`experiments/experiment_log.csv`).
  - Interactive confusion matrix inspection and learning curves.
  - 43-class GTSRB distribution and split audit (26,640 verified images, 0 duplicates, 0 corrupted).
- **1-Click Test Gallery**: Instant evaluation on curated benchmark test images without needing to upload files manually.
- **100% Local-First & Zero Paid APIs**: Runs completely offline on local CPU or CUDA GPU.

---

## Project Structure

```
mini project final/
├── app.py                     # Flask REST API server & predictor cache
├── requirements.txt           # Python backend dependencies
├── checkpoints/               # Trained PyTorch model weights
│   ├── custom_cnn_best.pth    # Custom CNN weights
│   ├── signnet_lite_best.pth  # SignNet-Lite weights
│   ├── mobilenet_v2_best.pth  # Fine-tuned MobileNetV2 weights
│   └── resnet18_best.pth      # Fine-tuned ResNet18 weights
├── data/                      # GTSRB dataset & taxonomy mappings
│   ├── class_names.json       # 43-class taxonomy
│   ├── dataset_summary.json   # Verified dataset quality audit metrics
│   ├── processed/             # Preprocessed GTSRB images
│   └── raw/                   # Raw GTSRB class folders
├── experiments/               # Empirical experiment records & plots
│   ├── experiment_log.csv     # Hyperparameters and evaluation metrics
│   └── plots/                 # Confusion matrices, curves, latency comparisons
├── src/                       # Deep learning core package
│   ├── models/                # Architecture definitions & factory
│   │   ├── custom_cnn.py      # Custom 4-layer CNN
│   │   ├── mobilenet_v2.py    # Inverted residual MobileNetV2
│   │   ├── resnet18.py        # Deep residual ResNet18
│   │   └── factory.py         # Registry mapping & strict checkpoint loader
│   ├── dataset.py             # Preprocessing & normalization pipeline
│   ├── explain.py             # Grad-CAM engine with safe hook cleanup
│   └── inference.py           # Unified TrafficSignPredictor engine
├── tests/                     # Automated pytest suite (29 tests)
│   ├── test_api.py            # REST endpoints, error cases & prediction tests
│   ├── test_architecture_mapping_and_preprocessing.py
│   ├── test_dataset.py
│   ├── test_explainability.py
│   ├── test_inference.py
│   └── test_model.py
└── frontend/                  # Modern React + Vite + TypeScript frontend
    ├── package.json           # Frontend dependencies (React, Tailwind, Lucide, Recharts)
    ├── vite.config.ts         # Vite configuration with backend proxy
    ├── tailwind.config.js     # Warm neutral design tokens
    └── src/
        ├── App.tsx            # Navigation & router layout
        ├── types/             # TypeScript type definitions
        ├── services/api.ts    # Centralized typed API client
        ├── components/        # Reusable UI components & ImageComparisonSlider
        └── pages/             # 7 dedicated application pages
```

---

## Quick Start & Installation

### Requirements
- **Python**: 3.10+ (tested on Python 3.12, 3.13, 3.14)
- **Node.js**: v18+ (tested on Node v22.19.0)
- **Git**

---

### Step 1: Backend Setup

1. Open a terminal in the project root:
   ```bash
   cd "mini project final"
   ```

2. Activate your virtual environment or install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the automated test suite to verify model checkpoints and pipelines:
   ```bash
   pytest tests/
   ```
   *(All 29 tests will execute and verify models, shapes, Grad-CAM, and REST endpoints).*

4. Start the Flask backend server:
   ```bash
   python app.py
   ```
   *The server will start listening at: `http://127.0.0.1:5000`*

---

### Step 2: Frontend Setup

1. Open a second terminal and navigate to `frontend`:
   ```bash
   cd "mini project final/frontend"
   ```

2. Install dependencies (if not already installed):
   ```bash
   npm install
   ```

3. Launch the development server:
   ```bash
   npm run dev
   ```
   *The application will open at: `http://localhost:5173`*

---

## API Documentation

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | System readiness, device detection (CPU/CUDA), and model availability. |
| `GET` | `/api/models` | List all architectures, parameters, sizes, and checkpoint statuses. |
| `POST` | `/api/predict` | Multipart or base64 image inference with top-3 predictions and Grad-CAM overlay. |
| `GET` | `/api/dataset/summary` | Verified GTSRB dataset audit metrics and split statistics. |
| `GET` | `/api/dataset/samples-gallery` | Curated sample signs for instant one-click testing in the UI. |
| `GET` | `/api/evaluation` | Real benchmark metrics and confusion matrix plots per model. |
| `GET` | `/api/experiments` | Empirical experiment history from `experiment_log.csv`. |
| `GET` | `/api/classes` | 43 GTSRB class taxonomy mapping. |
| `GET` | `/plots/<filename>` | Serves generated evaluation plots and confusion matrices. |

---

## Quantitative Benchmark Summary

Empirically verified on the held-out GTSRB test split ($N = 3,996$ test samples):

| Architecture | Test Accuracy | Macro F1 | Val Accuracy | CPU Latency | Parameters | Model Size |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **ResNet18 (Phase B)** | **99.67%** | **0.9972** | 99.42% | 10.94 ms | 11.20M | 128.28 MB |
| **Custom CNN (Baseline)** | 98.92% | 0.9895 | 99.05% | 3.82 ms | 2.32M | 8.92 MB |
| **SignNet-Lite** | 98.92% | 0.9895 | 99.05% | **1.97 ms** | 2.32M | 8.92 MB |
| **MobileNetV2 (Phase B)** | 98.40% | 0.9829 | 98.57% | 28.53 ms | 2.28M | 26.44 MB |

---

## Academic References

1. Stallkamp, J., Schlipsing, M., Salmen, J., & Igel, C. (2012). *Man vs. computer: Benchmarking machine learning algorithms for traffic sign recognition.* Neural Networks, 32, 323-332.
2. Selvaraju, R. R., Cogswell, M., Das, A., Vedaldi, A., Parikh, D., & Batra, D. (2017). *Grad-CAM: Visual explanations from deep networks via gradient-based localization.* IEEE ICCV, 618-626.
3. Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L. C. (2018). *MobileNetV2: Inverted residuals and linear bottlenecks.* IEEE CVPR, 4510-4520.
4. He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep residual learning for image recognition.* IEEE CVPR, 770-778.

---

## License
Built for B.Tech Computer Science Engineering academic evaluation and research. 100% open-source and local-first.
