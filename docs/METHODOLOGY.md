# System Methodology & Dataset Audit Specification

## 1. Dataset Audit & Data Quality Findings
- **Dataset Name**: German Traffic Sign Recognition Benchmark (GTSRB) / Local Benchmark Dataset
- **Category Map**: 43 distinct European / GTSRB road sign categories (`00` to `42`).
- **Total Image Count**: 1,075 measured valid image files.
- **Corrupted Image Count**: 0 (validated using PIL `verify()` byte integrity check).
- **Duplicate Image Count**: 0 (measured using exact MD5 file content hash matching).
- **Image Format & Channels**: PNG format, 3-channel RGB.
- **Resolution Distribution**:
  - Minimum Resolution: $48 \times 48$ pixels
  - Maximum Resolution: $127 \times 127$ pixels
  - Target Resized Input: $64 \times 64$ pixels
- **Class Imbalance**: 25 images per class across all 43 categories (Imbalance Ratio = 1.00).

---

## 2. Train / Validation / Test Split & Leakage Prevention
- **Split Breakdown**:
  - **Train**: 752 images (70.0%)
  - **Validation**: 161 images (15.0%)
  - **Test**: 162 images (15.0%)
- **Data Leakage Safeguards**:
  1. Image file paths are split deterministically using global seed `42` **before** any transform or augmentation is applied.
  2. **Augmentation is applied ONLY to the Training DataLoader**. Validation and Test DataLoaders receive strictly deterministic Resize and Normalization transforms.
  3. No test image feature statistics or labels leak into training decisions.

---

## 3. Data Preprocessing & Augmentation Rules

### 3.1 Preprocessing Pipeline (Train, Val, Test, & Inference)
- **Resize**: Resized to $64 \times 64$ pixels using Bilinear interpolation.
- **Tensor Conversion**: `ToTensor()` scales raw 8-bit RGB values from $[0, 255]$ to float tensors in range $[0.0, 1.0]$.
- **Normalization**: Normalized using ImageNet mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`.

### 3.2 Augmentation Strategy (Training DataLoader Only)
- `RandomRotation(degrees=12)`: Simulates minor camera tilt angles during vehicle movement.
- `ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1)`: Simulates daylight, shadow, glare, and weather variations.
- `RandomAffine(degrees=0, translate=(0.08, 0.08))`: Simulates small spatial alignment offsets.
- **EXPLICIT EXCLUSIONS (Viva Point)**:
  - *No Horizontal Flips*: Flipping directional signs (e.g., 'Turn Left', 'Keep Right') alters their semantic target label.
  - *No Vertical Flips*: Traffic signs are vertically oriented; upside-down signs do not naturally occur on public roads.

---

## 4. Reproducibility Assurance
- Global seed setting (`seed = 42`) applied across Python `random`, NumPy `np.random.seed(42)`, and PyTorch `torch.manual_seed(42)`.

---

## 5. Dataset Visualizations
The dataset pipeline automatically exports publication-grade plots into `experiments/plots/`:
1. `class_distribution.png`: Bar chart of image counts per class with mean line overlay.
2. `sample_images.png`: 4x4 grid of representative traffic sign images labeled by class ID and title.
3. `image_dimensions.png`: Scatter plot mapping raw width vs height distribution against the target input size ($64 \times 64$).
