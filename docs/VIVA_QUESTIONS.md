# Comprehensive Viva Defense Guide — Road Sign Recognition Project

This document provides concise, technically rigorous, and easily understandable answers to common oral examination (viva voce) questions regarding this road sign recognition project.

---

## 1. Custom CNN Architecture: SignNet-Lite Technical Note

### Q1: Why CNN over Fully Connected Neural Networks (MLPs) or SVMs?
**Answer**:
- **Spatial Grid Topology**: MLPs flatten 2D image matrices into 1D vectors, destroying spatial relationships between neighboring pixels.
- **Parameter Efficiency**: In an MLP, a $64 \times 64 \times 3$ image connected to 128 hidden neurons requires $12,288 \times 128 \approx 1.57 \text{ million}$ weights in the first layer alone. A $3 \times 3$ CNN kernel with 32 filters requires only $3 \times 3 \times 3 \times 32 + 32 = 896$ parameters due to weight sharing.
- **Translation Invariance**: CNN filters slide across the image, recognizing a "Stop" octagon or "Yield" triangle regardless of where it appears in the frame.

---

### Q2: Why Convolution?
**Answer**:
- Convolution applies a spatial filter (kernel) to compute local dot products across spatial windows.
- Early layers act as edge/gradient detectors (Sobel-like filters); middle layers extract geometric contours and curves; deeper layers combine contours into full road sign symbols.

---

### Q3: Why Pooling (MaxPool2D)?
**Answer**:
- **Spatial Downsampling**: MaxPool ($2 \times 2$, stride 2) reduces spatial height and width by half ($64 \times 64 \to 32 \to 16 \to 8 \to 4$), cutting memory compute by $75\%$ per block.
- **Feature Dominance**: Max pooling retains the maximum activation value in each $2 \times 2$ window, preserving dominant feature responses while conferring minor translation and deformation tolerance.

---

### Q4: Why ReLU (Rectified Linear Unit)?
**Answer**:
- **Non-Linearity**: $\text{ReLU}(x) = \max(0, x)$ enables the neural network to learn non-linear decision boundaries.
- **Vanishing Gradient Mitigation**: Unlike Sigmoid or Tanh, ReLU has a constant derivative of $1.0$ for positive inputs, preventing vanishing gradients during backpropagation.
- **Computational Efficiency**: Evaluated via simple thresholding without exponential operations.

---

### Q5: Why Batch Normalization (BatchNorm2D)?
**Answer**:
- **Internal Covariate Shift**: Normalizes feature activations per mini-batch to zero mean and unit variance.
- **Training Stability**: Prevents layer output scales from drifting during training, allowing a higher learning rate ($10^{-3}$) with AdamW without gradient explosion.

---

### Q6: Why Dropout (0.4)?
**Answer**:
- **Regularization**: Randomly zeroes out $40\%$ of neuron activations during training forward passes.
- **Ensemble Effect**: Forces the network to learn redundant, robust feature representations rather than relying on specific co-adapted weights, preventing overfitting on training sign shapes.

---

### Q7: Why CrossEntropyLoss?
**Answer**:
- Measures the divergence between predicted class probability distributions (from Softmax) and target one-hot class vectors.
- Heavily penalizes confident wrong predictions, driving steep gradients to correct classification errors rapidly.

---

### Q8: Why AdamW Optimizer?
**Answer**:
- Combines first-moment momentum (smoothing gradient direction) and second-moment RMSProp (scaling step size per parameter).
- Decouples Weight Decay ($\lambda = 0.01$) from gradient updates, preventing weight magnitude growth more effectively than standard Adam.

---

### Q9: Why $64 \times 64$ Input Size?
**Answer**:
- Raw GTSRB traffic sign images range from $25 \times 25$ to $243 \times 225$ pixels (average $50.9 \times 50.4$).
- Resizing to $64 \times 64$ RGB preserves critical text details (e.g. Speed 30 vs 80) while keeping training time under 2 minutes on CPU.

---

### Q10: Why start with a Custom CNN (`SignNet-Lite`) before Transfer Learning?
**Answer**:
- **Academic & Experimental Justification**:
  1. Establishes a true, lightweight baseline to evaluate parameter trade-offs.
  2. Proves fundamental understanding of network design, receptive fields, and hyperparameter tuning from first principles.
  3. Answers the key viva question: *"Why do we need a complex pre-trained network like MobileNetV2 if a simple 4-block CNN can already achieve high accuracy on this domain?"*

---

## 2. General Machine Learning & Evaluation Q&A

### Q11: Why strictly EXCLUDE Horizontal and Vertical Flips in Data Augmentation?
**Answer**:
- Traffic sign categories are semantically direction-dependent.
- Flipping "Turn Left" (Class 34) horizontally creates a "Turn Right" (Class 33) image while retaining its target label as Class 34. This would inject corrupt ground-truth data into the model.

---

### Q12: Why report Macro F1-Score in addition to Accuracy?
**Answer**:
- The GTSRB dataset exhibits class imbalance (Class 1 has 1,500 images; Class 0 has 150 images; Imbalance Ratio = 10.0).
- Macro F1-score calculates the unweighted harmonic mean of Precision and Recall across all 43 classes equally, revealing if minority classes are being misclassified.

---

## 3. Transfer Learning & Model Optimization Q&A

### Q13: What is Transfer Learning and why use ImageNet Pre-training?
**Answer**:
- **Definition**: Transfer learning leverages feature representations learned by a neural network trained on a large dataset (ImageNet: 1.2M images, 1,000 classes) and repurposes them for a target domain (GTSRB road signs).
- **Why ImageNet?**: Low-level filters (edges, textures, color gradients) in early layers generalize across computer vision tasks, accelerating convergence and reducing data requirements.

### Q14: What is the difference between Backbone Freezing and Fine-Tuning?
**Answer**:
- **Backbone Freezing (Phase A)**: `requires_grad = False` for feature extraction layers; only the classification head ($55\text{K}$ parameters) is trained. Prevents destroying pre-trained weights during early optimization.
- **Fine-Tuning (Phase B)**: `requires_grad = True` for selected upper blocks (e.g. `layer4` in ResNet-18) using a small differential learning rate ($10^{-4}$), adapting high-level features to domain-specific traffic sign symbols.

### Q15: Why select MobileNetV2? What makes it lightweight?
**Answer**:
- **Depthwise Separable Convolutions**: Replaces standard $3 \times 3$ convolution with a 2-step operation (spatial $3 \times 3$ depthwise + $1 \times 1$ pointwise channel mixing), reducing parameter count and compute by $8 \times \text{ to } 9 \times$.
- **Inverted Residual Blocks**: Expands channels internally before bottlenecking, maintaining high feature capacity with only 2.28M total parameters.

### Q16: Why select ResNet-18?
**Answer**:
- **Residual Skip Connections**: $y = F(x) + x$ allows gradients to flow directly through identity shortcuts, solving the vanishing gradient problem in deeper networks.
- **Robustness**: Acts as a classic heavy feature extractor benchmark (11.2M parameters) to measure parameter scaling vs accuracy trade-offs.

### Q17: Why isn't the highest accuracy model automatically the best selection?
**Answer**:
- Real-world deployment requires balancing **Accuracy**, **Inference Latency**, **Memory Footprint**, and **Computational Cost**.
- An 11.2M parameter model yielding $+0.1\%$ accuracy at $3 \times$ higher inference latency and $5 \times$ disk memory is impractical for real-time edge devices (e.g. Raspberry Pi / embedded automotive cameras).

### Q18: How did you prevent Data Leakage during hyperparameter tuning?
**Answer**:
- The 3,996-image held-out Test set was strictly excluded during training and hyperparameter selection.
- All model selection decisions were driven solely by validation set accuracy (`checkpoints/*_best.pth`). Test set evaluation occurred exactly once at the end.
