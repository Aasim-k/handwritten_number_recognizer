# 🖋️ Handwritten Digit Recognizer (MNIST)

## 📌 Assignment Description
**Objective:**  
Build an MNIST-based model with the following constraints:
- **Parameter Count:** Fewer than **25,000 parameters**  
- **Accuracy:** At least **95% test accuracy in just 1 epoch**

---

## ✅ Results Achieved
- **Total Parameters:** `24,394` (✔ under 25k)  
- **Test Accuracy after 1 Epoch:** `97.02%` (✔ above 95%)  
- **Requirement Status:** ✅ Successfully Achieved  

---
## 🏗️ Model Architecture

The model is a **CNN-based architecture** optimized for MNIST digits.  
Here is the detailed layer summary:

| Layer (type)    | Output Shape    | Param # |
|-----------------|-----------------|---------|
| Conv2d-1        | [-1, 16, 26, 26] | 160     |
| BatchNorm2d-2   | [-1, 16, 26, 26] | 32      |
| ReLU-3          | [-1, 16, 26, 26] | 0       |
| BatchNorm2d-4   | [-1, 16, 26, 26] | 32      |
| Conv2d-5        | [-1, 32, 24, 24] | 4,640   |
| BatchNorm2d-6   | [-1, 32, 24, 24] | 64      |
| ReLU-7          | [-1, 32, 24, 24] | 0       |
| BatchNorm2d-8   | [-1, 32, 24, 24] | 64      |
| MaxPool2d-9     | [-1, 32, 12, 12] | 0       |
| Dropout-10      | [-1, 32, 12, 12] | 0       |
| Conv2d-11       | [-1, 64, 10, 10] | 18,496  |
| BatchNorm2d-12  | [-1, 64, 10, 10] | 128     |
| ReLU-13         | [-1, 64, 10, 10] | 0       |
| BatchNorm2d-14  | [-1, 64, 10, 10] | 128     |
| Dropout-15      | [-1, 64]        | 0       |
| Linear-16       | [-1, 10]        | 650     |

**Total params:** 24,394  
**Trainable params:** 24,394  
**Non-trainable params:** 0  

### 🧩 Model Description

The model is a **lightweight Convolutional Neural Network (CNN)** specifically designed for the MNIST dataset. It balances **compactness (<25k parameters)** with **high accuracy (97%+ in 1 epoch)** through careful use of convolutional blocks, normalization, pooling, dropout, and global average pooling (GAP).

#### 🔹 Architecture Breakdown

1. **Block 1: Feature Extraction (Shallow features)**

   * `Conv2d(1→16, 3×3)` extracts low-level patterns (edges, strokes).
   * `BatchNorm2d` + `ReLU` → stabilizes training and introduces non-linearity.
   * An extra `BatchNorm2d` improves convergence stability.

2. **Block 2: Mid-Level Feature Extraction**

   * `Conv2d(16→32, 3×3)` captures more abstract digit structures.
   * `BatchNorm2d` + `ReLU` ensures stable gradients.
   * `MaxPool2d(2×2)` reduces spatial size (from 24×24 → 12×12), lowering computation.
   * `Dropout(0.1)` provides regularization, preventing overfitting.

3. **Block 3: High-Level Feature Extraction**

   * `Conv2d(32→64, 3×3)` detects complex digit parts.
   * `BatchNorm2d` + `ReLU` ensures robust deeper representations.
   * Another `BatchNorm2d` enhances stability.

4. **Global Average Pooling (GAP)**

   * Reduces feature map (`64×10×10`) to a **compact 64-dim vector**, ensuring full input coverage (RF = 28).
   * Eliminates the need for large fully connected layers → drastically reduces parameters.

5. **Classifier Head**

   * `Dropout(0.05)` before final FC layer → improves generalization.
   * `Linear(64→10)` outputs class logits.
   * `log_softmax` produces normalized log-probabilities for classification.

---

### ⚙️ Training Setup

* **Optimizer:** Adam (adaptive learning, fast convergence)
* **Loss Function:** Negative Log Likelihood (`NLLLoss`)
* **Regularization:** Dropout in both convolutional and classifier stages
* **Parameter Count:** 24,394 (well under 25k limit)

---

## 📊 Training & Evaluation Logs

### **Epoch 1**
- **Training Loss:** `0.1994`  
- **Training Accuracy:** `86.89%`  
- **Test Loss:** `0.1219`  
- **Test Accuracy:** `97.02%` 🎉  

## 📁 Project Structure
handwritten_number_recognizer/<br>
│── basic-neural-network-session/<br>
│├── basic_neural_network.py # Main training & evaluation script<br>
│── README.md # Project documentation<br>

