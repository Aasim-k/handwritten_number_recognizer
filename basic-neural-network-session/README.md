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
### ⚙️ Optimizer
The model is trained using the **Adam optimizer**, which adapts learning rates per parameter and provides fast convergence suitable for small networks like this one.

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

### 🔑 Key Design Choices
- **Convolutions + BatchNorm + ReLU** → Fast convergence and stable training.  
- **Dropout Layers** → Prevents overfitting even with a small parameter budget.  
- **Compact Dense Layer (10 units)** → Keeps parameter count low.  

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

