# Malaria Detection using Convolutional Neural Networks (CNN)

## Project Overview

Malaria is a life-threatening disease caused by *Plasmodium* parasites and remains a major global health challenge. Manual examination of blood smears under a microscope is time-consuming and highly dependent on expert knowledge.

In this project, a CNN model is trained to:

- Learn visual patterns from red blood cell images
- Distinguish infected cells from healthy ones
- Achieve robust classification performance using regularization and normalization techniques

---

## Dataset Used

- **Cell Images for Detecting Malaria**  
  https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria
- A public Kaggle dataset containing **parasitized** and **uninfected** red blood cell images.

---

## Model Architecture

The neural network architecture consists of:

1. Input Layer — 64×64 RGB image  
2. Convolution + MaxPooling layers × 2  
3. Batch Normalization + Dropout after each block  
4. Flatten layer  
5. Fully connected dense layers:
   - 512 neurons
   - 256 neurons  
6. Output layer — 2 neurons with **softmax** activation for binary classification  

**Loss Function:** Categorical Cross-Entropy  
**Optimizer:** Adam  
**Metric:** Accuracy  

### Model Architecture Visualization

<img width="300" height="475" alt="CNN Architecture" src="https://github.com/user-attachments/assets/783c84ce-e533-4ab7-89b6-44620fff4018" />

---

## Design Rationale

- Convolutional layers extract spatial and texture features such as parasite shapes and cell morphology.
- MaxPooling reduces spatial complexity while preserving important features.
- Batch Normalization stabilizes training and improves convergence.
- Dropout (20%) reduces overfitting and improves generalization.
- Softmax output produces class probabilities suitable for categorical cross-entropy loss.

---

## Training & Performance

The model was trained for **10 epochs** using early stopping and learning rate reduction.

Training and validation accuracy increase rapidly, while losses decrease steadily, indicating good convergence and generalization.

### Training Curves

<img width="720" height="280" alt="Training Performance" src="https://github.com/user-attachments/assets/46476d57-5473-4c2f-a38a-5a0c40c89cf5" />

The model achieves a final accuracy of approximately **95–96%** on the test set and successfully distinguishes infected from healthy cells.

---

## Dataset Structure

The dataset must be organized as follows:

```
cell_images/
├── Parasitized/
│ ├── *.png
│ └── ...
└── Uninfected/
  ├── *.png
  └── ...
```

**Labels:**
- `0` → Parasitized
- `1` → Uninfected

---

## Requirements

Main libraries used:

- numpy
- opencv-python
- Pillow
- matplotlib
- scikit-learn
- tensorflow
- keras

---

## How to Run

1. Clone this repository  
2. Download and place the dataset in the required folder structure  
3. Update the dataset path in the script:
   ```python
   image_directory = r'C:\Users\Asal\Desktop\cell_images'
4. Run the script:
   ```python
   python malaria_cnn.py
