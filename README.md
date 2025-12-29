# Malaria_Detection_CNN


## Project Overview

Malaria is a life-threatening disease caused by parasites transmitted through the bites of infected mosquitoes. Early and accurate detection is critical for effective treatment. This project leverages CNNs to analyze blood cell images and automatically classify them.


## Dataset Used:
- Cell Images for Detecting Malaria- https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria 
- a public Kaggle dataset of parastiized and uninfected cell images.


## Model Architecture
The neural network architecture consists of:
1. Input Layer — 64×64 RGB image
2. Convolution + MaxPooling Layers × 2
3. Batch Normalization + Dropout after each block
4. Flatten Layer
5. Fully Connected Dense Layers
- 512 neurons
- 256 neurons
6. Output Layer — 2 neurons with softmax for binary classification

Loss Function: Categorical Cross-Entropy / Optimizer: Adam / Metric: Accuracy

Visualization:

<img width="300" height="475" alt="image (6)" src="https://github.com/user-attachments/assets/783c84ce-e533-4ab7-89b6-44620fff4018" />


## Design Rationale
- Convolutional layers extract spatial and texture features such as parasite shapes and cell morphology.
- MaxPooling reduces spatial complexity while preserving important features.
- Batch Normalization stabilizes training and improves convergence.
- Dropout (20%) reduces overfitting and improves generalization.
- Softmax output produces class probabilities suitable for categorical cross-entropy loss.


## Training & Performance

The model was trained for 10 epochs with early stopping and learning rate reduction.

Train-validation accuracy increases rapidly, and losses decrease steadily, showing good convergence.

Here are the performance plots generated from training:

<img width="720" height="280" alt="14" src="https://github.com/user-attachments/assets/46476d57-5473-4c2f-a38a-5a0c40c89cf5" />

The model achieved high classification accuracy on the test set and successfully distinguishes infected vs. healthy cells.

## Dataset Structure
The dataset must be organized as:

```
cell_images/
├── Parasitized/
│      ├── *.png
│      └── ...
└── Uninfected/
       ├── *.png
       └── ...
```

Required:
- Both folders exist
- Images are .png format
- Correct path provided in your script

## How to Run

1. Clone this repository
2. Download and place the dataset in the required folder structure
3. Update the dataset path in the script:
```
image_directory = r'C:\Users\Asal\Desktop\cell_images'
```
4. Run the script:
```
Malaria-Detection-CNN.py
```

