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

