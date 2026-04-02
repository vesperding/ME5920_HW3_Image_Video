# ME5920 HW3: Image and Video Classification (UCF11)

This repository contains the implementation and analysis for Homework 3 of ME5920, focusing on video classification using the UCF11 dataset.

---

## Project Structure

- `hw3_ucf11_pipeline.py`  
  Main training and evaluation pipeline for all tasks.

- `posthoc_visualization.py`  
  Script for generating all post-hoc visualizations from saved outputs.

- `outputs/`  
  Contains all experiment results, including:
  - CSV logs
  - Performance metrics
  - Visualization figures

- `splits/`  
  Dataset splits (train / val / test)

- `HW3_ME5920.ipynb`  
  Colab notebook containing full experimental workflow and execution

---

## Tasks Overview

### Task 1: Dataset Split Visualization
- Visualizes class distribution across train, validation, and test sets

### Task 2: Frame Sampling Strategy (2D CNN)
- Varies number of sampled frames
- Studies trade-off between performance and efficiency

### Task 3: Temporal Crops (3D CNN)
- Uses multiple temporal crops
- Improves temporal modeling

### Task 4: Sequence Modeling (RNN / LSTM)
- Uses sequence-based representation
- Explores number of sequences and repetition strategies

---

## Key Visualizations

All figures are stored in:

