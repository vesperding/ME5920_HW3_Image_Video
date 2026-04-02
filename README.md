# ME5920 HW3: Image and Video Classification (UCF11)

This repository contains the implementation and results for Homework 3 (ME5920), focusing on video classification on the UCF11 dataset.

---

## Repository Structure

- `hw3_ucf11_pipeline.py`  
  Main script for training, evaluation, and result generation for all tasks.

- `outputs/`  
  All experiment outputs, including:
  - performance metrics (CSV)
  - confusion matrices
  - training curves
  - comparison plots

- `outputs/figures/`  
  Consolidated visualization results used for analysis and presentation.

- `splits/`  
  Train / validation / test splits based on video folders.

- `HW3_ME5920.ipynb`  
  Colab notebook with full experimental workflow.

---

## Tasks

### Task 1: Dataset Split
Videos are split by folder index:
- Test: folders 20–25  
- Validation: folders 17–19  
- Training: remaining folders  

A visualization of class distribution is provided in `outputs/figures/`.

---

### Task 2: Frame Sampling (2D CNN)
- Randomly sample N frames per video  
- Study how N affects accuracy, F1, and inference time  

---

### Task 3: Temporal Crops (3D CNN)
- Apply multiple temporal crops per video  
- Improve temporal coverage and robustness  

---

### Task 4: Sequence Modeling (RNN / LSTM)
- Represent each video as a sequence of features  
- Evaluate number of sequences and repetition strategies  

---

## Results and Visualizations

All final results are saved under: outputs/

Key figures include:

- Accuracy / Macro-F1 vs frames, crops, sequences  
- Inference time comparison  
- Per-class F1 scores  
- Confusion matrices  
- Prediction examples (correct vs incorrect)  
- Dataset split distribution  

---

## Reproducibility

To regenerate visualizations from saved outputs:

```bash
python posthoc_visualization.py
