# 1D-CNN-FAO for ECG Arrhythmia Classification

This repository contains the implementation of a 1D CNN optimized using the Fractional Adam Optimizer (FAO) for ECG arrhythmia classification on the MIT-BIH Arrhythmia dataset.

## Requirements
- Python 3.8+
- TensorFlow
- NumPy
- SciPy
- scikit-learn

## Usage
1. Clone the repository
2. Install dependencies
3. Run the training script

## Dataset
MIT-BIH Arrhythmia Dataset can be downloaded from [PhysioNet]{https://www.physionet.org/content/mitdb/1.0.0/}
MIT-BIH and INCART12 heart-beat segmented dataset can be downloaded from [Kaggle]{https://www.kaggle.com/datasets/sadmansakib7/ecg-arrhythmia-classification-dataset}

```
1D-CNN-ECG/
│
├── README.md
├── requirements.txt
├── LICENSE
│
├── preprocessing/
│   ├── dwt_denoising.py
│   ├── r_peak_detection.py
│   └── normalization.py
│
├── models/
│   ├── cnn_model.py
│   └── fractional_adam.py
│
├── experiments/
│   ├── train_fao.py
│   ├── train_adam.py
│   ├── optimizer_comparison.py
│
├── evaluation/
│   ├── metrics.py
│   ├── confusion_matrix.py
││
└── utils/
    ├── seed.py
    └── data_loader.py
```
=======
# 1D-CNN-FAO
ECG Signal-based Arrhythmia Classification Using 1D CNN with Fractional Adam Optimizer


**This repository is under development**
