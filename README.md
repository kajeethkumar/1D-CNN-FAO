


```
1D-CNN-ECG/
│
├── README.md
├── requirements.txt
├── LICENSE
│
├── data/
│   ├── README.md
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