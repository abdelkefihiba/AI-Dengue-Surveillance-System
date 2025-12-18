# AI-Driven Dengue Surveillance & Prediction System (Malaysia)

This project integrates **computer vision** and **time-series machine learning** to help monitor and predict dengue outbreaks in Malaysia. It consists of **two main modules**:

---

## Module A — Mosquito Species Detector (Computer Vision)

**Purpose:** Detect mosquito species in images to monitor Aedes mosquitoes that carry dengue.

**Input:** Mosquito images  
**Output:**  
- Species (`Aedes aegypti`, `Aedes albopictus`, or `Other`)  
- Confidence score  
- Optional: GPS/location metadata  

**Notes:**  
- Images are preprocessed using resizing, grayscale conversion, Gaussian blur, and HOG feature extraction.  
- Output features are saved as CSV for training classification models.

---

## Module B — Dengue Case Forecasting (Time-Series ML/DL)

**Purpose:** Predict weekly/monthly dengue cases per Malaysian state using historical data and weather conditions.  

**Input:**  
- Historical dengue cases (`Dataset Denggi Malaysia.xlsx`)  
- Weather data (`malaysia_weather.csv`)  
- Population data (`population_state.csv`)  
- Optional: Mosquito detection counts from Module A  

**Output:** Forecasted dengue cases per state  

**Techniques used:**  
- Feature engineering: lag features, 3-year rolling averages, outbreak labels  
- Data normalization and scaling for ML models  
- Models: LSTM, Random Forest, XGBoost, SARIMA, Prophet  

**Notes:**  
- All datasets are processed in Python using pandas, NumPy, and scikit-learn.  
- Final ML-ready datasets are normalized, with outbreak labels and scaled features.  
- Optional output is Weka-ready datasets.

---

## Project Folder Overview

AI-Dengue-Surveillance-System/
│── ModelA/
│ ├── data/ # Raw mosquito images
│ ├── preprocessing/ # Image preprocessing scripts & outputs
│ ├── models/ # Trained models, checkpoints
│ └── notebooks/ # Optional Jupyter notebooks
│
│── ModelB/
│ ├── data/
│ ├── raw/ # Raw dengue, weather, population datasets
│ └── processed/ # Cleaned and merged datasets
│ ├── preprocessing/ # Data cleaning and feature engineering scripts
│ ├── models/ # Trained forecasting models
│ └── notebooks/ # Jupyter notebooks for experiments
│
└── README.md
