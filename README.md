# FraudLens 🔍

FraudLens is a modern, full-stack machine learning solution explicitly designed to accurately identify suspicious credit card transactions within extremely imbalanced real-world financial datasets. By incorporating local explainability and dynamic tuning, the system focuses on intercepting rare fraudulent activities with robust recall metrics, while heavily minimizing false positives to ensure that genuine transactions are not unnecessarily blocked.

---

## ⚡ Core Features

- **End-to-End Analytics Dashboard**: An aesthetic, dark-mode React UI engineered to allow granular, physical inspection of transaction inputs against established fraud signatures. Features are intelligently sorted into positive and negative bias clusters for ease of manual testing.
- **Explainable AI (SHAP)**: Provides deeply transparent, localized explainers indicating precisely *why* the XGBoost ensemble flagged a transaction as illegitimate, mapped directly to global feature importance scales.
- **Dynamic Feature Reduction**: Automatically evaluates dropping noisy, insignificant PCA parameters sequentially provided that critical baseline evaluation metrics (PR-AUC, Recall) aren't heavily degraded.
- **Robust Fast API Integration**: A seamlessly scaled and beautifully structured endpoint handler (`/predict`, `/model/info`, `/model/all`) serving dynamic thresholds and multi-model inferences asynchronously.

---

## 🤖 Models & Optimization

The machine learning pipeline rigorously implements Synthetic Minority Over-sampling Technique (SMOTE) strictly localized to the cross-validation and training partitions to rigorously circumvent data leakage while resolving catastrophic class imbalances (Fraud ~0.17%). 

The internal pipeline concurrently trains three competitive algorithmic solutions:

1. **XGBoost (Best Selected Model)**: Tuned as the prevailing classifier yielding maximum Precision-Recall AUC (PR-AUC). Optimal thresholds are derived iteratively to secure recall bounds > 80% without destroying baseline F1 harmonic means.
2. **Random Forest Classifier**: Serves as a solid, high-variance baseline ensemble model. 
3. **Logistic Regression**: Retained primarily to establish a linear, weak baseline constraint to guarantee that non-linear methodologies are validating correctly.

### ⏱ Performance Results

Trained natively on roughly **284,807** transactions across **10 optimized PCA segments**, performance evaluates as follows (baseline standard bounds):

| Model | PR-AUC | ROC-AUC | F1-Score | Recall |
| :--- | :---: | :---: | :---: | :---: |
| **XGBoost (Optimized Tuned)** | **0.8641** | **0.9788** | **0.8852** | **~82.6%** |
| Random Forest | 0.8358 | 0.9821 | 0.6561 | ~84.6% |
| Logistic Regression | 0.7235 | 0.9698 | 0.1092 | ~91.8% |

*(Note: Random Forest inherently scored slightly tighter native recall bounds untuned, but XGBoost massively outpaced the ecosystem regarding precision metrics and threshold tunability, thereby serving as the ultimate production application model).*

---

## 📂 Project Structure

```text
FraudLens/
├── backend/            # FastAPI application codebase
│   ├── main.py         # Primary API endpoints and configurations
│   ├── model_loader.py # Singleton utility loading active .pkl states
│   └── schemas.py      # Pydantic structured transaction contracts
├── frontend/           # React + Vite Client Dashboard
│   ├── src/            # Contains primary styling, App logic, and state
│   └── public/         # Static global assets
├── ml/                 # Intelligence Pipelines
│   └── train.py        # Centralized ETL, scaling, SMOTE, reduction & modeling logic 
├── artifacts/          # (Ignored) Dynamically saved .pkl, .png, and .json models post-training
├── creditcard.csv      # (Ignored) Kaggle database 
└── requirements.txt    # Python runtime requirements
```

---

## 🚀 Getting Started

### 1. Requirements & Database
- Node.js (v16+)
- Python (3.9+)
- Download the raw `creditcard.csv` dataset from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it directly into the project root.

### 2. Environment Setup

*Install Python Dependencies (Backend/ML):*
```bash
pip install -r requirements.txt
```

*Install Node Dependencies (Frontend):*
```bash
cd frontend
npm install
```

### 3. Pipeline Execution (Crucial)
You **must** run the underlying ML pipeline exactly once prior to starting the servers so that the directory can successfully structure and export the local `best_model.pkl` and `features.json` parameters specifically optimized by your hardware constraint.
```bash
python ml/train.py
```

### 4. Running the Servers concurrently

*Start the FastAPI Backend (from the root directory):*
```bash
uvicorn backend.main:app --reload
```
*(The API documentation will be openly readable via Swagger UI at `http://localhost:8000/docs`).*

*Start the React Frontend:*
```bash
cd frontend
npm run dev
```
*(Navigate to `http://localhost:5173` to safely inject manual testing metrics against your active models!).*

---

## 📜 License

This project is generously licensed under the **MIT License**. You are perfectly free to use, distribute, and meticulously modify this software to better integrate into personal analytics boundaries.
