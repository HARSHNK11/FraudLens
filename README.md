# FraudLens 🔍

FraudLens is a modern, full-stack machine learning solution designed to accurately identify suspicious credit card transactions within extremely imbalanced real-world financial datasets. By incorporating local explainability and dynamic tuning, the system focuses on detecting rare fraudulent activities with strong recall while minimizing false positives.

---

## ⚡ Core Features

- 🧠 **End-to-End Analytics Dashboard**: A sleek dark-mode React UI for inspecting transaction inputs and analyzing fraud patterns.
- 🔍 **Explainable AI (SHAP)**: Provides clear, local explanations showing *why* a transaction was flagged.
- 📉 **Feature Importance-Based Reduction**: Reduced 30 features to 10 while preserving and slightly improving PR-AUC and recall.
- ⚙️ **FastAPI Backend**: Efficient API (`/predict`, `/model/info`, `/models/all`) for real-time predictions and model insights.

---

## 🤖 Models & Optimization

- 📊 **Dataset**: 284,807 transactions with extreme class imbalance (~0.17% fraud)

The pipeline uses **SMOTE (only on training data)** to handle imbalance and prevent data leakage.

### Models Used:
1. 🚀 **XGBoost (Best Model)** – Optimized for highest PR-AUC and balanced F1-score  
2. 🌳 **Random Forest** – Strong ensemble baseline  
3. 📈 **Logistic Regression** – Linear baseline for comparison  

---

## ⏱ Performance Results

Trained on approximately **284,807 transactions** using **10 optimized PCA features**:

| Model | PR-AUC | ROC-AUC | F1-Score | Precision | Recall |
| :--- | :---: | :---: | :---: | :---: | :---: |
| 🚀 **XGBoost (Optimized Tuned)** | **86.41%** | **97.88%** | **88.52%** | **95.29%** | **82.65%** |
| 🌳 Random Forest | 83.58% | 98.21% | 65.61% | 53.55% | 84.60% |
| 📈 Logistic Regression | 72.35% | 96.98% | 10.92% | 5.81% | 91.80% |

> ⚡ XGBoost was selected as the final production model due to superior PR-AUC, precision, and threshold tunability.

---

## 📂 Project Structure

```text
FraudLens/
├── backend/            # FastAPI backend
│   ├── main.py
│   ├── model_loader.py
│   └── schemas.py
├── frontend/           # React + Vite dashboard
│   ├── src/
│   └── public/
├── ml/                 # ML pipeline
│   └── train.py
├── artifacts/          # Saved models, plots, configs (ignored)
├── creditcard.csv      # Dataset (ignored)
└── requirements.txt
```

---

## 🚀 Getting Started

### 1️⃣ Requirements
- Node.js (v16+)
- Python (3.9+)

Download dataset from Kaggle:  
👉 https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud  

Place `creditcard.csv` in the root directory.

---

### 2️⃣ Install Dependencies

**Backend:**
```bash
pip install -r requirements.txt
```

**Frontend:**
```bash
cd frontend
npm install
```

---

### 3️⃣ Run ML Pipeline (Important)

```bash
python ml/train.py
```

This generates:
- `best_model.pkl`
- `features.json`
- other artifacts

---

### 4️⃣ Run Servers

**Backend:**
```bash
uvicorn backend.main:app --reload
```

👉 http://localhost:8000/docs

**Frontend:**
```bash
cd frontend
npm run dev
```

👉 http://localhost:5173

---

## 🧠 Highlights

- ✅ Handles extreme class imbalance effectively  
- ✅ Dynamic feature reduction improves efficiency  
- ✅ SHAP-based explainability for transparency  
- ✅ Multi-model comparison via dashboard  
- ✅ Production-ready architecture  

---
