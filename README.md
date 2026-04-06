# 🧠 Predictive Maintenance AI with FastAPI & RAG

## 📌 Overview

This project is an **AI-powered predictive maintenance system** designed to detect potential machine failures in an industrial environment.

It combines:

* Machine Learning (**Random Forest / XGBoost**)
* Feature Engineering
* FastAPI backend
* RAG (Retrieval-Augmented Generation) using XML + LLM (Ollama)

The system predicts whether a component will **Pass or Fail**, and provides a **technical explanation** for operators.

---

## 🏗️ Project Architecture

```
project-ia/
│
├── frontend/                 # Vue.js frontend
│   ├── public/
│   ├── src/
│   │   ├── assets/
│   │   └── App.vue
│   └── package.json
│
├── predictive-ai/            # Backend + ML
│   ├── notebooks/
│   │   ├── data_preparation.ipynb
│   │   ├── model_training.ipynb
│   │   └── models/
│   │       ├── rf_maintenance_model.pkl
│   │       └── xgboost_maintenance_model.pkl
│   ├── data/
│   │   └── inventaire.xml
│   ├── scripts/
│   │   └── llm.py            # FastAPI backend
│   └── requirements.txt
│
├── README.md
```

---

## ⚙️ Features

### ✅ Machine Learning

* Random Forest & XGBoost models
* Handles **imbalanced data** (SMOTE, class_weight)
* Feature importance analysis

### ✅ Feature Engineering

* `xy_interaction = x * y`
* `dist_center = sqrt(x² + y²)`
* `angle = arctan2(y, x)`

### ✅ API (FastAPI)

* Real-time prediction endpoint
* JSON input → probability output
* Swagger UI for testing

### ✅ RAG + LLM

* XML inventory retrieval
* Context-aware explanation using **Ollama (Llama3)**

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd project
```

### 2. Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the API

```bash
python -m uvicorn llm:app --reload
```

Open Swagger UI:

```
http://127.0.0.1:8000/docs
```

---

## 📥 Example Request

```json
{
  "designator": 1,
  "nozzle": 5,
  "feeder": 3,
  "material_part": 12345,
  "x": 12.3,
  "y": 4.5,
  "hour": 14
}
```

---

## 📤 Example Response

```json
{
  "probability": 0.62,
  "status": "Fail",
  "explanation": "Check feeder alignment and replace worn nozzle.",
  "inventory": {
    "stock": "120",
    "loc": "Zone A / Rack 3",
    "reel": "R12345"
  }
}
```

---

## 🧪 Model Training Workflow

1. Data extraction from MongoDB
2. Data cleaning & preprocessing
3. Feature engineering
4. Encoding categorical variables
5. Train/test split (stratified)
6. Model training (RF / XGBoost)
7. Evaluation (precision, recall, F1-score)
8. Model export with `joblib`

---

## 📊 Key Challenges

* ⚠️ **Highly imbalanced dataset**

  * Few "Fail" vs many "Pass"
* Solutions:

  * SMOTE / SMOTETomek
  * Class weighting
  * Threshold tuning

---

## 📦 Technologies Used

* Python 3.12
* FastAPI
* Scikit-learn
* XGBoost
* Pandas / NumPy
* Seaborn / Matplotlib
* MongoDB
* Ollama (Llama3)

---

## 🔮 Future Improvements

* Improve Fail detection (F1-score)
* Add real-time streaming data
* Deploy with Docker
* Add frontend dashboard (Vue.js)
* Use SHAP for explainability

---

## 👨‍💻 Author

* Sirine Hamdi

---

## 📄 License

This project is for educational and industrial experimentation purposes.
