# ML Model Comparator (Streamlit)

Upload a dataset, pick one or more ML models, optionally **fine-tune** them with Grid/Random Search, and compare results with clear metrics and plots.

---

## ✨ Features

- **Upload** CSV/XLSX or use built-in demo datasets (Breast Cancer, Wine)
- **Automatic preprocessing**: imputation, one-hot encoding, optional scaling
- **Models**: Logistic Regression, Random Forest, SVM (RBF), KNN, *(optional)* XGBoost
- **Metrics**: Accuracy, Precision/Recall/F1 (macro), full classification report
- **Visuals**: Confusion Matrix, ROC (when probabilities available)
- **Hyperparameter Tuning**: Grid Search / Randomized Search with cross-validation
- **Class balancing**: `class_weight='balanced'` for supported models
- **Fast UX**: Streamlit tabs, spinners, and intuitive sidebar controls

---

## 🧱 Tech Stack

- **Python**
- **Streamlit**
- **scikit-learn**
- **XGBoost** *(optional)*
- **pandas, numpy**
- **matplotlib, seaborn**

---

## 📁 Project Structure

ml-model-comparator/
├── app.py
├── src/
│ ├── preprocess.py # split, encode, scale, train/test split
│ ├── train.py # models, tuning helpers, training loop
│ ├── evaluate.py # metrics, plots, result table
│ └── utils.py # theme and helpers
├── data/
│ └── sample_dataset.csv (optional)
├── requirements.txt
└── README.md


---

## 🚀 Getting Started

### 1️⃣ Prerequisites

- Python **3.10+** (tested on 3.11)
- For macOS users who want to use **XGBoost**, install Homebrew first:
  ```bash
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

### 2️⃣ Install dependencies

- python -m venv venv
- source venv/bin/activate       # Windows: venv\Scripts\activate
- pip install -r requirements.txt
  
### 3️⃣ Run the app
streamlit run app.py

