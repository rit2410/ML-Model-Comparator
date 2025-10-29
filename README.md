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

---

### 🧪 Using the App

1. Upload a CSV/XLSX or select a built-in dataset.
2. Choose the target column, set test split, and random state.
3. Pick one or more models.
4. (Optional) Enable Hyperparameter Tuning:
5. Choose None, Grid Search, or Random Search
6. Adjust Cross-validation folds and iterations
7. Click 🚀 Train & Compare.
8. Review:
    - Summary table (Accuracy, Precision/Recall/F1)
    - Confusion Matrix and ROC curves per model
    - Classification Report
    - Best parameters if tuning is enabled
  
---

### 🤖 Supported Models & Tuning Grids
| Model                | Tunable Parameters                               |
| -------------------- | ------------------------------------------------ |
| Logistic Regression  | `C`, `solver`                                    |
| Random Forest        | `n_estimators`, `max_depth`, `min_samples_split` |
| SVM (RBF)            | `C`, `gamma`                                     |
| KNN                  | `n_neighbors`, `weights`                         |
| XGBoost *(optional)* | `n_estimators`, `max_depth`, `learning_rate`     |

Scoring during tuning uses f1_macro for balanced multi-class performance.

---

### ⚙️ Configuration Tips
- Scaling: Toggle numeric scaling under Advanced Preprocessing.
- Class imbalance: Enable Balance classes to use class_weight='balanced'.
- Performance: For large datasets, lower CV folds or disable tuning.

---

### 🩹 Troubleshooting

macOS + XGBoost: libomp.dylib not found

If you get:
- XGBoostError: Library (libxgboost.dylib) could not be loaded.

Fix:
# Install OpenMP runtime
brew install libomp

# Add library path (for Apple Silicon)
echo 'export DYLD_LIBRARY_PATH="/opt/homebrew/opt/libomp/lib:$DYLD_LIBRARY_PATH"' >> ~/.zshrc
source ~/.zshrc

# Reinstall XGBoost
pip install --force-reinstall xgboost
Or rely on the optional XGBoost import — the app will still work without it.
Excel Import Error
If .xlsx reading fails:
pip install openpyxl

---

🗺️ Roadmap
 SHAP explainability for tree/linear models
 Export trained models (joblib) and results (CSV)
 Add more models (LightGBM, Naive Bayes)
 Regression mode (MAE/RMSE, R²)
