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

