# File: src/train.py
from typing import Dict, List

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline

from .evaluate import evaluate_model


def get_supported_models() -> Dict[str, object]:
    """Return a mapping of human-readable names to model instances (unfitted)."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=200, n_jobs=None),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
        "SVM (RBF)": SVC(kernel="rbf", probability=True, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=42,
        ),
    }


def _maybe_apply_class_weight(name: str, model, use_class_weight: bool):
    if not use_class_weight:
        return model
    # Apply class_weight when model supports it
    if name in {"Logistic Regression", "Random Forest"}:
        model.set_params(class_weight="balanced")
    return model


from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

def fit_models(
    model_names,
    preprocessor,
    X_train,
    y_train,
    X_test,
    y_test,
    class_names,
    use_class_weight=True,
    tuning_method="None",
    cv_folds=3,
    n_iter=10,
):
    supported = get_supported_models()
    results = []

    for name in model_names:
        base_model = supported[name]
        base_model = _maybe_apply_class_weight(name, base_model, use_class_weight)
        param_grid = get_param_grid(name)

        # ----- 🔍 Hyperparameter tuning -----
        if tuning_method != "None" and param_grid:
            if tuning_method == "Grid Search":
                search = GridSearchCV(
                    estimator=base_model,
                    param_grid=param_grid,
                    cv=cv_folds,
                    n_jobs=-1,
                    scoring="f1_macro",
                )
            elif tuning_method == "Random Search":
                search = RandomizedSearchCV(
                    estimator=base_model,
                    param_distributions=param_grid,
                    cv=cv_folds,
                    n_jobs=-1,
                    scoring="f1_macro",
                    n_iter=n_iter,
                    random_state=42,
                )
            st_text = f"Tuning {name} ({tuning_method})..."
            print(st_text)
            search.fit(X_train, y_train)
            model = search.best_estimator_
            best_params = search.best_params_
        else:
            model = base_model.fit(X_train, y_train)
            best_params = None

        # Evaluate
        res = evaluate_model(name, model, X_test, y_test, class_names)
        res["best_params"] = best_params
        results.append(res)

    results.sort(key=lambda r: (r["metrics"]["f1_macro"], r["metrics"]["accuracy"]), reverse=True)
    return results

# inside src/train.py

def get_param_grid(model_name: str):
    """Return parameter grid or distributions for tuning."""
    grids = {
        "Logistic Regression": {
            "C": [0.01, 0.1, 1, 10],
            "solver": ["lbfgs", "liblinear"],
        },
        "Random Forest": {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5, 10],
        },
        "SVM (RBF)": {
            "C": [0.1, 1, 10],
            "gamma": ["scale", "auto"],
        },
        "KNN": {
            "n_neighbors": [3, 5, 7, 9],
            "weights": ["uniform", "distance"],
        },
        "XGBoost": {
            "n_estimators": [100, 300],
            "max_depth": [3, 6, 9],
            "learning_rate": [0.01, 0.1, 0.2],
        },
    }
    return grids.get(model_name, {})

