"""
    ml/pipeline.py
    --------------
    Trains, evaluates, and saves the best model for MedAcces
    
    Enterprise practices and standards used here:
    - sklearn pipeline (preprocessing + model as one artifact)
    - Mlflow tracking (every run logged)
    - Cross-Validation (honest evaluation / not just train and test)
    - Model Selection by CV Score (not test score -  avoids overfitting to test set)
    - Full Metadata saved alongside model artifact

    run directly to train 
        python ml/pipeline.py

"""

import pandas as pd
import numpy as np
import joblib
import json
import mlflow
import mlflow.sklearn
from pathlib import Path
from datetime import datetime

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from xgboost import XGBClassifier

from app.config import settings

# ________ Paths __________
DATA_PATH = Path(settings.data_path)
MODEL_PATH = Path(settings.model_path)
META_PATH = Path(settings.metadata_path)
MLFLOW_DIR = Path(settings.model_path).parent.parent / "mlruns"

# create directories if they dont exist
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
MLFLOW_DIR.mkdir(parents=True, exist_ok=True)


#_________ Feature Columns ____________________
# These are the only Columns the model sees during Training
# The same list is used at inference time -
# If you add a feature here you must also add it to the API
# Keep in sync
FEATURE_COLS = [
    "population_log",
    "urban_score",
    "elderly_ratio",
    "gp_count",
    "specialist_density",
    "pharmacy_score",
    "wealth_index",
    "population_growth_rate",
    "avg_gp_age",
    "teleconsult_score",
]

TARGET_COL = "medical_desert_risk"

RISK_LABELS = {
    0 : "Low",
    1 : "Medium",
    2 : "High",
}

# _____________________________________________
# STEP 1 : LOAD DATA
# _____________________________________________

def load_data() -> pd.DataFrame:
    """ Load the engineered dataset """
    if not DATA_PATH.exists():
        raise FileNotFoundError(f" Data file doesnt exist at {DATA_PATH}\n"
                                 " Run this First ml/data_ingestion.py"   
                                )
    
    df = pd.read_csv(DATA_PATH)
    print(f"loaded {len(df):,} rows x {len(df.columns)} columns")
    return df

#_____________________________________________
# STEP 2: PREPROCESS
#_____________________________________________

def preprocess(df: pd.DataFrame):
    """
    split dataset into train and test sets

    key decisions made:
    - test size = 0.2 (20% held out for final evaluation)
    - stratify = y ( ensures each split has proportional
                      class distribution
                      Without this, you might get all
                      'High' cases in test and none in train 
                    )
    returns x_train, x_test, y_train, y_test
    """
    print("pre-processing .....")

    # DROP ANY ROWS THAT ARE MISSING THE CRITICAL COLUMN
    df = df.dropna(subset = FEATURE_COLS + [TARGET_COL])

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    print(f"    Features    : {len(FEATURE_COLS)}")
    print(f"    Samples     : {len(df)}")
    print(f"    Class Dist  : {dict(y.value_counts().sort_index())}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size    = 0.2,  # test train split
        random_state = 42,   # same split every run - reproducable
        stratify     = y,   # proportional class distribution
    )

    print(f"    Train Size : {len(X_train):,}")
    print(f"    Test Size : {len(X_test):,}")

    return X_train, X_test, y_train, y_test


#____________________________________________
# STEP 3 : DEFINE MODELS
#____________________________________________

def get_candidature_models() -> dict[str, Pipeline]:
    """
    Returns candidature models, each wrapped in a sklearn Pipeline.

    Why three different model types ?
        - Logistic Regression : baseline, Fast, interpretable, linear.
                                if XGBOOST barely beats it, maybe your
                                features are linearly separable and 
                                you dont need complexity
        - Random Forest       : ensemle of decision trees.
                                Robust to outliers handles non-linearity.
                                Good when you have noisy features.
        - XGBoost             : gradient boosted trees. Usually the best
                                on structured tabular data.
                                More Powerful but slower to train.
    
    At a Enterprise level you would also fine tune hyperparameters 
    using GridSearchCV or Optuna. i am using sensible defaults here to keep it basic
    """

    return {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=1000,
                C=1.0,
                random_state=42,
            )),
        ]),
        "random_forest": Pipeline([
            ("scaler", StandardScaler()),
            ("model", RandomForestClassifier(
                n_estimators=200,   # number of trees
                max_depth = 10,     # prevents overfitting
                min_samples_leaf=5, # minimum samples per leaf
                random_state=42,    
                n_jobs=-1           # use all cpu cores
            )),
        ]),
        "xgboost": Pipeline([
            ("scaler", StandardScaler()),
            ("model", XGBClassifier(
                n_estimators = 200,     # number of trees
                max_depth = 6,          # prevents overfitting
                learning_rate = 0.1,    # learning rate
                subsample = 0.8,        # 80% of rows per tree
                colsample_bytree = 0.8, # 80% of samples per tree
                random_state = 42,      
                eval_metric="mlogloss", 
                verbosity=0,            # suppress XGBoost output
            )),
        ]),
    }


#__________________________________________
# STEP4 : TRAIN AND TRACK
#__________________________________________

def train_all_models(X_train, X_test, y_train, y_test) -> dict:
    """
    Trains every models and logs output in MLflow.

    For each model:
    1. Fit on Training Data
    2. Predict on Test Data
    3. Calculate metrics
    4. Log Params + metrics + artifact
    5. Store Results for comparison

    returns dict for all the models for model selection
    """

    # Configure ML FLOW
    # SQLITE Backend - stores rundata in local file
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DIR}/mlflow.db")
    mlflow.set_experiment("medacces_desert_prediction")

    models  = get_candidature_models()
    all_results = {}

    print("\n Training Models ..............")
    print("_" * 55)
    
    for name, pipeline in models.items():
        print(f"\n {name.upper()}")

        # Each model gets its own Mlflow Run 
        with mlflow.start_run(run_name=name):
            # ______train________
            pipeline.fit(X_train, y_train)

            #______Evaluate on test set ________
            y_pred = pipeline.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            f1       = f1_score(y_test, y_pred, average="weighted")

            # ______Cross Validation____________
            # Most Honest Metric -  model sees different data
            # every fold so you catch overfitting
            # cv = 5 -> 5 folds, each fold is test once
            # n_jobs = -1  run folds in parallel

            cv_scores = cross_val_score(
                        pipeline,
                        X_train, y_train,
                        cv=5,
                        scoring="f1_weighted",
                        n_jobs=-1,
            ) 
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()

            # _______Classification Report_________
            report = classification_report(
                y_test, y_pred,
                target_names=["Low", "Medium", "High"],
                output_dict=True,
            )

            #______Log to MLflow______________
            mlflow.log_params({
                "model_type": name,
                "n_features": len(FEATURE_COLS),
                "train_size": len(X_train),
                "test_size" : len(X_test),
            })

            mlflow.log_metrics({
                "accuracy":        round(accuracy, 4),
                "f1_weighted":     round(f1, 4),
                "cv_f1_mean":      round(cv_mean, 4),
                "cv_f1_std":       round(cv_std, 4),
                "f1_low":          round(report["Low"]["f1-score"], 4),
                "f1_medium":       round(report["Medium"]["f1-score"], 4),
                "f1_high":         round(report["High"]["f1-score"], 4),
                "precision_high":  round(report["High"]["precision"], 4),
                "recall_high":     round(report["High"]["recall"], 4),
            })

            # log the full pipeline as a model artifact
            mlflow.sklearn.log_model(pipeline, "model")

            # Print Summary
            print(f"     Accuracy     : {accuracy:.4f}")
            print(f"     F1 weighted  : {f1:.4f}")
            print(f"     CV F1 mean   : {cv_mean:.4f} ± {cv_std:.4f}")
            print(f"     F1 High Risk : {report['High']['f1-score']:.4f}")

            # Store for comparison
            all_results[name] = {
                "pipeline":  pipeline,
                "accuracy":  accuracy,
                "f1":        f1,
                "cv_mean":   cv_mean,
                "cv_std":    cv_std,
                "report":    report,
                "y_pred":    y_pred,
            }

    return all_results 

#__________________________________
# STEP 5: SELECT THE BEST MODEL
#__________________________________

def select_best(results: dict) -> tuple[str, dict]:
    """
    Picks the best model using CV F1 score.

    Why CV score, not test score?
    Test score is evaluated once on one fixed split.
    By chance, your model might do well or poorly on that
    particular split — it's noisy.

    CV score averages over 5 different splits — much more
    reliable estimate of true generalization performance.

    At a company, you might also weight recall_high heavily
    because missing a real medical desert (false negative)
    is worse than a false alarm (false positive).
    """

    best_name = max(results, key=lambda k: results[k]["cv_mean"])
    best      = results[best_name]

    print(f"\n🏆 Best model : {best_name.upper()}")
    print(f"   CV F1 mean : {best['cv_mean']:.4f}")
    print(f"   Accuracy   : {best['accuracy']:.4f}")

    return best_name, best

# ─────────────────────────────────────────────────────────────────────
# STEP 6: EXTRACT FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────────────
def get_feature_importance(name: str, pipeline: Pipeline) -> dict[str, float]:
    """
    Extracts feature importance from tree-based models.
    Logistic Regression uses coefficients instead.

    Feature importance tells you WHICH features drive predictions.
    Critical for:
    - Explaining model decisions to stakeholders
    - Detecting if the model learned spurious correlations
    - Knowing which data to prioritize collecting
    """
    try:
        inner = pipeline.named_steps["model"]

        if hasattr(inner, "feature_importances_"):
            # Tree based models (Random Forest, XGBoost)
            importances = inner.feature_importances_
            return{
                feat: round(float(imp), 4)
                for feat, imp in zip(FEATURE_COLS, importances)
            }
        elif hasattr(inner, "coef_"):
            # Logistic Regression — use absolute coefficient values
            coefs = np.abs(inner.coef_).mean(axis=0)
            coefs = coefs / coefs.sum()             # normalize sum to 1    
            return{
                feat: round(float(c), 4)
                for feat, c in zip(FEATURE_COLS, coefs)
            }
    except Exception as e:
        print(f"could not extract feature importance: {e}") 
    return {}


#__________________________________
# STEP 7: Save model + metadata
#__________________________________

def save_artifacts(name :str, pipeline: Pipeline, results: dict, y_test) -> dict:
    """
    Saves two files:
    1. medacces_model.joblib  → the full sklearn pipeline (scaler + model)
    2. model_metadata.json    → metrics, features, version, timestamp

    The API loads both at startup.
    The .joblib is for predictions.
    The .json is for the /model/info endpoint.

    Why save metadata separately?
    You can read the JSON without loading the full ML model.
    Fast, lightweight, no sklearn needed.
    """
    # 1. Save Pipeline
    joblib.dump(pipeline, MODEL_PATH)
    print(f" Model Saved to the path : {MODEL_PATH}")

    # 2. Build Metadata
    importance = get_feature_importance(name, pipeline)
    cm         = confusion_matrix(y_test, results["y_pred"]).tolist()

    metadata = {
        "model_name":    name,
        "version":       datetime.now().strftime("%Y%m%d_%H%M%S"),
        "trained_at":    datetime.now().isoformat(),
        "features":      FEATURE_COLS,
        "target":        TARGET_COL,
        "risk_labels":   RISK_LABELS,

        "metrics": {
            "accuracy":    round(results["accuracy"], 4),
            "f1_weighted": round(results["f1"], 4),
            "cv_f1_mean":  round(results["cv_mean"], 4),
            "cv_f1_std":   round(results["cv_std"], 4),
        },
        "class_report": {
            cls: {
                metric: round(val, 4)
                for metric, val in vals.items()
            }
            for cls, vals in results["report"].items()
            if isinstance(vals, dict)
        },

        "feature_importance": importance,
        "confusion_matrix":   cm,
    }

    #3. Save Metadata 
    with open(META_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f" Metadata saved {META_PATH}")

    return metadata

# ─────────────────────────────────────────────────────────────────────
# MASTER FUNCTION
# ─────────────────────────────────────────────────────────────────────

def run_pipeline() -> dict:
    """
        Orchestrates the full training pipeline
        Called by:
            - This file directly when ( python ml/pipeline.py)
            - THE APIs POST /model/retrain endpoint
            - The DockerFile at build time 
    """
    print("_" * 55)
    print("🇫🇷  MedAccès — ML Training Pipeline")
    print("_" * 55)

    # 1. LOAD
    df = load_data()

    # 2. preprocess 
    X_train, X_test, y_train, y_test = preprocess(df)

    # 3. Train all Candidates + log to MLFLOW
    results = train_all_models(X_train, X_test, y_train, y_test) 

    # 4. Pick Leading Model
    best_name, best =  select_best(results)

    # 5. Save Artifacts
    metadata = save_artifacts(best_name, best["pipeline"], best, y_test)

    # 6. Print Confusion Matrix
    print("\n Confusion Matrix (rows=actual, cols=predicted):")
    print("   Labels: [Low, Medium, High]")
    for row in metadata["confusion_matrix"]:
        print(f"   {row}")

    # 7. Print feature importance
    if metadata["feature_importance"]:
        print("\n Feature Importance (top 5):")
        sorted_fi = sorted(
            metadata["feature_importance"].items(),
            key=lambda x: x[1],
            reverse=True,
        )
        for feat, imp in sorted_fi[:5]:
            bar = "█" * int(imp * 40)
            print(f"   {feat:30s} {bar} {imp:.4f}")
    print("\n" + "=" * 55)
    print("✅ Pipeline complete!")
    print(f"   Model    : {best_name}")
    print(f"   Accuracy : {metadata['metrics']['accuracy']:.4f}")
    print(f"   CV F1    : {metadata['metrics']['cv_f1_mean']:.4f}")
    print("=" * 55)

    return metadata

if __name__ == "__main__":
    run_pipeline()
    