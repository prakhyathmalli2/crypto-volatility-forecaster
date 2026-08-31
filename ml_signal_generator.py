"""
Machine Learning Trading Signal Generator Module
Trains supervised classifiers (Random Forest, XGBoost, Gradient Boosting) on quantitative features,
provides model benchmarking, ROC-AUC metrics, and probability-calibrated live signals.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score
)
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_target_labels(
    df: pd.DataFrame,
    forward_window: int = 1,
    threshold_pct: float = 0.5,
    mode: str = "binary"
) -> pd.Series:
    """
    Generate forward return target labels for supervised learning.
    """
    future_close = df['Close'].shift(-forward_window)
    forward_return = ((future_close - df['Close']) / (df['Close'] + 1e-9)) * 100.0

    if mode == "binary":
        target = (forward_return > threshold_pct).astype(int)
    else:
        target = pd.Series(0, index=df.index)
        target[forward_return > threshold_pct] = 1
        target[forward_return < -threshold_pct] = -1

    target.name = "Target"
    return target


def prepare_train_test_data(
    features_df: pd.DataFrame,
    target_series: pd.Series,
    train_ratio: float = 0.75
):
    """
    Align feature matrix and target labels chronologically without future leakage.
    """
    common_idx = features_df.index.intersection(target_series.dropna().index)
    X = features_df.loc[common_idx].copy()
    y = target_series.loc[common_idx].copy()

    # Drop any trailing rows where target could be NaN
    valid_mask = ~y.isna()
    X = X[valid_mask]
    y = y[valid_mask]

    split_idx = int(len(X) * train_ratio)

    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    return X_train, X_test, y_train, y_test, X, y


def train_signal_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: str = "Random Forest",
    random_state: int = 42
):
    """
    Train supervised ML classification model.
    """
    logger.info(f"Training {model_type} classifier on {len(X_train)} samples...")

    if model_type == "Random Forest":
        model = RandomForestClassifier(
            n_estimators=150,
            max_depth=5,
            min_samples_split=8,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1
        )
    elif model_type == "XGBoost":
        try:
            import xgboost as xgb
            model = xgb.XGBClassifier(
                n_estimators=120,
                max_depth=4,
                learning_rate=0.04,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                random_state=random_state
            )
        except Exception as e:
            logger.warning(f"XGBoost unavailable ({e}), using Gradient Boosting fallback.")
            model = GradientBoostingClassifier(
                n_estimators=100, max_depth=4, learning_rate=0.05, random_state=random_state
            )
    elif model_type == "Gradient Boosting":
        model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            random_state=random_state
        )
    else:
        model = LogisticRegression(class_weight="balanced", max_iter=500, random_state=random_state)

    model.fit(X_train, y_train)
    return model


def evaluate_model_performance(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> dict:
    """
    Evaluate classification metrics on out-of-sample test data.
    """
    y_pred = model.predict(X_test)
    
    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    
    roc_auc = 0.5
    if hasattr(model, "predict_proba"):
        try:
            y_prob = model.predict_proba(X_test)[:, 1]
            if len(np.unique(y_test)) > 1:
                roc_auc = float(roc_auc_score(y_test, y_prob))
        except Exception:
            roc_auc = 0.5

    cm = confusion_matrix(y_test, y_pred).tolist()

    return {
        "accuracy": round(acc * 100.0, 2),
        "precision": round(prec * 100.0, 2),
        "recall": round(rec * 100.0, 2),
        "f1_score": round(f1 * 100.0, 2),
        "roc_auc": round(roc_auc, 3),
        "confusion_matrix": cm,
        "test_samples": len(y_test)
    }


def benchmark_multiple_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> pd.DataFrame:
    """
    Train and benchmark multiple candidate ML models for academic comparison.
    """
    algorithms = ["Random Forest", "Gradient Boosting", "Logistic Regression"]
    results = []

    for algo in algorithms:
        m = train_signal_classifier(X_train, y_train, model_type=algo)
        metrics = evaluate_model_performance(m, X_test, y_test)
        results.append({
            "Algorithm": algo,
            "Accuracy (%)": metrics["accuracy"],
            "Precision (%)": metrics["precision"],
            "Recall (%)": metrics["recall"],
            "F1-Score (%)": metrics["f1_score"],
            "ROC-AUC": metrics["roc_auc"]
        })

    return pd.DataFrame(results)


def extract_feature_importance(model, feature_names: list) -> pd.DataFrame:
    """
    Extract and rank feature importance scores.
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        imp_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False).reset_index(drop=True)
        imp_df["Importance_Pct"] = np.round((imp_df["Importance"] / (imp_df["Importance"].sum() + 1e-9)) * 100.0, 2)
        return imp_df
    return pd.DataFrame()


def generate_live_signal(model, latest_features_row: pd.DataFrame) -> dict:
    """
    Generate actionable trading signal with confidence probability meters.
    """
    pred = int(model.predict(latest_features_row)[0])
    
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(latest_features_row)[0]
        bullish_prob = float(probs[1]) if len(probs) > 1 else 0.5
        bearish_prob = float(probs[0]) if len(probs) > 1 else 0.5
    else:
        bullish_prob = 1.0 if pred == 1 else 0.0
        bearish_prob = 1.0 - bullish_prob

    if pred == 1 and bullish_prob >= 0.53:
        signal_text = "BUY / LONG"
        badge = "success"
        action = "Bullish momentum confirmed. High probability long setup."
        confidence = bullish_prob
    elif pred == 0 and bearish_prob >= 0.53:
        signal_text = "SELL / CASH"
        badge = "error"
        action = "Bearish pressure detected. De-risk into stablecoins or exit."
        confidence = bearish_prob
    else:
        signal_text = "HOLD / NEUTRAL"
        badge = "warning"
        action = "Consolidation noise detected. Maintain position or wait for confirmation."
        confidence = max(bullish_prob, bearish_prob)

    return {
        "signal": signal_text,
        "prediction_raw": pred,
        "bullish_probability": round(bullish_prob * 100.0, 1),
        "bearish_probability": round(bearish_prob * 100.0, 1),
        "confidence_pct": round(confidence * 100.0, 1),
        "badge": badge,
        "action_recommendation": action
    }
