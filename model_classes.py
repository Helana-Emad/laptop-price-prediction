import re

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from catboost import CatBoostRegressor

# الكلاسات دي منسوخة من النوتبوك زي ما هي، لازم تفضل بنفس الشكل
# عشان joblib يقدر يفتح الـ pipeline المحفوظ.

NUMERIC_FEATURES = ["RAM", "Storage", "Screen"]
CATEGORICAL_FEATURES = [
    "Status", "Brand", "Model", "CPU", "Storage type", "GPU", "Touch",
    "CPU_Detail", "GPU_Detail"
]
MODEL_FEATURES = [
    "Status", "Brand", "Model", "CPU", "RAM", "Storage", "Storage type",
    "GPU", "Screen", "Touch", "CPU_Detail", "GPU_Detail"
]
RAW_REQUIRED_FEATURES = [
    "Status", "Brand", "Model", "CPU", "RAM", "Storage",
    "Storage type", "GPU", "Screen", "Touch"
]

CPU_PATTERN = re.compile(
    r"\b(?:"
    r"Intel(?: Evo)?\s+Core\s+i[3579]-\d{4,5}[A-Z]{0,3}\d{0,2}"
    r"|Intel\s+Core\s+Ultra\s+[579]\s+\d{3,4}[A-Z]{0,3}\d{0,2}"
    r"|AMD\s+Ryzen\s+[3579]\s+\d{4,5}[A-Z]{0,3}\d{0,2}"
    r"|Intel\s+Celeron\s+[A-Z]?\d{3,5}[A-Z]{0,2}"
    r"|Intel\s+Pentium\s+[A-Z]?\d{3,5}[A-Z]{0,2}"
    r"|Apple\s+M[1-5](?:\s+(?:Pro|Max|Ultra))?"
    r")\b",
    flags=re.IGNORECASE,
)

GPU_PATTERN = re.compile(
    r"\b(?:"
    r"(?:NVIDIA\s+)?(?:GeForce\s+)?RTX\s*\d{3,4}(?:\s*(?:Ti|SUPER))?"
    r"|(?:NVIDIA\s+)?(?:GeForce\s+)?GTX\s*\d{3,4}(?:\s*Ti)?"
    r"|Radeon\s+RX\s*\d{3,4}[A-Z]{0,2}"
    r"|RX\s*\d{3,4}[A-Z]{0,2}"
    r"|GeForce\s+MX\s*\d{3}"
    r"|Intel\s+(?:Arc\s+[A-Z]\d{3}|Iris\s+Xe|UHD\s+Graphics|HD\s+Graphics)"
    r")\b",
    flags=re.IGNORECASE,
)


def _clean_text(value, fallback="Unknown"):
    if pd.isna(value) or str(value).strip() == "":
        return fallback
    return str(value).strip()


class LaptopFeatureEngineer(BaseEstimator, TransformerMixin):
    """Prepare raw laptop specification columns for CatBoost."""

    def fit(self, X, y=None):
        X = self._as_frame(X)
        missing = [col for col in RAW_REQUIRED_FEATURES if col not in X.columns]
        if missing:
            raise ValueError(f"Missing required input columns: {missing}")

        numeric = X[NUMERIC_FEATURES].apply(pd.to_numeric, errors="coerce")
        self.numeric_medians_ = numeric.median().fillna(
            {"RAM": 16.0, "Storage": 512.0, "Screen": 15.6}
        )
        self.feature_names_out_ = np.asarray(MODEL_FEATURES, dtype=object)
        return self

    @staticmethod
    def _as_frame(X):
        if isinstance(X, pd.DataFrame):
            return X.copy()
        raise TypeError("LaptopFeatureEngineer expects a pandas DataFrame.")

    def transform(self, X):
        if not hasattr(self, "numeric_medians_"):
            raise RuntimeError("Fit the feature transformer before calling transform.")

        X = self._as_frame(X)
        missing = [col for col in RAW_REQUIRED_FEATURES if col not in X.columns]
        if missing:
            raise ValueError(f"Missing required input columns: {missing}")

        laptop_names = X["Laptop"] if "Laptop" in X.columns else pd.Series("", index=X.index)
        result = X[RAW_REQUIRED_FEATURES].copy()

        cpu_details = []
        gpu_details = []
        for idx, name in laptop_names.items():
            text = "" if pd.isna(name) else str(name)
            cpu_match = CPU_PATTERN.search(text)
            gpu_match = GPU_PATTERN.search(text)

            raw_cpu = X.at[idx, "CPU"]
            raw_gpu = X.at[idx, "GPU"]
            cpu_details.append(cpu_match.group(0).strip() if cpu_match else _clean_text(raw_cpu))
            gpu_details.append(gpu_match.group(0).strip() if gpu_match else _clean_text(raw_gpu, "Not listed"))

        result["CPU_Detail"] = cpu_details
        result["GPU_Detail"] = gpu_details

        for col in NUMERIC_FEATURES:
            result[col] = pd.to_numeric(result[col], errors="coerce")
            result[col] = result[col].fillna(float(self.numeric_medians_[col]))

        for col in CATEGORICAL_FEATURES:
            result[col] = result[col].map(lambda value: _clean_text(value, "Unknown"))

        return result[MODEL_FEATURES]

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_out_.copy()


class CompatibleCatBoostRegressor(RegressorMixin, BaseEstimator):
    """Small sklearn-compatible wrapper around CatBoostRegressor."""

    def __init__(
        self,
        iterations=1716,
        learning_rate=0.04,
        depth=6,
        loss_function="RMSE",
        eval_metric="RMSE",
        l2_leaf_reg=5,
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
        cat_features=None,
    ):
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.loss_function = loss_function
        self.eval_metric = eval_metric
        self.l2_leaf_reg = l2_leaf_reg
        self.random_seed = random_seed
        self.verbose = verbose
        self.allow_writing_files = allow_writing_files
        self.thread_count = thread_count
        self.cat_features = cat_features

    def fit(self, X, y):
        self.model_ = CatBoostRegressor(
            iterations=self.iterations,
            learning_rate=self.learning_rate,
            depth=self.depth,
            loss_function=self.loss_function,
            eval_metric=self.eval_metric,
            l2_leaf_reg=self.l2_leaf_reg,
            random_seed=self.random_seed,
            verbose=self.verbose,
            allow_writing_files=self.allow_writing_files,
            thread_count=self.thread_count,
            cat_features=self.cat_features,
        )
        self.model_.fit(X, y)
        return self

    def predict(self, X):
        if not hasattr(self, "model_"):
            raise RuntimeError("Train the regressor before prediction.")
        return self.model_.predict(X)

    @property
    def feature_importances_(self):
        return self.model_.feature_importances_
