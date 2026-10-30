"""风电大屏的数据与模型公共逻辑。

模型估计同期气象条件下的相对功率；CSV 没有时间戳，不能据此解释为未来预测。
"""
from __future__ import annotations

import hashlib
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

ROOT = Path(__file__).resolve().parent
SOURCE_FEATURES = ["V", "D", "air density", "humidity", "I", "S_a", "S_b"]
TARGET = "y (% relative to rated power)"
ID_COLUMN = "Sequence No."
TURBINES = ("WT6", "WT5")
FEATURE_LABELS = {
    "V": "风速 (m/s)",
    "D": "风向 (°)",
    "air density": "空气密度 (kg/m³)",
    "humidity": "湿度（原始值）",
    "I": "湍流强度",
    "S_a": "上部风切变指数",
    "S_b": "下部风切变指数",
}


def validate_features(raw: pd.DataFrame) -> pd.DataFrame:
    """按固定列顺序返回有限数值，明确拒绝缺失值与无效物理输入。"""
    if not isinstance(raw, pd.DataFrame):
        raise TypeError("模型输入必须为 pandas DataFrame。")
    if raw.empty:
        raise ValueError("模型输入没有样本。")
    if raw.columns.duplicated().any():
        raise ValueError("数据包含重复列名。")
    missing = [name for name in SOURCE_FEATURES if name not in raw.columns]
    if missing:
        raise ValueError("缺少特征列：" + ", ".join(missing))
    try:
        values = raw.loc[:, SOURCE_FEATURES].apply(pd.to_numeric, errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError("七个模型特征必须都是数值。") from exc
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("特征包含空值、无穷值或非法数值。请先修正数据。")
    checks = [
        ((values["V"] < 0).any(), "风速不能为负数。"),
        ((values["air density"] <= 0).any(), "空气密度必须大于 0。"),
        ((values["humidity"] < 0).any(), "湿度原始值不能为负数。"),
        ((values["I"] < 0).any(), "湍流强度不能为负数。"),
        (((values["D"] < 0) | (values["D"] > 360)).any(), "风向须在 0–360° 范围内。"),
    ]
    for invalid, message in checks:
        if invalid:
            raise ValueError(message)
    return values.astype(float)


def make_features(raw: pd.DataFrame) -> pd.DataFrame:
    """风向进行 sin/cos 编码，7 个原始输入变为 8 个特征。"""
    values = validate_features(raw).copy()
    direction = np.deg2rad(values.pop("D"))
    values["wind_direction_sin"] = np.sin(direction)
    values["wind_direction_cos"] = np.cos(direction)
    return values


def load_data(turbine: str = "WT6", data_dir: str | Path | None = None) -> pd.DataFrame:
    """读取原始 CSV，按发布序号排序；序号不是时间戳。"""
    if turbine not in TURBINES:
        raise ValueError(f"不支持的风机：{turbine}")
    path = Path(data_dir or ROOT / "data") / f"{turbine}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"缺少数据文件：{path}。请保留项目内的 data 文件夹。")
    frame = pd.read_csv(path)
    required = [ID_COLUMN] + SOURCE_FEATURES + [TARGET]
    missing = [name for name in required if name not in frame]
    if missing:
        raise ValueError("数据缺少列：" + ", ".join(missing))
    if frame.columns.duplicated().any():
        raise ValueError("数据包含重复列名。")
    frame = frame.loc[:, required].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError("数据包含空值或非有限数值，程序不会自动删除或插补。")
    validate_features(frame)
    if frame[ID_COLUMN].duplicated().any():
        raise ValueError("样本序号重复。")
    if (frame[ID_COLUMN] < 1).any() or (frame[ID_COLUMN] % 1 != 0).any():
        raise ValueError("样本序号必须是正整数。")
    frame[ID_COLUMN] = frame[ID_COLUMN].astype("int64")
    frame = frame.sort_values(ID_COLUMN, kind="stable").reset_index(drop=True)
    frame.attrs["source_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    frame.attrs["source_file"] = path.name
    return frame


def scores(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """RMSE/MAE 的单位为额定功率的百分点。"""
    return {
        "r2": float(r2_score(actual, predicted)),
        "rmse_rated_power_percentage_points": float(np.sqrt(mean_squared_error(actual, predicted))),
        "mae_rated_power_percentage_points": float(mean_absolute_error(actual, predicted)),
    }


def _pipeline(estimator) -> Pipeline:
    return Pipeline([
        ("wind_features", FunctionTransformer(make_features, validate=False)),
        ("regressor", estimator),
    ])


def train_demo(data: pd.DataFrame, turbine: str = "WT6", gap: int = 100) -> dict:
    """验证集选择 RF/HGB，前 80% 重训，末段留出集只用于评估。"""
    if gap < 0:
        raise ValueError("gap 必须为非负整数。")
    if not isinstance(gap, int):
        raise TypeError("gap 必须是整数。")
    data = data.sort_values(ID_COLUMN, kind="stable").reset_index(drop=True)
    raw = validate_features(data)
    y = data[TARGET].to_numpy(dtype=float)
    if not np.isfinite(y).all():
        raise ValueError("目标功率包含非有限值。")
    n = len(data)
    b1, b2 = int(n * 0.60), int(n * 0.80)
    if b1 < 100 or b1 + gap >= b2 or b2 + gap + 1 >= n:
        raise ValueError("样本不足，或 gap 过大，不能建立有效的训练/验证/测试分块。")
    validation = slice(b1 + gap, b2)
    test = slice(b2 + gap, n)
    candidates = {
        "RandomForest": _pipeline(RandomForestRegressor(
            n_estimators=180, min_samples_leaf=5, max_features=1.0,
            random_state=42, n_jobs=2,
        )),
        "HistGradientBoosting": _pipeline(HistGradientBoostingRegressor(
            learning_rate=0.07, max_iter=300, max_leaf_nodes=31,
            min_samples_leaf=20, l2_regularization=1.0,
            early_stopping=False, random_state=42,
        )),
    }
    validation_metrics = {}
    for name, model in candidates.items():
        model.fit(raw.iloc[:b1], y[:b1])
        validation_metrics[name] = scores(y[validation], model.predict(raw.iloc[validation]))
    chosen = min(validation_metrics,
                 key=lambda name: validation_metrics[name]["rmse_rated_power_percentage_points"])
    model = clone(candidates[chosen]).fit(raw.iloc[:b2], y[:b2])
    predicted = np.asarray(model.predict(raw.iloc[test]), dtype=float)
    mean_model = DummyRegressor(strategy="mean").fit(raw.iloc[:b2], y[:b2])
    mean_prediction = mean_model.predict(raw.iloc[test])
    bounds = {
        col: {"min": float(raw.iloc[:b2][col].min()),
              "max": float(raw.iloc[:b2][col].max()),
              "median": float(raw.iloc[:b2][col].median())}
        for col in SOURCE_FEATURES
    }
    # 在独立测试集的固定抽样上做置换解释，不用于模型选择；没有因果含义。
    explanation_indices = np.linspace(b2 + gap, n - 1, min(1200, n - b2 - gap), dtype=int)
    importance = permutation_importance(
        model, raw.iloc[explanation_indices], y[explanation_indices],
        n_repeats=3, random_state=42, scoring="neg_root_mean_squared_error", n_jobs=2,
    )
    feature_importance = pd.DataFrame({
        "feature": SOURCE_FEATURES,
        "label": [FEATURE_LABELS[col] for col in SOURCE_FEATURES],
        "importance": importance.importances_mean,
        "std": importance.importances_std,
    }).sort_values("importance", ascending=False, kind="stable").reset_index(drop=True)
    metrics = {
        "artifact_version": 1,
        "dataset": "DSWE Inland-Offshore Wind Farm Dataset1",
        "source_doi": "10.5281/zenodo.5516552",
        "source_sha256": data.attrs.get("source_sha256", ""),
        "turbine": turbine,
        "task": "同期气象条件下的相对功率回归；非未来时点预测",
        "split": "按 Sequence No. 排序 60/20/20 分块；两边界后各隔离 gap 行；无实际时间戳",
        "gap_rows": gap,
        "total_rows": n,
        "training_rows_for_model_selection": b1,
        "validation_rows": b2 - b1 - gap,
        "final_fit_rows": b2,
        "test_rows": n - b2 - gap,
        "test_start_sequence": int(data.iloc[b2 + gap][ID_COLUMN]),
        "source_features": SOURCE_FEATURES.copy(),
        "encoded_features": list(make_features(raw.iloc[:1]).columns),
        "validation_metrics": validation_metrics,
        "selected_model": chosen,
        "test_metrics": scores(y[test], predicted),
        "mean_baseline_test_metrics": scores(y[test], mean_prediction),
        "target_unit": "percent_of_rated_power",
        "error_unit": "rated_power_percentage_points",
        "importance_method": "测试集固定抽样置换重要性；数值为 RMSE 增量（额定功率百分点）；3 次重复",
        "importance_rows": len(explanation_indices),
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
    }
    test_predictions = pd.DataFrame({
        "source_sequence_no": data.iloc[test][ID_COLUMN].to_numpy(),
        "wind_speed_m_per_s": data.iloc[test]["V"].to_numpy(),
        "actual_power_pct_rated": y[test],
        "predicted_power_pct_rated": predicted,
        "mean_baseline_power_pct_rated": mean_prediction,
        "residual_percentage_points": y[test] - predicted,
    })
    return {"model": model, "metrics": metrics, "test_predictions": test_predictions,
            "feature_bounds": bounds, "feature_importance": feature_importance}


def extrapolation_flags(raw: pd.DataFrame, bounds: dict) -> list[str]:
    """返回超出训练范围的变量提示；不裁剪输入/输出，也不代表可靠性保证。"""
    data = validate_features(raw)
    messages = []
    for name in SOURCE_FEATURES:
        if name not in bounds:
            continue
        lo, hi = bounds[name]["min"], bounds[name]["max"]
        # 风向为周期变量，0 与 360° 相邻；不以度数极值判定外推。
        if name == "D":
            continue
        count = int(((data[name] < lo) | (data[name] > hi)).sum())
        if count:
            messages.append(f"{FEATURE_LABELS[name]}：{count} 条输入超出训练范围 [{lo:.4g}, {hi:.4g}]。")
    return messages
