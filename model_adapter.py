"""模型接入层：大屏只调用 predict_power，即可替换成用户自己的模型。

自定义模型应提供 predict(DataFrame)；输入为 SOURCE_FEATURES 的 7 个原始列，
输出为一维“额定功率百分比”，例如 61.2 表示 61.2% 额定功率。
joblib/pickle 文件只能来自可信本地来源，不能加载陌生人上传的模型。
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
import sklearn

from wind_core import ROOT, TURBINES, load_data, train_demo, validate_features, extrapolation_flags


@lru_cache(maxsize=4)
def load_demo_bundle(turbine: str = "WT6", model_dir: str | Path | None = None,
                     data_dir: str | Path | None = None) -> dict:
    """载入随项目发布的可信模型；缺失时只在内存中训练，不自动写磁盘。"""
    if turbine not in TURBINES:
        raise ValueError(f"不支持的风机：{turbine}")
    path = Path(model_dir or ROOT / "models") / f"{turbine}.joblib"
    if path.exists():
        # 不提供网页上传模型入口。此文件来自本项目或用户本人本地训练。
        bundle = joblib.load(path)
        required = {"model", "metrics", "test_predictions", "feature_bounds", "feature_importance"}
        if not isinstance(bundle, dict) or not required.issubset(bundle):
            raise ValueError(f"{path.name} 不是支持的模型包，请运行 python train_model.py 重新训练。")
        report = bundle["metrics"]
        if report.get("turbine") != turbine or report.get("artifact_version") != 1:
            raise ValueError("模型包对应风机或格式不匹配，请重新训练。")
        if report.get("sklearn_version") != sklearn.__version__:
            warnings.warn("当前 scikit-learn 版本与模型训练版本不同，建议使用 requirements.txt 或重训。",
                          RuntimeWarning)
        # 防止替换数据之后，大屏继续把旧模型结果标为新数据的测试结果。
        data = load_data(turbine, data_dir=data_dir)
        if report.get("source_sha256") and report["source_sha256"] != data.attrs["source_sha256"]:
            raise ValueError("数据文件与模型训练来源不一致，请运行 python train_model.py 重训。")
        return bundle
    return train_demo(load_data(turbine, data_dir=data_dir), turbine=turbine)


@lru_cache(maxsize=4)
def _load_custom(path: str, modified_ns: int):
    """文件修改时间加入缓存键，重训替换后会载入新版本。"""
    del modified_ns
    model = joblib.load(path)
    if not callable(getattr(model, "predict", None)):
        raise TypeError("自定义文件必须是包含 predict 方法的模型或 sklearn Pipeline。")
    return model


def predict_power(raw: pd.DataFrame, turbine: str = "WT6", mode: str = "demo",
                  custom_model_path: str | Path | None = None,
                  bundle: dict | None = None) -> np.ndarray:
    """模型按钮的统一接口；返回预测百分比，不自动裁剪为 0–100。"""
    values = validate_features(raw)
    if mode == "demo":
        selected = bundle if bundle is not None else load_demo_bundle(turbine)
        model = selected["model"]
    elif mode == "custom":
        path = Path(custom_model_path or ROOT / "models" / "custom_model.joblib").resolve()
        if not path.is_file():
            raise FileNotFoundError(f"未找到自定义模型：{path}")
        model = _load_custom(str(path), path.stat().st_mtime_ns)
    else:
        raise ValueError("mode 只能是 'demo' 或 'custom'。")
    predicted = np.asarray(model.predict(values), dtype=float)
    if predicted.ndim == 2 and predicted.shape[1] == 1:
        predicted = predicted[:, 0]
    if predicted.ndim != 1 or len(predicted) != len(values):
        raise ValueError("模型应为每条样本输出一个数值；目前输出形状不正确。")
    if not np.isfinite(predicted).all():
        raise ValueError("模型返回了空值或无穷值。")
    return predicted
