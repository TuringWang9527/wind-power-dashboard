"""本地重训模型：python train_model.py --turbine all。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib

from wind_core import ROOT, load_data, train_demo


def main() -> None:
    parser = argparse.ArgumentParser(description="训练同期工况 → 相对功率的示例模型")
    parser.add_argument("--turbine", choices=["WT5", "WT6", "all"], default="all")
    parser.add_argument("--gap", type=int, default=100, help="划分边界隔离行数，不代表固定时间")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models")
    args = parser.parse_args()
    args.model_dir.mkdir(parents=True, exist_ok=True)
    turbines = ["WT6", "WT5"] if args.turbine == "all" else [args.turbine]
    for turbine in turbines:
        print(f"开始训练 {turbine}……", flush=True)
        bundle = train_demo(load_data(turbine, args.data_dir), turbine, args.gap)
        joblib.dump(bundle, args.model_dir / f"{turbine}.joblib", compress=3)
        (args.model_dir / f"{turbine}_metrics.json").write_text(
            json.dumps(bundle["metrics"], ensure_ascii=False, indent=2), encoding="utf-8")
        bundle["test_predictions"].to_csv(
            args.model_dir / f"{turbine}_test_predictions.csv", index=False, encoding="utf-8-sig")
        print(json.dumps({"turbine": turbine, "model": bundle["metrics"]["selected_model"],
                          **bundle["metrics"]["test_metrics"]}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
