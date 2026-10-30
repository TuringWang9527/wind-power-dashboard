"""检查 GitHub 项目结构、原始数据一致性和上传大小；只依赖标准库。"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import tomllib

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = [
    "app.py", "charts.py", "model_adapter.py", "wind_core.py", "train_model.py",
    "requirements.txt", "README.md", "test_dashboard.py", "DATA_SOURCES.md",
    ".streamlit/config.toml", ".github/workflows/check.yml", ".gitattributes", ".gitignore",
    "assets/style.css", "assets/turbine.svg", "docs/USAGE.md",
    "data/WT5.csv", "data/WT6.csv", "data/source_metadata.json",
    "models/WT5.joblib", "models/WT6.joblib", "models/WT5_metrics.json", "models/WT6_metrics.json",
]


def main() -> None:
    missing = [name for name in REQUIRED if not (ROOT / name).is_file()]
    if missing:
        raise SystemExit("缺少文件：" + ", ".join(missing))
    for name in REQUIRED:
        path = ROOT / name
        if path.stat().st_size > 25 * 1024 * 1024:
            raise SystemExit(f"{name} 超过 GitHub 网页上传的每文件 25 MiB 上限。")
        if path.suffix == ".py":
            ast.parse(path.read_text(encoding="utf-8"), filename=name)
    with (ROOT / ".streamlit/config.toml").open("rb") as stream:
        tomllib.load(stream)
    for turbine in ("WT5", "WT6"):
        data_bytes = (ROOT / "data" / f"{turbine}.csv").read_bytes()
        metrics = json.loads((ROOT / "models" / f"{turbine}_metrics.json").read_text(encoding="utf-8"))
        if hashlib.sha256(data_bytes).hexdigest() != metrics["source_sha256"]:
            raise SystemExit(f"{turbine} CSV 与模型报告的 SHA256 不一致。请恢复原始字节或重新训练。")
        if metrics["sklearn_version"] != "1.8.0":
            raise SystemExit(f"{turbine} 模型版本与 requirements.txt 不一致。")
        print(f"{turbine}: {metrics['total_rows']:,} rows; checksum OK")
    print("Repository layout, Python syntax, Streamlit config and dataset checksums: OK")


if __name__ == "__main__":
    main()
