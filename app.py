"""海上风电智慧大屏。

启动：python -m streamlit run app.py
预测按钮统一调用 model_adapter.predict_power；默认使用真实训练的示例模型。
"""
from __future__ import annotations

import base64
import hashlib
import html
import io
import json
import os
from pathlib import Path

# 限制 CPU 线程，避免演示机器同时绘图和批量预测时过度占用资源。
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import numpy as np
import pandas as pd
import streamlit as st

import charts
from model_adapter import load_demo_bundle, predict_power
from wind_core import (
    FEATURE_LABELS, ID_COLUMN, ROOT, SOURCE_FEATURES, TARGET, TURBINES,
    extrapolation_flags, load_data, validate_features,
)

st.set_page_config(
    page_title="海上风电智慧大屏", page_icon="🌊", layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "公开海上风机数据上的工况分析与相对功率建模展示。"},
)
st.html(f"<style>{(ROOT / 'assets' / 'style.css').read_text(encoding='utf-8')}</style>")

CUSTOM_MODEL = ROOT / "models" / "custom_model.joblib"
INPUT_KEYS = {name: f"feature_{i}" for i, name in enumerate(SOURCE_FEATURES)}
CHART_CONFIG = {
    "displayModeBar": False, "responsive": True, "scrollZoom": False,
    "toImageButtonOptions": {"format": "png", "scale": 2},
}


@st.cache_data(show_spinner=False)
def cached_data(turbine: str, modified_ns: int) -> pd.DataFrame:
    del modified_ns
    return load_data(turbine)


@st.cache_resource(show_spinner=False)
def cached_bundle(turbine: str) -> dict:
    return load_demo_bundle(turbine)


def plot(fig, key: str) -> None:
    # Streamlit 1.50 的官方接口；依赖版本已在 requirements.txt 固定。
    st.plotly_chart(fig, use_container_width=True, theme=None, key=key, config=CHART_CONFIG)


def panel_title(title: str, subtitle: str = "") -> None:
    st.html(f'<div class="panel-title"><strong>{html.escape(title)}</strong>'
            f'<span>{html.escape(subtitle)}</span></div>')


def kpi(label: str, value: str, unit: str, foot: str, accent: bool = False) -> None:
    css = "kpi-card kpi-accent" if accent else "kpi-card"
    st.html(f'<div class="{css}"><div class="kpi-label">{html.escape(label)}</div>'
            f'<div class="kpi-value">{html.escape(value)}'
            f'<span class="kpi-unit">{html.escape(unit)}</span></div>'
            f'<div class="kpi-foot">{html.escape(foot)}</div></div>')


def csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode("utf-8-sig")


def seed_inputs(row: pd.Series) -> None:
    for name in SOURCE_FEATURES:
        st.session_state[INPUT_KEYS[name]] = float(row[name])
    st.session_state["last_prediction"] = None


with st.sidebar:
    st.html('<div class="side-brand"><span>◈</span> 海风数智</div>'
            '<div class="eyebrow">OFFSHORE WIND INSIGHT</div>')
    turbine = st.selectbox("分析机组", TURBINES, key="turbine")
    mode_label = st.selectbox("预测模型", ["内置示例模型", "自定义本地模型"], key="model_mode")
    mode = "demo" if mode_label == "内置示例模型" else "custom"
    st.caption("数据：DSWE 公开海上机组样本\n\n采集期：2009 年；界面展示历史工况。")
    if mode == "custom":
        if CUSTOM_MODEL.is_file():
            st.success("已找到自定义模型文件。")
        else:
            st.warning("尚未接入自定义模型。请按 README 放置 models/custom_model.joblib。")
    st.divider()
    st.markdown("**展示范围**")
    window_size = st.select_slider("测试样本窗口", options=[40, 80, 120, 200], value=80, key="window_size")
    threshold = st.slider("偏差提示阈值 / 个百分点", 5, 50, 20, 1, key="residual_threshold")
    st.caption("仅标记 |实测 − 预测| 超过阈值的样本，供后续核查。")
    st.divider()
    convert_mw = st.checkbox("显示自设额定容量换算", value=False, key="convert_mw")
    rated_mw = None
    if convert_mw:
        rated_mw = st.number_input("自设额定容量 / MW", min_value=.1, max_value=100., value=5.,
                                   step=.1, key="rated_mw")
        st.caption("5 MW 为可修改的演示假设，原数据未披露机组额定容量。")
    st.divider()
    st.caption("建议在电脑浏览器全屏展示。\n\n模型接入与重训步骤见项目 README。")

try:
    data = cached_data(turbine, (ROOT / "data" / f"{turbine}.csv").stat().st_mtime_ns)
    with st.spinner("正在加载示例模型与测试集……"):
        bundle = cached_bundle(turbine)
except Exception as exc:
    st.error(f"初始化失败：{exc}")
    st.info("请确认完整解压项目并安装 requirements.txt；数据改动后请重训并重启应用。")
    st.stop()

metrics = bundle["metrics"]
if metrics.get("source_sha256") != data.attrs.get("source_sha256"):
    st.error("当前 CSV 已变化，缓存模型与数据不一致。请重新训练并重启应用。")
    st.stop()
test = bundle["test_predictions"]
bounds = bundle["feature_bounds"]
evaluation = metrics["test_metrics"]
source_seed = data.loc[data[ID_COLUMN] == int(test.iloc[0]["source_sequence_no"])].iloc[0]
custom_stamp = CUSTOM_MODEL.stat().st_mtime_ns if CUSTOM_MODEL.is_file() else 0
context = f"{turbine}|{mode}|{custom_stamp if mode == 'custom' else 0}"
if st.session_state.get("prediction_context") != context:
    seed_inputs(source_seed)
    st.session_state["prediction_context"] = context
    st.session_state["batch_result"] = None

st.html(f'''<div class="dashboard-head">
  <div class="title-line"><div class="brand-symbol">≋</div><div>
    <div class="eyebrow">OFFSHORE WIND · INTELLIGENT DASHBOARD</div>
    <h1>海上风电智慧大屏</h1><p>工况洞察　/　功率建模　/　性能分析</p>
  </div></div>
  <div class="head-right"><div class="status-pill"><i class="status-dot"></i>公开数据演示</div>
  <div>{turbine} 海上机组 · 预测接口：{html.escape(mode_label)}</div></div>
</div>''')

cards = st.columns(4, gap="small")
with cards[0]:
    kpi("当前机组样本", f"{len(data):,}", "条", "完整公开数据 · 未补造时间戳")
with cards[1]:
    kpi("样本平均风速", f"{data['V'].mean():.2f}", "m/s", "当前机组全量样本均值")
with cards[2]:
    kpi("平均相对功率", f"{data[TARGET].mean():.2f}", "%", "相对额定功率 · 全量实测均值")
with cards[3]:
    kpi("示例模型 · 测试集 R²", f"{evaluation['r2']:.4f}", "", f"留出集 {len(test):,} 条 · {metrics['selected_model']}", True)

overview, evaluation_tab, batch_tab, data_tab = st.tabs(["运行总览", "模型评估", "批量预测", "数据与说明"])

with overview:
    left, center, right = st.columns([1.02, 1.58, 1.08], gap="small")
    with left:
        with st.container(key="panel_power_curve"):
            panel_title("风速—功率分布", "POWER CURVE")
            plot(charts.power_curve(data), "power_curve")
            st.caption("散点抽样最多 1,600 条；曲线为全量样本按 0.5 m/s 分箱中位数。")
        with st.container(key="panel_rose"):
            panel_title("风向与风速分布", "WIND ROSE")
            plot(charts.wind_rose(data), "wind_rose")
            st.caption("径向为样本频率；由深至浅对应 <5、5–8、8–11、11–14、≥14 m/s。")
    with center:
        with st.container(key="panel_hero"):
            panel_title("机组工况视图", "ASSET OVERVIEW")
            # 图是自绘结构示意，并非风场真实地理位置或在线遥测。
            svg = base64.b64encode((ROOT / "assets" / "turbine.svg").read_bytes()).decode()
            st.html(f'''<div class="hero-stage"><img src="data:image/svg+xml;base64,{svg}" alt="海上风机结构示意"/>
            <div class="hero-stamp">SELECTED TURBINE<b>{turbine}</b>公开海上机组数据</div>
            <div class="hero-badge">历史样本分析</div><div class="hero-caption">风机结构示意 · 无实际地理坐标</div></div>
            <div class="hero-values"><div><b>{data['V'].min():.1f}–{data['V'].max():.1f}</b>风速范围 / m·s⁻¹</div>
            <div><b>{data['air density'].mean():.3f}</b>平均空气密度 / kg·m⁻³</div>
            <div><b>{metrics['test_rows']:,}</b>独立留出样本 / 条</div></div>''')
        with st.container(key="panel_trend"):
            panel_title("测试集实测与预测", "SAMPLE SEQUENCE")
            max_start = max(0, len(test) - window_size)
            start_key = f"window_start_{turbine}"
            if st.session_state.get(start_key, 0) > max_start:
                st.session_state[start_key] = max_start
            start = st.slider("窗口起点（测试集内位置）", 0, max_start, 0, key=start_key,
                              label_visibility="collapsed", help="按发布序号查看测试样本，不代表连续的 10 分钟时间序列。")
            window = test.iloc[start:start + window_size]
            plot(charts.sequence_chart(window, threshold), "sequence")
            flagged_count = int((window["residual_percentage_points"].abs() > threshold).sum())
            st.caption(f"窗口 {len(window)} 条 · {flagged_count} 条超过 {threshold} 个百分点阈值 · 曲线使用内置示例模型。")
    with right:
        with st.container(key="panel_prediction"):
            panel_title("AI 功率预测", "PREDICTION LAB")
            st.caption("输入同一工况的 7 个环境变量，点击按钮计算相对功率。")
            if st.button("使用窗口首条工况", key="use_window_sample", width="stretch"):
                selected_id = int(window.iloc[0]["source_sequence_no"])
                seed_inputs(data.loc[data[ID_COLUMN] == selected_id].iloc[0])
            with st.form("prediction_form", clear_on_submit=False, border=False):
                inputs = {}
                col_a, col_b = st.columns(2, gap="small")
                specs = [
                    ("V", "风速 / m·s⁻¹", 0., None, .1, "%.3f", col_a),
                    ("D", "风向 / °", 0., 360., 1., "%.2f", col_b),
                    ("air density", "空气密度 / kg·m⁻³", .001, None, .001, "%.4f", col_a),
                    ("humidity", "湿度 / 原始值", 0., None, .1, "%.3f", col_b),
                    ("I", "湍流强度", 0., None, .005, "%.4f", col_a),
                    ("S_a", "上部风切变指数", None, None, .01, "%.4f", col_b),
                    ("S_b", "下部风切变指数", None, None, .01, "%.4f", col_a),
                ]
                for name, label, minimum, maximum, step, fmt, container in specs:
                    with container:
                        inputs[name] = st.number_input(
                            label, min_value=minimum, max_value=maximum,
                            step=step, format=fmt, key=INPUT_KEYS[name],
                        )
                submitted = st.form_submit_button("执行功率预测", type="primary", width="stretch")

            if submitted:
                # 真实模型推理入口：替换 model_adapter.py 即可接入自己的模型或 API。
                st.session_state["last_prediction"] = None
                try:
                    raw = pd.DataFrame([inputs], columns=SOURCE_FEATURES)
                    prediction = float(predict_power(raw, turbine=turbine, mode=mode,
                                                    custom_model_path=CUSTOM_MODEL, bundle=bundle)[0])
                    warnings = extrapolation_flags(raw, bounds) if mode == "demo" else []
                    st.session_state["last_prediction"] = {
                        "context": context, "turbine": turbine, "mode": mode,
                        "value": prediction, "inputs": inputs.copy(), "warnings": warnings,
                    }
                except Exception as exc:
                    st.error(f"预测未完成：{exc}")

            result = st.session_state.get("last_prediction")
            if result is not None and result["context"] == context:
                conversion = (f"按自设 {rated_mw:.2f} MW 额定容量换算：{result['value'] * rated_mw / 100:.3f} MW"
                              if rated_mw is not None else "输出为额定功率的百分比")
                st.html(f'''<div class="result-card"><div class="result-title">已提交工况 · {html.escape(mode_label)}</div>
                <div class="result-value">{result['value']:.2f}<small>% 额定功率</small></div>
                <p>{html.escape(conversion)}<br>所用风速 {result['inputs']['V']:.3f} m/s · 风向 {result['inputs']['D']:.2f}°</p></div>''')
                st.caption("结果对应上次点击提交的工况；修改输入后请再次执行预测。")
                for message in result["warnings"]:
                    st.warning(message)
                if result["value"] < 0 or result["value"] > 100:
                    st.warning("模型输出超出 0–100% 范围，已保留原始结果，请核查适用工况。")
                out = pd.DataFrame([result["inputs"]])
                out.insert(0, "turbine", turbine)
                out["prediction_pct_rated"] = result["value"]
                out["model_mode"] = mode
                if rated_mw is not None:
                    out["assumed_rated_capacity_mw"] = rated_mw
                    out["prediction_mw_from_assumption"] = result["value"] * rated_mw / 100
                st.download_button("下载本次预测", csv_bytes(out), f"{turbine}_prediction.csv",
                                   mime="text/csv", key="download_single", width="stretch")
            else:
                st.html('<div class="empty-result"><b>等待工况推演</b>填写或载入工况，然后执行功率预测。</div>')
            st.caption("湿度保留数据原始单位。超出训练范围时会提示；不确定性区间尚未估计。")

    if st.session_state.get("last_prediction") and st.session_state["last_prediction"]["mode"] == "demo":
        with st.expander("查看已提交工况的风速响应", expanded=False):
            last = st.session_state["last_prediction"]
            raw = pd.DataFrame([last["inputs"]], columns=SOURCE_FEATURES)
            plot(charts.wind_sensitivity(raw, lambda x: predict_power(x, turbine=turbine, bundle=bundle), bounds), "sensitivity")
            st.caption("固定其余输入，仅改变风速。部分组合可能不常见，此图表示模型响应，不代表因果关系。")

with evaluation_tab:
    st.markdown(f"### {turbine} · 内置示例模型评估")
    st.caption("本页指标固定对应随项目提供的示例模型；自定义模型接入后，不会自动继承这些成绩。")
    cols = st.columns(4)
    for col, label, value, foot in [
        (cols[0], "测试集 R²", f"{evaluation['r2']:.4f}", "决定系数 · 非置信度"),
        (cols[1], "测试集 RMSE", f"{evaluation['rmse_rated_power_percentage_points']:.3f}", "额定功率百分点"),
        (cols[2], "测试集 MAE", f"{evaluation['mae_rated_power_percentage_points']:.3f}", "额定功率百分点"),
        (cols[3], "最终拟合样本", f"{metrics['final_fit_rows']:,}", "不包含测试集和末段隔离区"),
    ]:
        with col:
            kpi(label, value, "", foot)
    scatter_col, residual_col = st.columns(2)
    with scatter_col, st.container(key="panel_scatter"):
        panel_title("实测值与预测值", "HELD-OUT TEST")
        plot(charts.actual_vs_predicted(test), "test_scatter")
    with residual_col, st.container(key="panel_residual"):
        panel_title("残差分布", "RESIDUAL DISTRIBUTION")
        plot(charts.residual_histogram(test), "residual_histogram")
    importance_col, score_col = st.columns([1.2, 1])
    with importance_col, st.container(key="panel_importance"):
        panel_title("变量重要性", "PERMUTATION IMPORTANCE")
        plot(charts.importance_chart(bundle["feature_importance"]), "importance")
        st.caption(metrics["importance_method"] + "；误差线为重复结果的标准差。")
    with score_col, st.container(key="panel_scores"):
        panel_title("模型选择与基线", "VALIDATION PROTOCOL")
        rows = []
        for name, scores in metrics["validation_metrics"].items():
            rows.append({"模型": name, "评估集": "验证集", "R²": scores["r2"],
                         "RMSE / 百分点": scores["rmse_rated_power_percentage_points"],
                         "MAE / 百分点": scores["mae_rated_power_percentage_points"]})
        baseline = metrics["mean_baseline_test_metrics"]
        for name, scores in [(metrics["selected_model"], evaluation), ("训练均值基线", baseline)]:
            rows.append({"模型": name, "评估集": "测试集", "R²": scores["r2"],
                         "RMSE / 百分点": scores["rmse_rated_power_percentage_points"],
                         "MAE / 百分点": scores["mae_rated_power_percentage_points"]})
        st.dataframe(pd.DataFrame(rows).round(4), hide_index=True, width="stretch")
        st.markdown("按发布序号 **60% / 20% / 20%** 分块，两处边界后各隔离 **100 行**。"
                    "验证集 RMSE 决定模型类型，选定后在前 80% 重训。测试集仅用于最终评估。")
        st.caption("样本隔离行数不代表固定时间长度。R² 不能解释为准确率或置信度。")
    st.download_button("下载完整评估报告 JSON", json.dumps(metrics, ensure_ascii=False, indent=2).encode(),
                       f"{turbine}_metrics.json", mime="application/json", key="download_metrics")
    with st.expander("偏差超过当前阈值的测试样本"):
        flagged = test[test["residual_percentage_points"].abs() > threshold]
        st.caption(f"共 {len(flagged)} / {len(test)} 条；阈值为 {threshold} 个百分点，不代表已确认设备故障。")
        st.dataframe(flagged, hide_index=True, width="stretch")
        st.download_button("下载待核查样本", csv_bytes(flagged), f"{turbine}_residual_review.csv",
                           mime="text/csv", key="download_flags")

with batch_tab:
    st.markdown("### 批量工况预测")
    st.caption(f"使用侧栏选择的 {turbine} / {mode_label}；CSV 必须包含下列 7 个原始输入列。")
    st.code(", ".join(SOURCE_FEATURES), language="text")
    example = data.loc[data[ID_COLUMN].isin(test.head(8)["source_sequence_no"]), SOURCE_FEATURES]
    st.download_button("下载输入模板 CSV", csv_bytes(example), "wind_prediction_template.csv",
                       mime="text/csv", key="batch_template")
    upload = st.file_uploader("上传工况 CSV（UTF-8，最多 10,000 行）", type=["csv"], key="batch_file")
    upload_bytes = upload.getvalue() if upload else b""
    signature = f"{context}|{hashlib.sha256(upload_bytes).hexdigest()}"
    if st.session_state.get("batch_signature") != signature:
        st.session_state["batch_result"] = None
        st.session_state["batch_signature"] = signature
    if st.button("执行批量预测", type="primary", disabled=upload is None, key="batch_predict"):
        st.session_state["batch_result"] = None
        try:
            if len(upload_bytes) > 10 * 1024 * 1024:
                raise ValueError("文件超过 10 MB。")
            # 多读取一行以识别超限，避免一次解析超大 CSV。
            frame = pd.read_csv(io.BytesIO(upload_bytes), encoding="utf-8-sig", nrows=10001)
            if len(frame) > 10000:
                raise ValueError("一次最多处理 10,000 行，请拆分文件。")
            raw = validate_features(frame)
            predictions = predict_power(raw, turbine=turbine, mode=mode,
                                        custom_model_path=CUSTOM_MODEL, bundle=bundle)
            out = raw.copy()
            out.insert(0, "input_row", np.arange(1, len(out) + 1))
            out.insert(1, "turbine", turbine)
            out["model_mode"] = mode
            out["prediction_pct_rated"] = predictions
            warning_list = extrapolation_flags(raw, bounds) if mode == "demo" else []
            if mode == "demo":
                flags = pd.DataFrame({name: (raw[name] < bounds[name]["min"]) | (raw[name] > bounds[name]["max"])
                                      for name in SOURCE_FEATURES if name != "D"})
                out["outside_demo_training_bounds"] = flags.any(axis=1)
            st.session_state["batch_result"] = {"data": out, "warnings": warning_list, "signature": signature}
        except Exception as exc:
            st.error(f"批量预测未完成：{exc}")
    batch_result = st.session_state.get("batch_result")
    if batch_result and batch_result["signature"] == signature:
        st.success(f"已完成 {len(batch_result['data']):,} 条工况预测。")
        for message in batch_result["warnings"]:
            st.warning(message)
        out = batch_result["data"].copy()
        if rated_mw is not None:
            out["assumed_rated_capacity_mw"] = rated_mw
            out["prediction_mw_from_assumption"] = out["prediction_pct_rated"] * rated_mw / 100
        abnormal = ((out["prediction_pct_rated"] < 0) | (out["prediction_pct_rated"] > 100)).sum()
        if abnormal:
            st.warning(f"{abnormal} 条预测超出 0–100%，已保留模型原始输出。")
        st.dataframe(out.head(1000), hide_index=True, width="stretch")
        st.caption("页面最多展示 1,000 行；下载包含全部预测。input_row 对应输入数据行，第一行表头不计。")
        st.download_button("下载完整批量预测 CSV", csv_bytes(out), f"{turbine}_batch_predictions.csv",
                           mime="text/csv", key="download_batch")

with data_tab:
    st.markdown("### 数据与模型说明")
    st.markdown(
        "数据来自 **Ding, Y. (2021), Inland-Offshore Wind Farm Dataset1 (v1)**。"
        "本项目选用 WT5、WT6 两台海上机组，保留原始 CSV 数据。"
        "[官方数据集与 DOI](https://doi.org/10.5281/zenodo.5516552) · "
        "[许可 CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)"
    )
    st.info("任务口径：同一观测工况下的相对功率回归。发布 CSV 没有真实日期时间，不能据此声明未来 10 分钟或日前预测。")
    dictionary = [
        (ID_COLUMN, "样本序号", "仅识别与分块划分，不进入模型"),
        ("V", "风速", "m/s"), ("D", "风向", "角度；内置模型用 sin/cos 编码"),
        ("air density", "空气密度", "kg/m³"),
        ("humidity", "湿度原始记录", "公开说明未明确单位，不改名为相对湿度百分比"),
        ("I", "湍流强度", "无量纲"), ("S_a", "轮毂上方风切变指数", "无量纲"),
        ("S_b", "轮毂下方风切变指数", "无量纲"),
        (TARGET, "实测相对功率 / 预测目标", "额定功率百分比；60 表示额定功率的 60%"),
    ]
    st.dataframe(pd.DataFrame(dictionary, columns=["原始字段", "含义", "单位 / 处理"]),
                 hide_index=True, width="stretch")
    st.markdown("#### 使用范围")
    st.markdown("""
- 原始观测为 10 分钟统计量，但发布序号存在缺号，不能把相邻行当作连续 10 分钟。
- 图表来自实测样本和已训练模型，不包含虚构实时告警、发电量、地理位置或故障标签。
- 两台机组分别训练和评估；没有将共享气象观测混合后随机拆分。
- 数据不覆盖全部停机与极端工况。少量功率高于 100% 的原值予以保留。
- 预测输出不自动截断；RMSE / MAE 单位为额定功率的百分点。
- 置换重要性用于模型解释，不等同于 SHAP，也不用于推断因果关系。
- 自定义模型仅从可信本地文件加载；其输入单位、预处理、训练范围和评估需由模型提供者确认。
""")
    st.markdown("#### 当前机组数据预览")
    st.dataframe(data.head(300), hide_index=True, width="stretch", height=320)
    st.download_button(f"下载 {turbine} 完整原始数据", (ROOT / "data" / f"{turbine}.csv").read_bytes(),
                       f"{turbine}.csv", mime="text/csv", key="download_raw")
    with st.expander("训练输入范围（内置示例模型）"):
        st.dataframe(pd.DataFrame(bounds).T.rename(columns={"min": "最小值", "max": "最大值", "median": "中位数"}),
                     width="stretch")
    st.caption("参考：Lee, Ding, Genton & Xie (2015), Power Curve Estimation With Multivariate Environmental Factors "
               "for Inland and Offshore Wind Farms. DOI: 10.1080/01621459.2014.977385。")

st.html('<div class="footer-strip"><span>DSWE OPEN DATA · 历史工况展示 · 同期相对功率回归</span>'
        '<span>Streamlit + Plotly + scikit-learn　|　可替换模型接口</span></div>')
