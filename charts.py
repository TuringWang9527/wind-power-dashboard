"""Plotly 图表：所有坐标、单位和数据范围均显式标注。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from wind_core import ID_COLUMN, TARGET

CYAN = "#42e8d0"
BLUE = "#67a8ff"
ORANGE = "#ffc078"
MUTED = "#8ea7bc"
GRID = "rgba(133,170,197,.12)"
FONT = "Microsoft YaHei, Noto Sans CJK SC, Arial, sans-serif"


def style(fig: go.Figure, height: int = 250) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=48, r=16, t=32, b=45),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, size=11, color=MUTED),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0,
                    font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="#153043", bordercolor="#2e6571", font=dict(color="#e8f7ff")),
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False, showline=False, title_font_size=10,
                     automargin=True, title_standoff=8, tickfont_size=10)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, showline=False, title_font_size=10,
                     automargin=True, title_standoff=7, tickfont_size=10)
    return fig


def power_curve(data: pd.DataFrame) -> go.Figure:
    sample = data.sample(min(1600, len(data)), random_state=42).sort_values("V")
    fig = go.Figure(go.Scatter(
        x=sample["V"], y=sample[TARGET], mode="markers", name="实测样本",
        marker=dict(size=3, color=CYAN, opacity=.26),
        hovertemplate="风速 %{x:.2f} m/s<br>实测 %{y:.2f}%<extra></extra>",
    ))
    grouped = data.assign(speed_bin=np.floor(data["V"] * 2) / 2).groupby("speed_bin")
    bins = grouped.agg(wind=("V", "mean"), power=(TARGET, "median"), n=("V", "size"))
    bins = bins[bins["n"] >= 10]
    fig.add_trace(go.Scatter(
        x=bins["wind"], y=bins["power"], mode="lines", name="分箱中位数",
        line=dict(color=ORANGE, width=2), hovertemplate="风速 %{x:.2f}<br>中位数 %{y:.2f}%<extra></extra>",
    ))
    style(fig, 238)
    fig.update_xaxes(title="风速 / m·s⁻¹")
    fig.update_yaxes(title="额定功率 / %", rangemode="tozero")
    return fig


def wind_rose(data: pd.DataFrame) -> go.Figure:
    """16 方位、5 风速区间；径向为全部样本中的频率。"""
    sector = np.floor(((data["D"].to_numpy() + 11.25) % 360) / 22.5).astype(int)
    categories = pd.cut(data["V"], [0, 5, 8, 11, 14, np.inf],
                        labels=["<5", "5–8", "8–11", "11–14", "≥14"], right=False)
    colors = ["#23485d", "#306b88", "#4c9ab0", "#42caba", "#b0f5d7"]
    fig = go.Figure()
    for label, color in zip(categories.cat.categories, colors):
        counts = np.bincount(sector[(categories == label).to_numpy()], minlength=16)
        fig.add_trace(go.Barpolar(
            r=counts / len(data) * 100, theta=np.arange(16) * 22.5, width=20,
            name=f"{label} m/s", marker_color=color, marker_line_width=0,
            hovertemplate="方向 %{theta}°<br>样本频率 %{r:.2f}%<extra>%{fullData.name}</extra>",
        ))
    style(fig, 228)
    fig.update_layout(
        barmode="stack", showlegend=False,
        polar=dict(bgcolor="rgba(0,0,0,0)",
                   radialaxis=dict(ticksuffix="%", gridcolor=GRID, linecolor=GRID,
                                   tickfont=dict(size=9), nticks=4, angle=45),
                   angularaxis=dict(rotation=90, direction="clockwise", gridcolor=GRID,
                                    linecolor=GRID, tickvals=[0, 90, 180, 270],
                                    ticktext=["N", "E", "S", "W"])),
        margin=dict(l=32, r=32, t=24, b=24),
    )
    return fig


def sequence_chart(test: pd.DataFrame, threshold: float = 20.) -> go.Figure:
    fig = go.Figure()
    for column, name, color in [
        ("actual_power_pct_rated", "实测功率", CYAN),
        ("predicted_power_pct_rated", "示例模型", ORANGE),
    ]:
        fig.add_trace(go.Scatter(
            x=test["source_sequence_no"], y=test[column], mode="lines", name=name,
            line=dict(color=color, width=1.8),
            hovertemplate="样本 %{x}<br>功率 %{y:.2f}%<extra>%{fullData.name}</extra>",
        ))
    flagged = test[test["residual_percentage_points"].abs() > threshold]
    fig.add_trace(go.Scatter(
        x=flagged["source_sequence_no"], y=flagged["actual_power_pct_rated"],
        mode="markers", name="偏差待核查", marker=dict(color="#f99292", size=5, symbol="circle-open"),
        hovertemplate="样本 %{x}<br>实测 %{y:.2f}%<extra>超过所选偏差阈值</extra>",
    ))
    style(fig, 253)
    fig.update_xaxes(title="发布样本序号（非时间轴）", tickformat="d")
    fig.update_yaxes(title="额定功率 / %", rangemode="tozero")
    return fig


def actual_vs_predicted(test: pd.DataFrame) -> go.Figure:
    sample = test.sample(min(1800, len(test)), random_state=42)
    lo = float(min(test["actual_power_pct_rated"].min(), test["predicted_power_pct_rated"].min()))
    hi = float(max(test["actual_power_pct_rated"].max(), test["predicted_power_pct_rated"].max()))
    fig = go.Figure(go.Scatter(
        x=sample["actual_power_pct_rated"], y=sample["predicted_power_pct_rated"],
        mode="markers", marker=dict(size=4, color=CYAN, opacity=.42), name="测试样本",
        hovertemplate="实测 %{x:.2f}%<br>预测 %{y:.2f}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="理想预测",
                             line=dict(color=ORANGE, width=1.5, dash="dash")))
    style(fig, 310)
    fig.update_xaxes(title="实测额定功率 / %")
    fig.update_yaxes(title="预测额定功率 / %")
    return fig


def residual_histogram(test: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Histogram(
        x=test["residual_percentage_points"], nbinsx=40,
        marker=dict(color=BLUE, line_width=0), opacity=.82, name="残差",
        hovertemplate="残差区间 %{x}<br>样本数 %{y}<extra></extra>",
    ))
    style(fig, 310)
    fig.update_layout(showlegend=False, bargap=.08)
    fig.add_vline(x=0, line_width=1, line_dash="dash", line_color=CYAN)
    fig.update_xaxes(title="实测 − 预测 / 额定功率百分点")
    fig.update_yaxes(title="样本数")
    return fig


def importance_chart(importance: pd.DataFrame) -> go.Figure:
    data = importance.sort_values("importance")
    fig = go.Figure(go.Bar(
        x=data["importance"], y=data["label"], orientation="h",
        marker_color=[CYAN if x >= 0 else ORANGE for x in data["importance"]],
        error_x=dict(type="data", array=data["std"], color="#9ab5c6", thickness=1),
        hovertemplate="%{y}<br>RMSE 增量 %{x:.3f} 个百分点<extra></extra>",
    ))
    style(fig, 310)
    fig.update_layout(showlegend=False, margin=dict(l=125, r=25, t=15, b=48))
    fig.update_xaxes(title="置换后 RMSE 增量 / 额定功率百分点")
    return fig


def wind_sensitivity(raw: pd.DataFrame, predict_fn, bounds: dict) -> go.Figure:
    """固定其余输入，只改变风速，展示已提交工况的模型响应。"""
    wind = np.linspace(bounds["V"]["min"], bounds["V"]["max"], 75)
    cases = pd.concat([raw] * len(wind), ignore_index=True)
    cases["V"] = wind
    values = predict_fn(cases)
    fig = go.Figure(go.Scatter(x=wind, y=values, mode="lines", name="工况响应",
                              line=dict(color=CYAN, width=2.5), fill="tozeroy",
                              fillcolor="rgba(66,232,208,.08)"))
    current = float(predict_fn(raw)[0])
    fig.add_trace(go.Scatter(x=[float(raw.iloc[0]["V"])], y=[current], mode="markers",
                             name="已提交工况", marker=dict(size=9, color=ORANGE)))
    style(fig, 255)
    fig.update_xaxes(title="风速 / m·s⁻¹")
    fig.update_yaxes(title="预测额定功率 / %")
    return fig
