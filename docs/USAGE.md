# 海上风电智慧大屏 · Streamlit 完整项目

这是一个可在本地运行、可替换预测模型的中文风电展示项目。内含 WT5、WT6 两台真实海上风机的公开数据，以及训练好的示例模型；首次运行即可浏览图表、输入工况并执行预测。页面使用深蓝与青绿色主题，适合模型汇报、课程演示和项目作品展示。

## 1. 安装与启动

建议使用 **Python 3.11 或 3.12**。本项目在 Python 3.12、Streamlit 1.50.0、Plotly 6.3.0、scikit-learn 1.8.0 环境验证。请先解压整个项目，并在 `app.py` 所在目录打开终端。

最少只需两条命令：

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

浏览器访问 **http://localhost:8501**。命令行显示的地址仅在运行这份程序的电脑及其允许访问的网络中生效，项目没有自动发布到公网。

推荐创建独立环境。Windows PowerShell：

```powershell
cd wind_dashboard
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

macOS / Linux：

```bash
cd wind_dashboard
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m streamlit run app.py
```

安装完成后，也可以双击 `start_windows.bat`，或执行 `bash start.sh`。脚本优先使用项目内 `.venv`。展示时可收起侧栏、按 F11 全屏；根据屏幕尺寸将浏览器缩放调整到 85%–100%。

## 2. 页面功能

- **运行总览**：切换 WT5 / WT6，查看样本指标、机组示意、风速与功率关系、风况和样本序列。
- **主屏右侧的功率预测面板**：填写七项环境变量，点击“执行功率预测”，调用真实模型输出相对功率；模型接入位置集中在 `model_adapter.py`。
- **模型评估**：查看独立留出集的 R²、RMSE、MAE、预测对比、残差与置换重要性。
- **批量预测**：上传含七个输入字段的 CSV，执行预测并下载结果。
- **数据与说明**：查看数据字段、来源、范围与使用方法。

右侧工况表单属于一次完整预测请求。修改输入后，需要再次点击预测按钮；已有预测结果对应的是上一次提交的工况。自定义模型模式若缺少模型文件，会提示缺失，不会生成随机结果或自动改用其他模型。

页面中的风机图案是结构示意，不表示公开数据中未披露的实际场址。偏差提示只说明“实测功率与模型估计存在差异”，不能据此确认机械故障。

## 3. 预测输入与输出

输入必须为下列七列。批量 CSV 可以带其他列，预测接口只按固定顺序提取这七列；导出文件使用 input_row 对应输入数据行。一次最多上传 10 MB、10,000 行，页面预览最多显示 1,000 行，下载包含全部结果。

| 原始列名 | 含义 | 单位 / 约定 |
| --- | --- | --- |
| `V` | 风速 | m/s |
| `D` | 风向 | 0–360° |
| `air density` | 空气密度 | kg/m³ |
| `humidity` | 湿度原始记录 | 来源未明确单位，保持原值 |
| `I` | 湍流强度 | 无量纲 |
| `S_a` | 轮毂上方风切变指数 | 无量纲 |
| `S_b` | 轮毂下方风切变指数 | 无量纲 |

`Sequence No.` 是样本标识，不进入模型。`y (% relative to rated power)` 是目标列，也不进入模型输入。

**模型输出为“占额定功率的百分比”**，例如 `61.2` 表示额定功率的 61.2%，不是 61.2 MW。需要展示 MW 时，由使用者提供额定容量，再按 `MW = 百分比 / 100 × 额定容量 MW` 换算；该容量是使用者指定的演示假设，并非数据集披露值。

原数据少量功率略高于 100%，本项目保留原值，不截断预测输出。内置示例模型的输入超出训练变量范围时会提示；自定义模型未提供训练范围，因此不会沿用示例模型的范围判断。处于范围内也不等于获得工程可靠性保证。

## 4. 接入自己的模型

页面统一调用以下接口：

```python
from model_adapter import predict_power
from wind_core import SOURCE_FEATURES

# raw_df 必须包含七个原始字段，不要事先重复进行风向 sin/cos 编码。
predicted_pct = predict_power(
    raw_df[SOURCE_FEATURES],
    turbine="WT6",
    mode="demo",
)
```

### 使用自己的 sklearn / AutoML 模型

将**已训练好、包含所需预处理、支持 `predict(DataFrame)`** 的模型或 Pipeline 保存为：

```python
import joblib

joblib.dump(trained_pipeline, "models/custom_model.joblib")
```

这里的 `trained_pipeline` 是你已有的训练结果，并非本项目中预定义的变量。它必须接收七个原始字段，并为每条输入返回一个有限数值，单位为额定功率百分比。若原模型输出 0–1，需要在接入层显式乘 100；若原模型需要标准化、类别编码或不同特征顺序，应将这些步骤一并封装进 Pipeline。

随后在界面切换到自定义模型。也可以直接调用：

```python
predicted_pct = predict_power(
    raw_df[SOURCE_FEATURES],
    turbine="WT6",
    mode="custom",
    custom_model_path="models/custom_model.joblib",
)
```

对于模型 API、PyTorch 或其他接口，在 `model_adapter.py` 的 `predict_power()` 中增加自己的调用分支，保持输入和输出约定即可。该文件是预留的模型接口；图表和页面无需随模型算法一起修改。

仅加载你本人训练或本项目提供的可信本地 joblib 文件。项目未提供网页上传模型文件的入口。预置评估指标属于各机组的示例模型，接入自定义模型后不能将它们当作新模型的验证结果。

## 5. 示例模型与复现

重新训练单台或全部机组：

```bash
python train_model.py --turbine WT6
python train_model.py --turbine WT5
python train_model.py --turbine all
```

可选参数为 `--gap 100`、`--data-dir data`、`--model-dir models`。输出位于 `models/`：模型包、指标 JSON 和测试集预测 CSV。重训完成后重启 Streamlit，以清除旧模型缓存。

训练流程按样本序号排序，前 60% 用于候选训练，随后 20% 用于验证，末段 20% 用于测试；两个划分边界后各隔离 100 行。候选模型为 RandomForest 与 HistGradientBoosting，以验证 RMSE 选择模型，再以前 80% 重训。测试集不参与选择。七个原始变量经过风向 sin/cos 编码后形成八个数值特征。

随包模型的实际留出集结果如下：

| 机组 | 验证集选出的模型 | 测试 R² | 测试 RMSE | 测试 MAE |
| --- | --- | ---: | ---: | ---: |
| WT6 | RandomForest | 0.887393 | 11.060115 | 7.819567 |
| WT5 | HistGradientBoosting | 0.967775 | 6.034168 | 4.251598 |

**RMSE / MAE 的单位为额定功率的百分点**，R² 无量纲。完整划分数量、版本与数据 SHA256 记录在 `models/*_metrics.json`。两机组的测试样本不同，不能只根据这张表推断哪种算法普遍更优。

重要性图采用测试集固定抽样的**置换重要性**，表示打乱某个输入后 RMSE 的增量，重复三次；它不是 SHAP，也不构成因果解释。报告使用留出集解释，不据此继续调参。

## 6. 数据来源与任务边界

- 数据集：[Ding, Y. (2021), Inland-Offshore Wind Farm Dataset1 (v1)](https://doi.org/10.5281/zenodo.5516552)。[官方记录](https://zenodo.org/records/5516552)。
- 许可：[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)。本项目保留作者、标题、DOI 和来源元数据。
- 本包只包含原数据中的海上 WT5（16,443 行）和 WT6（19,041 行），共 35,484 行；CSV 内容保持原样，仅简化文件名。
- [字段与功率曲线研究原文](https://marcgenton.github.io/2015.LDGX.JASA.pdf)：Lee, Ding, Genton & Xie (2015), DOI [10.1080/01621459.2014.977385](https://doi.org/10.1080/01621459.2014.977385)。

**本项目估计同一观测工况下的功率。** 原始观测为十分钟统计值，但发布 CSV 没有真实日期时间，序号有缺号。因此横轴使用样本序号，不能把相邻行解释为连续十分钟，也不能将这个演示称为未来一小时、日前预测或实时风场监控。

数据不含故障事件、设备可用率、实际额定 MW 容量、经纬度或完整停机工况。项目不会据此计算实际累计发电量或确认故障类别。两个机组共享部分气象观测，本项目按单台机组建模，避免将同一气象观测简单随机混入训练与测试。

## 7. 文件说明与常见问题

| 文件 / 目录 | 用途 |
| --- | --- |
| `app.py` | Streamlit 页面与交互 |
| `charts.py` | Plotly 图表 |
| `assets/style.css` | 大屏样式 |
| `wind_core.py` | 数据检查、特征、训练与指标 |
| `model_adapter.py` | 统一预测接口与自定义模型载入 |
| `train_model.py` | 本地模型重训命令 |
| `data/` | 两台机组 CSV 与来源信息 |
| `models/` | 预训练模型、评估报告与测试预测 |
| `.streamlit/config.toml` | 主题与运行配置 |
| `requirements.txt` | 已验证的依赖版本 |

**提示找不到 Streamlit：** 用同一个 Python 运行 `python -m pip install -r requirements.txt` 与 `python -m streamlit run app.py`，不要混用多个环境。

**提示模型与数据不一致：** 检查是否替换了 `data/WT5.csv` 或 `data/WT6.csv`。若有意更换数据，请保留字段约定后重训，并重新启动应用；不要只替换 CSV 而沿用旧指标。

**模型包缺失：** 程序可在内存中训练示例模型，首次载入会较慢。正式展示前建议运行训练脚本，把模型保存好。

**端口被占用：** 使用 `python -m streamlit run app.py --server.port 8502`，并打开终端提示的新地址。

## 8. 验证

在项目目录运行 `python -m unittest -v test_dashboard.py`，可检查默认启动、单次真实预测、修改输入、机组切换和自定义模型缺失处理。项目还经过浏览器中的单次预测、CSV 上传、批量预测、结果下载及缺失列拒绝检查。
