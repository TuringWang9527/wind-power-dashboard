# 海上风电智慧大屏 · GitHub 仓库版

基于 **Streamlit + Plotly + scikit-learn** 的中文风电展示项目。内含 WT5、WT6 海上风机公开数据、训练好的模型和完整预测按钮，可直接上传 GitHub 并部署到 Streamlit Community Cloud。

**入口：`app.py` ｜ Python：`3.12` ｜ 默认无需 API Key**

## 1. 上传 GitHub

1. 新建仓库，例如 `wind-power-dashboard`。
2. 解压项目，在仓库选择 **Add file → Upload files**。
3. 上传解压后的全部文件和文件夹，保留目录结构。仓库首页应直接看到 `app.py`、`requirements.txt`、`README.md`、`data/`、`models/` 和 `assets/`。
4. 同时上传 `.streamlit/`、`.github/`、`.gitattributes` 和 `.gitignore`。如果点号开头的文件被隐藏，请先显示隐藏文件。
5. 提交到 `main`，或记住自己实际使用的分支名。

压缩包用于传递项目；GitHub 仓库中需要保存解压后的源文件。项目最大文件约 14.23 MB，低于 GitHub 网页上传的每文件 25 MiB 上限，本包无需配置 Git LFS。

## 2. 在线部署

打开 [Streamlit Community Cloud](https://share.streamlit.io/)，登录并连接你的 GitHub 仓库，点击 **Create app**，填写：

| 设置 | 内容 |
| --- | --- |
| Repository | 刚上传项目的仓库，例如 `你的用户名/wind-power-dashboard` |
| Branch | `main`，或代码所在实际分支 |
| Main file path | **`app.py`** |
| Advanced settings → Python version | **`3.12`** |
| Secrets | 默认留空 |

点击 **Deploy**。平台根据 `requirements.txt` 安装依赖并启动应用，成功后分配 `streamlit.app` 访问地址。

GitHub 负责保存源码，Streamlit Community Cloud 负责运行 Python 大屏。GitHub Pages 是静态站点托管，不能直接执行本项目的 Streamlit 后端。项目未绑定任何用户名、仓库或线上域名；上传和部署由仓库所有者完成。

## 3. 功能

- 深蓝与青绿色智慧大屏，WT5 / WT6 切换。
- 样本指标、风速—功率曲线、风向玫瑰图、风机结构示意和测试样本序列。
- **右侧“执行功率预测”按钮**，输入七项环境参数后调用真实模型，并可下载结果。
- 上传 CSV 批量预测：最多 10 MB、10,000 行，支持完整结果导出。
- 独立测试集 R²、RMSE、MAE、残差图和置换重要性。
- `model_adapter.py` 提供可替换的模型接口。

默认数据和预训练模型全部随仓库提供，启动时可以直接读取。完整操作方法见 [使用说明](docs/USAGE.md)。

## 4. 接入自己的模型

输入原始字段：

```python
["V", "D", "air density", "humidity", "I", "S_a", "S_b"]
```

输出为一维额定功率百分比，例如 `61.2` 表示额定功率的 61.2%。你训练的模型应支持 `predict(DataFrame)`，并包含所需预处理。

将模型保存并上传为 `models/custom_model.joblib`，在侧栏选择“自定义本地模型”。云端的本地文件指随 GitHub 仓库提供的文件。只加载自己训练或可信来源的模型。

`.gitignore` 默认忽略自定义模型。若使用 Git 命令，并已决定把自己的模型提交到仓库，可明确执行：

```bash
git add -f models/custom_model.joblib
```

GitHub 网页上传不依赖本地 `.gitignore`。上传模型后，必要时在 Streamlit Cloud 管理界面重启应用。自定义模型不会自动继承示例模型的评估指标。

## 5. 项目结构

| 文件 / 目录 | 用途 |
| --- | --- |
| `app.py` | Streamlit 页面入口 |
| `charts.py`、`assets/` | 图表、样式和风机 SVG |
| `model_adapter.py` | 模型接入与预测 |
| `wind_core.py`、`train_model.py` | 数据检查、特征、训练与评估 |
| `data/`、`models/` | 原始数据、模型和评估结果 |
| `requirements.txt` | 固定的依赖版本 |
| `.streamlit/config.toml` | 大屏主题和上传配置 |
| `.gitattributes` | 保留 CSV 和 joblib 字节，避免换行转换影响校验 |
| `.github/workflows/check.yml` | GitHub Actions 自动检查 |
| `scripts/check_repository.py` | 文件完整性、数据 SHA256 和上传大小检查 |
| `docs/USAGE.md` | 完整本地运行和模型使用说明 |
| `DATA_SOURCES.md` | 数据来源、许可和引用 |

## 6. 本地运行和验证（可选）

在仓库根目录运行：

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

验证和重训：

```bash
python scripts/check_repository.py
python -m unittest -v test_dashboard.py
python train_model.py --turbine all
```

重训后重启应用。GitHub Actions 在 push / pull request 后运行检查，也支持手动触发；它负责验证，网页部署由 Streamlit Community Cloud 完成。

随包版本已在本地 Python 3.12 通过文件检查和四项交互测试。GitHub Actions 与 Streamlit Cloud 的实际运行结果应在你上传部署后查看。

## 7. 数据口径

数据来自 [DSWE Inland-Offshore Wind Farm Dataset1](https://doi.org/10.5281/zenodo.5516552)，许可 CC BY 4.0，原作者与引用见 [DATA_SOURCES.md](DATA_SOURCES.md)。WT5 有 16,443 行，WT6 有 19,041 行。

发布 CSV 没有真实时间戳。此演示开展同期环境工况下的相对功率回归，不能称为已验证的未来一小时或日前预测。未披露的实际场址、额定 MW 容量与故障标签不作补造。

| 机组 | 示例模型 | 测试 R² | RMSE / 额定功率百分点 | MAE / 额定功率百分点 |
| --- | --- | ---: | ---: | ---: |
| WT6 | RandomForest | 0.887393 | 11.060115 | 7.819567 |
| WT5 | HistGradientBoosting | 0.967775 | 6.034168 | 4.251598 |

完整划分、环境与结果见 `models/*_metrics.json`。

## 常见问题

**找不到 app.py：** 确认入口位于仓库根目录。若多套一层文件夹，推荐将内容移到根目录，并保持 `.streamlit/config.toml` 位于根目录。

**缺少模块：** 确认根目录 `requirements.txt` 已上传，Python 设置为 3.12。

**数据与模型不一致：** CSV 可能被修改或换行格式改变。恢复原始文件，或有意更换数据后重新训练。`.gitattributes` 会在 Git 工作流中保留 CSV 字节。

**自定义模型不存在：** 使用默认“内置示例模型”，或将自己的模型文件一并上传。

## 官方文档

- [GitHub 网页上传](https://docs.github.com/en/repositories/working-with-files/managing-files/adding-a-file-to-a-repository)
- [Streamlit 部署与 Python 版本](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)
- [Streamlit 仓库结构](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/file-organization)
- [Streamlit 应用依赖](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies)
