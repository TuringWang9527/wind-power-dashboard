# 数据来源与引用

## 原始数据

- 作者：Ding, Yu。
- 标题：Inland-Offshore Wind Farm Dataset1。
- 版本：v1，2021 年发布。
- DOI：<https://doi.org/10.5281/zenodo.5516552>。
- 官方记录：<https://zenodo.org/records/5516552>。
- 许可：Creative Commons Attribution 4.0 International（CC BY 4.0），<https://creativecommons.org/licenses/by/4.0/>。
- 配套数据页：<https://aml.engr.tamu.edu/book-dswe/dswe-datasets/>。

建议引用：

> Ding, Y. (2021). Inland-Offshore Wind Farm Dataset1 (v1) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.5516552

本仓库仅选取原数据的两台海上机组，将文件名简化为 `WT5.csv` 与 `WT6.csv`，数据内容字节保持原样。保留少量超过 100% 的功率原值，没有补造日期或实际额定容量。官方元数据保存在 `data/source_metadata.json`。

数据作者及发布机构未被表示为本大屏或模型的认可方。页面、模型代码、评估结果和模型文件是基于数据另外构建的演示内容，不代表原作者的性能结论。CC BY 4.0 声明针对引用的数据，不替代用户对其他内容的许可安排。

## 字段定义参考

Lee, G., Ding, Y., Genton, M. G., & Xie, L. (2015). Power Curve Estimation With Multivariate Environmental Factors for Inland and Offshore Wind Farms. *Journal of the American Statistical Association*, 110(509).

- DOI：<https://doi.org/10.1080/01621459.2014.977385>。
- 作者公开论文：<https://marcgenton.github.io/2015.LDGX.JASA.pdf>。

输入为风速、风向、空气密度、湿度原始记录、湍流强度和上下部风切变指数。湿度单位在公开说明中未明确，保留原值。输出为额定功率百分比，RMSE / MAE 为额定功率百分点。

公开 CSV 没有真实日期时间，序号存在缺号。此演示为同期工况功率回归，不能称为已验证的未来时点预测；数据不含故障标签、实际场址坐标或完整停机工况。
