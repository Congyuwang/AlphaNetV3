<!-- markdownlint-disable -->

# API Overview

## Modules

- [`alphanet`](./alphanet.md#module-alphanet): 复现华泰金工 alpha net V3 版本.
- [`alphanet.data`](./alphanet.data.md#module-alphanetdata): 多维多时间序列神经网络滚动训练数据工具箱.
- [`alphanet.metrics`](./alphanet.metrics.md#module-alphanetmetrics): 训练的计量信息.

## Classes

- [`alphanet.AlphaNetV3`](./alphanet.md#class-alphanetv3): alpha net v3版本模型.
- [`alphanet.Correlation`](./alphanet.md#class-correlation): 每个stride各个时间序列的相关系数.
- [`alphanet.Covariance`](./alphanet.md#class-covariance): 每个stride各个时间序列的covariance.
- [`alphanet.FeatureExpansion`](./alphanet.md#class-featureexpansion): 时间序列特征扩张层.
- [`alphanet.LinearDecay`](./alphanet.md#class-lineardecay): 每个序列各个stride的线性衰减加权平均.
- [`alphanet.Return`](./alphanet.md#class-return): 每个序列各个stride的回报率.
- [`alphanet.Std`](./alphanet.md#class-std): 每个序列各个stride的标准差.
- [`alphanet.ZScore`](./alphanet.md#class-zscore): 每个序列各个stride的均值除以其标准差.
- [`data.TimeSeriesData`](./alphanet.data.md#class-timeseriesdata): 单个时间序列信息.
- [`data.TrainValData`](./alphanet.data.md#class-trainvaldata): 该类用于生成不同训练阶段的tensorflow dataset.
- [`metrics.UpDownAccuracy`](./alphanet.metrics.md#class-updownaccuracy): 通过对return的预测来计算涨跌准确率.

## Functions

- [`alphanet.load_model`](./alphanet.md#function-load_model): 包装``tf.kreas``的``load_model``，添加``UpDownAccuracy``.


---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
