<!-- markdownlint-disable -->

# API Overview

## Modules

- [`alphanet`](./alphanet.md#module-alphanet): 时间序列计算层、神经网络模型定义.
- [`alphanet.data`](./alphanet.data.md#module-alphanetdata): 多维多时间序列神经网络滚动训练的数据工具.
- [`alphanet.metrics`](./alphanet.metrics.md#module-alphanetmetrics): 训练中的辅助准确率信息.

## Classes

- [`alphanet.AlphaNetV2`](./alphanet.md#class-alphanetv2): 神经网络模型，继承``keras.Model``类.
- [`alphanet.AlphaNetV3`](./alphanet.md#class-alphanetv3): 神经网络模型，继承``keras.Model``类.
- [`alphanet.AlphaNetV4`](./alphanet.md#class-alphanetv4): 神经网络模型，继承``keras.Model``类.
- [`alphanet.Correlation`](./alphanet.md#class-correlation): 计算每个stride各时间序列的相关系数.
- [`alphanet.Covariance`](./alphanet.md#class-covariance): 计算每个stride各时间序列片段的covariance.
- [`alphanet.FeatureExpansion`](./alphanet.md#class-featureexpansion): 计算时间序列特征扩张层，汇总6个计算层.
- [`alphanet.LinearDecay`](./alphanet.md#class-lineardecay): 计算每个序列各stride的线性衰减加权平均.
- [`alphanet.Return`](./alphanet.md#class-return): 计算每个序列各stride的回报率.
- [`alphanet.Std`](./alphanet.md#class-std): 计算每个序列各stride的标准差.
- [`alphanet.ZScore`](./alphanet.md#class-zscore): 计算每个序列各stride的均值除以其标准差.
- [`data.TimeSeriesData`](./alphanet.data.md#class-timeseriesdata): 单个时间序列信息.
- [`data.TrainValData`](./alphanet.data.md#class-trainvaldata): 根据训练天数、验证天数、样本历史长度、训练起点生成不同训练阶段的数据.
- [`metrics.UpDownAccuracy`](./alphanet.metrics.md#class-updownaccuracy): 通过对return的预测来计算涨跌准确率.

## Functions

- [`alphanet.load_model`](./alphanet.md#function-load_model): 用于读取已存储的模型，可识别自定义metric: UpDownAccuracy.


---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
