<!-- markdownlint-disable -->

<a href="../src/alphanet/data.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `alphanet.data`
多维多时间序列神经网络滚动训练数据工具箱. 

version:0.3 

author: Congyu Wang 

date: 2021-07-26 



---

<a href="../src/alphanet/data.py#L17"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `TimeSeriesData`
单个时间序列信息. 



**Notes:**

> 储存个股的数据信息及预测目标，全部使用numpy，日期格式为整数: ``YYYYMMDD``. 

<a href="../src/alphanet/data.py#L25"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(dates: ndarray, data: ndarray, labels: ndarray)
```

储存个股的数据信息及预测目标，全部使用numpy，日期格式为整数: ``YYYYMMDD``. 



**Args:**
 
 - <b>`dates`</b>:  日期列, 1D ``numpy.ndarray`` 
 - <b>`data`</b>:  训练输入的X，2D ``numpy.ndarray``, (日期长度 x 特征数量) 
 - <b>`labels`</b>:  训练标签Y, 1D ``numpy.ndarray``, 长度与dates相同 





---

<a href="../src/alphanet/data.py#L53"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `TrainValData`
该类用于生成不同训练阶段的tensorflow dataset. 

<a href="../src/alphanet/data.py#L56"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    time_series_list: List[TimeSeriesData],
    train_length: int = 1200,
    validate_length: int = 300,
    history_length: int = 30,
    train_val_gap: int = 10,
    sample_step: int = 2,
    fill_na: float = nan
)
```

用于获取不同阶段的训练集和验证集. 



**Notes:**

> 储存全部的时间序列信息，通过get(start_date)方法获取从start_date 开始的训练机和验证集。 
>train_val_gap参数为验证集第一天与训练集最后一天中间间隔的天数， 如果是相临，则train_val_gap = 0。 
>设置该参数的目的如下： 
>如果希望预测未来十天的累计收益，则预测时用到的输入数据为最近的历史数据来预测 未来十天的累计收益，即用t(-history)到t(0)的数据来预测t(1)到t(11)的累计收益 而训练时因为要用到十天累计收益做标签，最近的一个十天累计收益是从t(-10)到t(0)， 用到的历史数据则必须是t(-history-11)到t(-11)的数据，为了确保validation的 有效性，则最好将validation的第一个数据位置与train的最后一个数据位置在时间上 相差11天，即间隔10天，因此使用train_val_gap=10。 
>时间为t(0)的每个样本数据的历史数据时间范围为t(-30)至t(0)， 即用到了当天的收盘数据。标签数据为 p(11) / p(1) - 1， 即往后11天的收盘价除以明天的收盘价减去1。不用 p(10) / p(0) - 1 的原因是， 当天收盘时未做预测，不能以当天收盘价购入。 
>

**Args:**
 
 - <b>`time_series_list`</b>:  TimeSeriesData 列表 
 - <b>`train_length`</b>:  训练集天数 
 - <b>`validate_length`</b>:  验证集天数 
 - <b>`history_length`</b>:  每个sample的历史天数 
 - <b>`train_val_gap`</b>:  训练集与验证集的间隔 
 - <b>`sample_step`</b>:  采样sample时步进的天数 
 - <b>`fill_na`</b>:  默认填充为np.NaN，训练时会跳过有确实数据的样本 




---

<a href="../src/alphanet/data.py#L165"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get`

```python
get(start_date: int, order='by_date', mode='in_memory', validate_only=False)
```

获取从某天开始的训练集和验证集. 



**Notes:**

> 根据设定的训练集天数以及验证集天数，从start_date开始获取正确的 训练集以及验证集，以及他们各自的日期范围信息(该信息以字典形式返回)。 
>training set 的开始和结束是指其data, label所有的时间范围(inclusive)。 validation set 的开始和结束则只是指其label的时间范围， 因为validation set允许用到training set内的X历史数据。 具体时间信息参考返回的第三个元素dates_info。 
>如果日期范围超出最大日期会报ValueError。 
>

**Args:**
 
 - <b>`start_date`</b>:  该轮训练开始日期，整数``YYYYMMDD`` 
 - <b>`order`</b>:  有三种顺序 ``shuffle``, ``by_date``, ``by_series``。  分别为随机打乱股票和时间，按时间顺序优先，按股票顺序优先，默认by_date。 
 - <b>`mode`</b>:  `generator` 或 `in_memory`. generator 速度极慢，  in_memory速度较快，默认in_memory。feature、series数量大内存不足时  可以使用generator。'in_memory'模式股票数量较大以及step较小时，  可能会要求较大显卡内存。 
 - <b>`validate_only`</b>:  如果设置为True，则只返回validate set  和训练集、验证集时间信息 



**Returns:**
 如果``validate_only=False``，返回训练集、验证集，以及日期信息： (train, val, dates_info(dict))。如果为``True``，则返回 验证集以及日期信息。 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
