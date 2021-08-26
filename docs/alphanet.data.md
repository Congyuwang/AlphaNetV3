<!-- markdownlint-disable -->

<a href="../src/alphanet/data.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `alphanet.data`
多维多时间序列神经网络滚动训练的数据工具. 

version: 0.0.19 

author: Congyu Wang 

date: 2021-08-26 



---

<a href="../src/alphanet/data.py#L18"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `TimeSeriesData`
单个时间序列信息. 



**Notes:**

> 用于储存个股的数据信息及预测label，全部使用numpy，日期格式为整数: ``YYYYMMDD``。 数据分三个部分：时间，数据，标签，第一个维度都是时间，数据的第二个维度为特征。 

<a href="../src/alphanet/data.py#L27"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(dates: ndarray, data: ndarray, labels: ndarray)
```

储存个股的数据信息及预测目标，全部使用numpy，日期格式为整数: ``YYYYMMDD``. 



**Args:**
 
 - <b>`dates`</b>:  日期列, 1D ``numpy.ndarray``, 整数 
 - <b>`data`</b>:  训练输入的X，2D ``numpy.ndarray``, (日期长度 x 特征数量) 
 - <b>`labels`</b>:  训练标签Y, 1D ``numpy.ndarray``, 长度与dates相同。  如果为分类问题则是2D, (日期长度 x 类别数量) 





---

<a href="../src/alphanet/data.py#L56"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `TrainValData`
根据训练天数、验证天数、样本历史长度、训练起点生成不同训练阶段的数据. 

<a href="../src/alphanet/data.py#L59"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    time_series_list: List[TimeSeriesData],
    train_length: int = 1200,
    validate_length: int = 300,
    history_length: int = 30,
    train_val_gap: int = 10,
    sample_step: int = 2,
    fill_na: float = nan,
    normalize: bool = False
)
```

用于获取不同阶段的训练集和验证集. 



**Notes:**

> ``time_series_list``储存全部的时间序列信息， 其中每支股票序列为一个单独``TimeSeriesData``， 完整数据为``List[TimeSeriesData]``类型。 
>此外需要提供训练集总交易天数(``train_length``)、 验证集总交易天数(``validate_length``)、 单个样本用到的历史长度(``history_length``)、 采样步进大小(``sample_step``)。 
>使用方法为：通过get(start_date)方法获取从start_date 开始的训练机和验证集。通过逐渐增大start_date训练多个模型回测。 
>``train_val_gap``参数为验证集第一天与训练集最后一天中间间隔的天数， 如果是相临，则train_val_gap = 0。设置该参数的目的如下： 
>如果希望预测未来十天的累计收益，则预测时用到的输入数据为最近的历史数据来预测 未来十天的累计收益，即用t(-history)到t(0)的数据来预测t(1)到t(11)的累计收益 而训练时因为要用到十天累计收益做标签，最近的一个十天累计收益是从t(-10)到t(0)， 用到的历史数据则必须是t(-history-11)到t(-11)的数据。 而validation时，如果第一个预测点是t(1)(明天收盘价)至t(11)的累计收益， 则与最后一个训练的数据即：t(-10)至t(0)之间间隔了10天， 使用``train_val_gap=10``。 
>可选项为fill_na，缺失数据填充值，默认为np.Na 训练时跳过所有有缺失数据的样本。 
>

**Args:**
 
 - <b>`time_series_list`</b>:  TimeSeriesData 列表 
 - <b>`train_length`</b>:  训练集天数 
 - <b>`validate_length`</b>:  验证集天数 
 - <b>`history_length`</b>:  每个样本的历史天数 
 - <b>`train_val_gap`</b>:  训练集与验证集的间隔 
 - <b>`sample_step`</b>:  采样sample时步进的天数 
 - <b>`fill_na`</b>:  默认填充为np.NaN，训练时会跳过有确实数据的样本 
 - <b>`normalize`</b>:  是否对非率值做每个历史片段的max/min标准化 




---

<a href="../src/alphanet/data.py#L207"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get`

```python
get(
    start_date: int,
    order='by_date',
    validate_only=False,
    validate_length=None,
    normalize=False
)
```

获取从某天开始的训练集和验证集. 



**Notes:**

> 根据设定的训练集天数以及验证集天数，从start_date开始获取正确的 训练集以及验证集，以及他们各自的日期范围信息(该信息以字典形式返回)。 
>需要注意: 训练集的的开始和结束是指其data, label时间范围并集，而 验证集的开始和结束则只是指其label的时间范围。 验证集的输入数据可以与训练集重叠，只要其标签数据的包含的时间范围 与训练集数据包含的时间范围没有交集即可。 
>具体时间信息参考函数返回的最后一个元素，是一个包含时间信息的``dict``。 
>

**Args:**
 
 - <b>`start_date`</b>:  该轮训练开始日期，整数``YYYYMMDD`` 
 - <b>`order`</b>:  有三种顺序 ``shuffle``, ``by_date``, ``by_series``。  分别为随机打乱股票和时间，按时间顺序优先，按股票顺序优先，默认by_date。 
 - <b>`validate_only`</b>:  如果设置为True，则只返回validate set  和训练集、验证集时间信息。可以用于训练后的分析。 
 - <b>`validate_length`</b> (int):  override class validate_length 
 - <b>`normalize`</b> (bool):  override class normalize 



**Returns:**
 如果``validate_only=False``，返回训练集、验证集、日期信息： (train, val, dates_info(dict))。 如果为``True``，则返回验证集、日期信息。 



**Raises:**
 
 - <b>`ValueError`</b>:  日期范围超出最大日期会报ValueError。 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
