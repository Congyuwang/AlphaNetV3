"""多维多时间序列神经网络滚动训练的数据工具.

version: 0.0.11

author: Congyu Wang

date: 2021-07-26
"""
import tensorflow as _tf
import numpy as _np
from typing import List as _List


__all__ = ["TimeSeriesData", "TrainValData"]


class TimeSeriesData:
    """单个时间序列信息.

    Notes:
        用于储存个股的数据信息及预测label，全部使用numpy，日期格式为整数: ``YYYYMMDD``。
        数据分三个部分：时间，数据，标签，第一个维度都是时间，数据的第二个维度为特征。

    """

    def __init__(self,
                 dates: _np.ndarray,
                 data: _np.ndarray,
                 labels: _np.ndarray):
        """储存个股的数据信息及预测目标，全部使用numpy，日期格式为整数: ``YYYYMMDD``.

        Args:
            dates: 日期列, 1D ``numpy.ndarray``, 整数
            data: 训练输入的X，2D ``numpy.ndarray``, (日期长度 x 特征数量)
            labels: 训练标签Y, 1D ``numpy.ndarray``, 长度与dates相同。
                如果为分类问题则是2D, (日期长度 x 类别数量)

        """
        # 检查参数类型
        if (not type(dates) is _np.ndarray or
                not type(data) is _np.ndarray or
                not type(labels) is _np.ndarray):
            raise ValueError("Data should be numpy arrays")
        # 检查日期、数据、标签长度是否一致
        if len(dates) != len(data) or len(dates) != len(labels):
            raise ValueError("Bad data shape")
        # 检查维度是否正确
        if dates.ndim != 1 or data.ndim != 2 or not 1 <= labels.ndim <= 2:
            raise ValueError("Wrong dimensions")
        self.dates = dates.astype(_np.int32)
        self.data = data
        self.labels = labels


class TrainValData:
    """根据训练天数、验证天数、样本历史长度、训练起点生成不同训练阶段的数据."""

    def __init__(self,
                 time_series_list: _List[TimeSeriesData],
                 train_length: int = 1200,
                 validate_length: int = 300,
                 history_length: int = 30,
                 train_val_gap: int = 10,
                 sample_step: int = 2,
                 fill_na: _np.float = _np.NaN):
        """用于获取不同阶段的训练集和验证集.

        Notes:
            ``time_series_list``储存全部的时间序列信息，
            其中每支股票序列为一个单独``TimeSeriesData``，
            完整数据为``List[TimeSeriesData]``类型。

            此外需要提供训练集总交易天数(``train_length``)、
            验证集总交易天数(``validate_length``)、
            单个样本用到的历史长度(``history_length``)、
            采样步进大小(``sample_step``)。

            使用方法为：通过get(start_date)方法获取从start_date
            开始的训练机和验证集。通过逐渐增大start_date训练多个模型回测。

            ``train_val_gap``参数为验证集第一天与训练集最后一天中间间隔的天数，
            如果是相临，则train_val_gap = 0。设置该参数的目的如下：

            如果希望预测未来十天的累计收益，则预测时用到的输入数据为最近的历史数据来预测
            未来十天的累计收益，即用t(-history)到t(0)的数据来预测t(1)到t(11)的累计收益
            而训练时因为要用到十天累计收益做标签，最近的一个十天累计收益是从t(-10)到t(0)，
            用到的历史数据则必须是t(-history-11)到t(-11)的数据。
            而validation时，如果第一个预测点是t(1)(明天收盘价)至t(11)的累计收益，
            则与最后一个训练的数据即：t(-10)至t(0)之间间隔了10天，
            使用``train_val_gap=10``。

            可选项为fill_na，缺失数据填充值，默认为np.Na
            训练时跳过所有有缺失数据的样本。

        Args:
            time_series_list: TimeSeriesData 列表
            train_length: 训练集天数
            validate_length: 验证集天数
            history_length: 每个样本的历史天数
            train_val_gap: 训练集与验证集的间隔
            sample_step: 采样sample时步进的天数
            fill_na: 默认填充为np.NaN，训练时会跳过有确实数据的样本

        """
        # 检查参数类型
        if type(time_series_list) is not list:
            raise ValueError("time_series_list should be a list")
        # 不允许空列表
        if len(time_series_list) == 0:
            raise ValueError("Empty time_series_list")
        # 检查列表元素类型
        for t in time_series_list:
            if type(t) is not TimeSeriesData:
                raise ValueError("time_series_data should be a list "
                                 "of TimeSeriesData objects")
        # 检查参数数值
        if (type(history_length) is not int or
                type(validate_length) is not int or
                type(sample_step) is not int or
                type(train_length) is not int or
                type(train_val_gap) is not int or
                history_length < 1 or
                validate_length < 1 or
                sample_step < 1 or
                train_val_gap < 0 or
                train_length < history_length):
            raise ValueError("bad arguments")

        if type(fill_na) is not _np.float:
            raise ValueError("fill_na should be numpy float")

        # 确保数据特征数量一致
        self.__feature_counts = time_series_list[0].data.shape[1]
        for series in time_series_list:
            if series.data.shape[1] != self.__feature_counts:
                raise ValueError("time series do not have "
                                 "the same number of features")

        # 确保标签维度一致
        label_dims = time_series_list[0].labels.ndim
        for series in time_series_list:
            if series.labels.ndim != label_dims:
                raise ValueError("time labels do not have "
                                 "the same number of dimensions")

        # 标签类别数量
        class_num = 0
        if label_dims == 2:
            class_num = time_series_list[0].labels.shape[1]
            for series in time_series_list:
                if series.labels.shape[1] != class_num:
                    raise ValueError("time series labels do not have "
                                     "the same number of classes")

        self.__class_num = class_num

        # 获取日期列表（所有时间序列日期的并集）
        self.__distinct_dates = _np.unique([date for stock in time_series_list
                                            for date in stock.dates])
        self.__distinct_dates.sort()

        # 聚合数据为(序列(股票), 时间, 特征数量)的张量，缺失数据为np.NaN
        # 标签维度为(序列(股票), 时间)
        self.__data = _np.empty((len(time_series_list),
                                 len(self.__distinct_dates),
                                 self.__feature_counts))
        if self.__class_num == 0:
            self.__labels = _np.empty((len(time_series_list),
                                       len(self.__distinct_dates)))
        else:
            self.__labels = _np.empty((len(time_series_list),
                                       len(self.__distinct_dates),
                                       self.__class_num))

        self.__series_date_matrix = _np.empty((len(time_series_list),
                                               len(self.__distinct_dates), 2))
        self.__data[:] = fill_na
        self.__labels[:] = fill_na
        self.__series_date_matrix[:] = fill_na

        # 根据日期序列的位置向张量填充数据
        dates_positions = {date: index
                           for index, date in enumerate(self.__distinct_dates)}
        dates_position_mapper = _np.vectorize(lambda d: dates_positions[d])
        for i, series in enumerate(time_series_list):
            # 找到该序列series.dates日期在日期列表中的位置
            # 将第i个序列填充至tensor的第i行
            position_index = dates_position_mapper(series.dates)
            self.__data[i, position_index, :] = series.data
            if self.__class_num == 0:
                self.__labels[i, position_index] = series.labels
            else:
                self.__labels[i, position_index, :] = series.labels
            self.__series_date_matrix[i, position_index, 0] = series.dates
            self.__series_date_matrix[i, position_index, 1] = i

        # numpy -> tensor
        self.__data = _tf.constant(self.__data, dtype=_tf.float32)
        self.__labels = _tf.constant(self.__labels, dtype=_tf.float32)
        self.__train_length = train_length
        self.__validate_length = validate_length
        self.__history_length = history_length
        self.__sample_step = sample_step
        self.__train_val_gap = train_val_gap

    def get(self,
            start_date: int,
            order="by_date",
            validate_only=False):
        """获取从某天开始的训练集和验证集.

        Notes:
            根据设定的训练集天数以及验证集天数，从start_date开始获取正确的
            训练集以及验证集，以及他们各自的日期范围信息(该信息以字典形式返回)。

            需要注意:
            训练集的的开始和结束是指其data, label时间范围并集，而
            验证集的开始和结束则只是指其label的时间范围。
            验证集的输入数据可以与训练集重叠，只要其标签数据的包含的时间范围
            与训练集数据包含的时间范围没有交集即可。

            具体时间信息参考函数返回的最后一个元素，是一个包含时间信息的``dict``。

        Args:
            start_date: 该轮训练开始日期，整数``YYYYMMDD``
            order: 有三种顺序 ``shuffle``, ``by_date``, ``by_series``。
                分别为随机打乱股票和时间，按时间顺序优先，按股票顺序优先，默认by_date。
            validate_only: 如果设置为True，则只返回validate set
                和训练集、验证集时间信息。可以用于训练后的分析。

        Returns:
            如果``validate_only=False``，返回训练集、验证集、日期信息：
            (train, val, dates_info(dict))。
            如果为``True``，则返回验证集、日期信息。

        Raises:
            ValueError: 日期范围超出最大日期会报ValueError。

        """
        return self.__get_in_memory__(start_date,
                                      order,
                                      validate_only)

    def __get_in_memory__(self,
                          start_date,
                          order="by_date",
                          validate_only=False):
        """使用显存生成历史数据.

        使用tensorflow from_tensor_slices，通过传递完整的tensor进行训练，
        股票数量大时，需要较大内存

        """
        # 获取用于构建训练集、验证集的相关信息
        train_args, val_args, dates_info = self.__get_period_info__(start_date,
                                                                    order)
        # 将输入的数据、标签片段转化为单个sample包含history日期长度的历史信息
        (val_x,
         val_y,
         val_dates_series) = __full_tensor_generation__(*val_args)
        # 转化为tensorflow DataSet
        val = _tf.data.Dataset.from_tensor_slices((val_x, val_y))
        val_dates_series = val_dates_series.numpy().astype(int)
        val_dates_list = val_dates_series[:, 0].tolist()
        val_series_list = val_dates_series[:, 1].tolist()
        dates_info["validation"]["dates_list"] = val_dates_list
        dates_info["validation"]["series_list"] = val_series_list

        if validate_only:
            return val, dates_info

        (train_x,
         train_y,
         train_dates_series) = __full_tensor_generation__(*train_args)
        train = _tf.data.Dataset.from_tensor_slices((train_x, train_y))
        train_dates_series = train_dates_series.numpy().astype(int)
        train_dates_list = train_dates_series[:, 0].tolist()
        train_series_list = train_dates_series[:, 1].tolist()
        dates_info["training"]["dates_list"] = train_dates_list
        dates_info["training"]["series_list"] = train_series_list
        return train, val, dates_info

    def __get_period_info__(self, start_date, order="by_date"):
        """根据开始时间计算用于构建训练集、验证集的相关信息."""
        if type(start_date) is not int:
            raise ValueError("start date should be an integer YYYYMMDD")

        # 找到大于等于start_date的最小日期
        after_start_date = self.__distinct_dates >= start_date
        first_date = _np.min(self.__distinct_dates[after_start_date])

        # 查看剩余日期数量是否大于等于训练集验证集总长度
        if _np.sum(after_start_date) < (self.__train_length +
                                        self.__validate_length +
                                        self.__train_val_gap):
            raise ValueError("date range exceeded end of dates")

        # 计算各个时间节点在时间列表中的位置
        # 训练集开始位置(data的开始位置)
        train_start_index = __first_index__(self.__distinct_dates, first_date)
        # 训练集结束位置(不包含)
        train_end_index = train_start_index + self.__train_length
        # 验证集开始位置(data的开始位置)
        val_start_index = (train_end_index -
                           self.__history_length +
                           self.__train_val_gap + 1)
        # 验证集结束位置(不包含)
        val_end_index = (train_end_index +
                         self.__validate_length +
                         self.__train_val_gap)

        # 根据各个数据集的开始结束位置以及训练数据的顺序选项，获取构建数据的参数
        train_args = self.__get_generator_args__(train_start_index,
                                                 train_end_index,
                                                 order=order)
        val_args = self.__get_generator_args__(val_start_index,
                                               val_end_index,
                                               order=order)
        dates_info = self.__dates_info__(train_start_index,
                                         val_start_index,
                                         train_args,
                                         val_args)

        return train_args, val_args, dates_info

    def __get_generator_args__(self, start_index, end_index, order="by_date"):
        """获取单个generator需要的数据片段.

        Notes:
            根据数据集的开始、结束位置以及的顺序选项，获取该训练集的数据、标签片段
            以及用于生成训练数据的(序列, 日期)pair列表的顺序信息(generation_list)。

            generation_list第一列为序列编号，第二列为日期。

            generation_list中的日期数字代表*每个历史数据片段*的第一个日期相对
            该数据集片段（start_index:end_index）的位置。

            注意：

                - 该处的日期列表不代表每个历史片段的结束位置

                - 也不是相对TrainValData类日期列表的位置
        """
        length = end_index - start_index
        data = self.__data[:, start_index:end_index, :]
        label = self.__labels[:, start_index:end_index]
        dates_series = self.__series_date_matrix[:, start_index:end_index, :]
        generation_list = [[series_i, t]
                           for t in range(0,
                                          length - self.__history_length + 1,
                                          self.__sample_step)
                           for series_i in range(len(data))]

        if order == "shuffle":
            _np.random.shuffle(generation_list)
        elif order == "by_date":
            pass
        elif order == "by_series":
            generation_list = sorted(generation_list, key=lambda k: k[0])
        else:
            raise ValueError("wrong order argument, choose from `shuffle`, "
                             "`by_date`, and `by_series`")

        generation_list = _tf.constant(generation_list, dtype=_tf.int32)
        history_length = _tf.constant(self.__history_length, dtype=_tf.int32)

        return data, label, generation_list, history_length, dates_series

    def __dates_info__(self,
                       train_start_index,
                       val_start_index,
                       train_args,
                       val_args):
        """根据生成数据的列表，计算用于显示的日期信息."""
        # 获取generation_list(生成数据的顺序信息)的时间列
        # 该时间列为相对数据片段开头的时间位置
        train_generation_list = train_args[2]
        val_generation_list = val_args[2]
        train_time_index = train_generation_list[:, 1]
        val_time_index = val_generation_list[:, 1]

        # 加上片段的开始位置，得到相对TrainValData类日期列表的位置
        train_time_index += train_start_index
        val_time_index += val_start_index

        # 训练集在日期列表中的开始位置
        training_beginning = _tf.reduce_min(train_time_index)

        # 结束位置：加上历史长度减去一，获取最大日期位置(inclusive)
        training_ending = _tf.reduce_max(train_time_index)
        training_ending += self.__history_length - 1

        # validation集每次取的都是某历史片段末尾的数据
        # 所以加上历史减去一
        validation_index = val_time_index + self.__history_length - 1
        validation_beginning = _tf.reduce_min(validation_index)
        validation_ending = _tf.reduce_max(validation_index)

        dates_info = {
            "training": {
                "start_date": int(self.__distinct_dates[training_beginning]),
                "end_date": int(self.__distinct_dates[training_ending])
            },
            "validation": {
                "start_date": int(self.__distinct_dates[validation_beginning]),
                "end_date": int(self.__distinct_dates[validation_ending]),
            }
        }
        return dates_info


def __history_expander__(data_tensor, history_length):
    """错位叠加历史数据，获取(序列，时间，历史，特征)的四个维度."""
    total_time_length = data_tensor.shape[1]
    data_expanded = _tf.stack([data_tensor[:, i: (total_time_length + i
                                                  - history_length + 1), :]
                               for i in _tf.range(history_length)])
    # 调整维度顺序 (history, sample)
    return _tf.transpose(data_expanded, perm=[1, 2, 0, 3])


def __full_tensor_generation__(data,
                               label,
                               generation_list,
                               history,
                               dates_series):
    """将输入的数据、标签片段转化为单个sample包含history日期长度的历史信息."""
    # 先将该数据片段的历史维度展开

    # 根据generation_list指定的series，日期，获取标签及数据片段
    total_time_length = data.shape[1]
    expanded = [_tf.gather_nd(data[:, i: (total_time_length + i
                                          - history + 1), :], generation_list)
                for i in _tf.range(history)]
    data_all = _tf.stack(expanded, axis=1)
    label_all = _tf.gather_nd(label[:, history - 1:], generation_list)
    dates_series_all = _tf.gather_nd(dates_series[:, history - 1:],
                                     generation_list)

    # 去掉所有包含缺失数据的某股票某时间历史片段
    label_nan = _tf.math.is_nan(label_all)
    data_nan = _tf.math.is_nan(data_all)
    nan_series_time_index = _tf.math.reduce_any(
        _tf.math.reduce_any(data_nan, axis=2),
        axis=1
    )
    not_nan = _tf.math.logical_not(_tf.math.logical_or(nan_series_time_index, label_nan))
    return data_all[not_nan], label_all[not_nan], dates_series_all[not_nan]


def __first_index__(array, element):
    """计算第一个出现的元素的位置."""
    return _np.min(_np.where(array == element))
