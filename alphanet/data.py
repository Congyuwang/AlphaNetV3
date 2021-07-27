"""
多维多时间序列神经网络滚动训练数据工具箱

version: 0.3
author: Congyu Wang
date: 2021-07-26
"""
import tensorflow as tf
import numpy as np
from typing import List


__all__ = ["TimeSeriesData", "TrainValData"]


class TimeSeriesData:
    """
    储存个股的数据信息及预测目标，全部使用numpy，日期格式为整数: `YYYYMMDD`.
    """

    def __init__(self,
                 dates: np.ndarray,
                 data: np.ndarray,
                 labels: np.ndarray):
        """
        储存个股的数据信息及预测目标，全部使用numpy，日期格式为整数: `YYYYMMDD`

        :param dates: 日期列, 1D numpy array
        :param data: 训练输入的X，2D numpy array, (日期长度 x 特征数量)
        :param labels: 训练标签Y, 1D numpy array, 长度与dates相同
        """
        # 检查参数类型
        if (not type(dates) is np.ndarray or
                not type(data) is np.ndarray or
                not type(labels) is np.ndarray):
            raise Exception("Data should be numpy arrays")
        # 检查日期、数据、标签长度是否一致
        if len(dates) != len(data) or len(dates) != len(labels):
            raise Exception("Bad data shape")
        # 检查维度是否正确
        if dates.ndim != 1 or data.ndim != 2 or labels.ndim != 1:
            raise Exception("Wrong dimensions")
        self.dates = dates.astype(np.int32)
        self.data = data
        self.labels = labels


class TrainValData:
    """
    获取train, validation tensorflow dataset, 以及日期信息
    """

    def __init__(self,
                 time_series_list: List[TimeSeriesData],
                 train_length=1200,
                 validate_length=300,
                 history_length=30,
                 train_val_gap=10,
                 sample_step=2,
                 fill_na=np.NaN):
        """
        储存全部的时间序列信息，通过get(start_date)方法获取从start_date
        开始的训练机和验证集。

        train_val_gap参数为验证集第一天与训练集最后一天中间间隔的天数，
        如果是相邻，则train_val_gap = 0。

        设置该参数的目的如下：
        如果希望预测未来十天的累计收益，则预测时用到的输入数据为最近的历史数据来预测
        未来十天的累计收益，即用t(-history)到t(0)的数据来预测t(1)到t(11)的累计收益
        而训练时因为要用到十天累计收益做标签，最近的一个十天累计收益是从t(-10)到t(0)，
        用到的历史数据则必须是t(-history-11)到t(-11)的数据，为了确保validation的
        有效性，则最好将validation的第一个数据位置与train的最后一个数据位置在时间上
        相差11天，即间隔10天。

        :param time_series_list: TimeSeriesData 列表
        :param train_length: 训练集天数
        :param validate_length: 验证集天数
        :param history_length: 每个sample的历史天数
        :param train_val_gap: 训练集与验证集的间隔
        :param sample_step: 采样sample时步进的天数
        :param fill_na: 默认填充为np.NaN，训练时会跳过有确实数据的样本
        """

        # 检查参数类型
        if type(time_series_list) is not list:
            raise Exception("time_series_list should be a list")
        # 不允许空列表
        if len(time_series_list) == 0:
            raise Exception("Empty time_series_list")
        # 检查列表元素类型
        for t in time_series_list:
            if type(t) is not TimeSeriesData:
                raise Exception("time_series_data should be a list "
                                "of TimeSeriesData objects")
        # 检查参数数值
        if (history_length < 1 or
                validate_length < 1 or
                sample_step < 1 or
                train_val_gap < 0 or
                train_length < history_length):
            raise ValueError("bad arguments")
        # 确保数据特征数量一致
        self.__feature_counts = time_series_list[0].data.shape[1]
        for series in time_series_list:
            if series.data.shape[1] != self.__feature_counts:
                raise Exception("time series do not have "
                                "the same number of features")

        # 获取日期列表（所有时间序列日期的并集）
        self.__distinct_dates = np.unique([date for stock in time_series_list
                                           for date in stock.dates])
        self.__distinct_dates.sort()

        # 聚合数据为(序列(股票), 时间, 特征数量)的张量，缺失数据为np.NaN
        # 标签维度为(序列(股票), 时间)
        self.__data = np.empty((len(time_series_list),
                                len(self.__distinct_dates),
                                self.__feature_counts))
        self.__labels = np.empty((len(time_series_list),
                                  len(self.__distinct_dates)))
        self.__data[:] = fill_na
        self.__labels[:] = fill_na

        # 根据日期序列的位置向张量填充数据
        dates_positions = {date: index
                           for index, date in enumerate(self.__distinct_dates)}
        dates_position_mapper = np.vectorize(lambda d: dates_positions[d])
        for i, series in enumerate(time_series_list):
            # 找到该序列series.dates日期在日期列表中的位置，
            # 将第i个序列填充至tensor的第i行
            position_index = dates_position_mapper(series.dates)
            self.__data[i, position_index, :] = series.data
            self.__labels[i, position_index] = series.labels

        # numpy -> tensor
        self.__data = tf.constant(self.__data, dtype=tf.float32)
        self.__labels = tf.constant(self.__labels, dtype=tf.float32)
        self.__train_length = train_length
        self.__validate_length = validate_length
        self.__history_length = history_length
        self.__sample_step = sample_step
        self.__train_val_gap = train_val_gap

    def get(self,
            start_date,
            order="by_date",
            mode="in_memory"):
        """
        根据设定的训练集天数以及验证集天数，从start_date开始获取正确的
        训练集以及验证集，以及他们各自的日期范围信息(该信息以字典形式返回)。

        training set 的开始和结束是指其data, label所有的时间范围(inclusive)。
        validation set 的开始和结束则只是指其label的时间范围，
        因为validation set允许用到training set内的X历史数据。
        具体时间信息参考返回的第三个元素dates_info。

        如果日期范围超出最大日期会报ValueError。

        :param start_date: 该轮训练开始日期，整数YYYYMMDD。
        :param order: 有三种顺序: shuffle, by_date, by_series,
        分别为随机打乱股票和时间，按时间顺序优先，按股票顺序优先，默认by_date。
        :param mode: `generator` 或 `in_memory`. generator 速度极慢，
        in_memory速度较快，默认in_memory。feature、series数量大内存不足时
        可以使用generator。'in_memory'模式股票数量较大以及step较小时，
        可能会要求较大内存。

        :return: train, val, dates_info(dict)
        """

        if mode == "generator":
            return self.__get_from_generator__(start_date, order)
        elif mode == "in_memory":
            return self.__get_in_memory__(start_date, order)
        else:
            raise ValueError("mode unimplemented, choose from `generator` "
                             "and `in_memory`")

    def __get_in_memory__(self, start_date, order="by_date"):
        """
        使用tensorflow from_tensor_slices，通过传递完整的tensor进行训练，
        股票数量大时，需要较大内存
        """
        # 获取用于构建训练集、验证集的相关信息
        train_args, val_args, dates_info = self.__get_period_info__(start_date,
                                                                    order)
        # 将输入的数据、标签片段转化为单个sample包含history日期长度的历史信息
        train_x, train_y = __full_tensor_generation__(*train_args)
        val_x, val_y = __full_tensor_generation__(*val_args)

        # 转化为tensorflow DataSet
        train = tf.data.Dataset.from_tensor_slices((train_x, train_y))
        val = tf.data.Dataset.from_tensor_slices((val_x, val_y))
        return train, val, dates_info

    def __get_from_generator__(self, start_date, order="by_date"):
        """
        使用tensorflow.data.DataSet.from_generator，占用内存少，生成数据慢
        """
        # 获取用于构建训练集、验证集的相关信息
        train_args, val_args, dates_info = self.__get_period_info__(start_date,
                                                                    order)
        sig = (
            tf.TensorSpec(
                shape=(self.__history_length,
                       self.__feature_counts),
                dtype=tf.float32
            ),
            tf.TensorSpec(
                shape=(),
                dtype=tf.float32
            )
        )
        train_dataset = tf.data.Dataset.from_generator(__generator__,
                                                       args=train_args,
                                                       output_signature=sig)
        val_dataset = tf.data.Dataset.from_generator(__generator__,
                                                     args=val_args,
                                                     output_signature=sig)
        return train_dataset, val_dataset, dates_info

    def __get_period_info__(self, start_date, order="by_date"):
        """
        根据开始时间计算用于构建训练集、验证集的相关信息
        """
        if type(start_date) is not int:
            raise Exception("start date should be an integer YYYYMMDD")

        # 找到大于等于start_date的最小日期
        after_start_date = self.__distinct_dates >= start_date
        first_date = np.min(self.__distinct_dates[after_start_date])

        # 查看剩余日期数量是否大于等于训练集验证集总长度
        if np.sum(after_start_date) < (self.__train_length +
                                       self.__validate_length +
                                       self.__train_val_gap):
            raise ValueError("date range exceeded end of dates")

        # 计算各个时间节点在时间列表中的位置
        # 训练集开始位置
        train_start_index = __first_index__(self.__distinct_dates, first_date)
        # 训练集结束位置(不包含)
        train_end_index = train_start_index + self.__train_length
        # 验证集开始位置
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
        """
        根据数据集的开始、结束位置以及的顺序选项，获取该训练集的数据、标签片段
        以及用于生成训练数据的(序列, 日期)pair列表的顺序信息(generation_list)。

        generation_list第一列为序列编号，第二列为日期。

        generation_list中的日期数字代表*每个历史数据片段*的第一个日期相对
        该数据集片段（start_index:end_index）的位置。

        注意：
        1. 该处的日期列表不代表每个历史片段的结束位置
        2. 也不是相对TrainValData类日期列表的位置
        """
        length = end_index - start_index
        data = self.__data[:, start_index:end_index, :]
        label = self.__labels[:, start_index:end_index]
        generation_list = [[series_i, t]
                           for t in range(0,
                                          length - self.__history_length + 1,
                                          self.__sample_step)
                           for series_i in range(len(data))]

        if order == "shuffle":
            np.random.shuffle(generation_list)
        elif order == "by_date":
            pass
        elif order == "by_series":
            generation_list = sorted(generation_list, key=lambda k: k[0])
        else:
            raise Exception("wrong order argument, choose from `shuffle`, "
                            "`by_date`, and `by_series`")

        generation_list = tf.constant(generation_list, dtype=tf.int32)
        history_length = tf.constant(self.__history_length, dtype=tf.int32)

        return data, label, generation_list, history_length

    def __dates_info__(self,
                       train_start_index,
                       val_start_index,
                       train_args,
                       val_args):
        """
        根据生成数据的列表，计算用于显示的日期信息
        """

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
        training_beginning = tf.reduce_min(train_time_index)

        # 结束位置：加上历史长度减去一，获取最大日期位置(inclusive)
        training_ending = tf.reduce_max(train_time_index)
        training_ending += self.__history_length - 1

        # validation集每次取的都是某历史片段末尾的数据
        # 所以加上历史减去一
        validation_index = val_time_index + self.__history_length - 1
        validation_beginning = tf.reduce_min(validation_index)
        validation_ending = tf.reduce_max(validation_index)

        dates_info = {
            "training": {
                "start_date": int(self.__distinct_dates[training_beginning]),
                "end_date": int(self.__distinct_dates[training_ending])
            },
            "validation": {
                "start_date": int(self.__distinct_dates[validation_beginning]),
                "end_date": int(self.__distinct_dates[validation_ending])
            }
        }
        return dates_info


def __history_expander__(data_tensor, history_length):
    """
    错位叠加历史数据，获取(序列，时间，历史，特征)的四个维度
    """
    total_time_length = data_tensor.shape[1]
    data_expanded = tf.stack([data_tensor[:, i: (total_time_length + i
                                                 - history_length + 1), :]
                              for i in tf.range(history_length)])
    # 调整维度顺序 (series, time, history, features)
    return tf.transpose(data_expanded, perm=[1, 2, 0, 3])


def __full_tensor_generation__(data, label, generation_list, history):
    """
    将输入的数据、标签片段转化为单个sample包含history日期长度的历史信息
    """

    # 先将该数据片段的历史维度展开
    data_expanded = __history_expander__(data, history)

    # 根据generation_list指定的series，日期，获取标签及数据片段
    label_all = tf.gather_nd(label[:, history - 1:], generation_list)
    data_all = tf.gather_nd(data_expanded, generation_list)

    # 去掉所有包含缺失数据的某股票某时间历史片段
    label_nan = tf.cast(tf.math.is_nan(label_all), tf.int64)
    data_nan = tf.cast(tf.math.is_nan(data_all), tf.int64)
    nan_series_time_index = tf.reduce_sum(tf.reduce_sum(data_nan, axis=2), axis=1)
    not_nan = tf.add(nan_series_time_index, label_nan) == 0
    return data_all[not_nan], label_all[not_nan]


def __training_example_getter__(data, label, series_i, i, history_length):
    """
    DataSet.from_generator会用到的函数，获取单个训练数据
    """
    x = data[series_i][i: i + history_length]
    y = label[series_i][i + history_length - 1]

    if (tf.reduce_sum(tf.cast(tf.math.is_nan(x), tf.int64)) == 0 and
            tf.reduce_sum(tf.cast(tf.math.is_nan(y), tf.int64)) == 0):
        return x, y
    else:
        return None


def __generator__(data, label, generation_list, history_length):
    """
    DataSet.from_generator会用到的generator
    """
    for series_i, i in generation_list:
        d = __training_example_getter__(data, label, series_i, i, history_length)
        if d:
            x, y = d
            yield x, y


def __first_index__(array, element):
    """
    计算第一个出现的元素的位置
    """
    return np.min(np.where(array == element))
