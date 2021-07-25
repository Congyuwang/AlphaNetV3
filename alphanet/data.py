"""
复现华泰金工 alpha net V3 版本

```
input: (batch_size, history time steps, features)

                stride = 5
        +-> expand features -> BN -> GRU -> BN -+
input --|       stride = 10                     |- concat -> Dense(linear)
        +-> expand features -> BN -> GRU -> BN -+
```

(BN: batch normalization)

version: 0.2
author: Congyu Wang
date: 2021-07-22
"""
import tensorflow as tf
import numpy as np
from typing import List


__all__ = ["TimeSeriesData", "TrainValData"]


class TimeSeriesData:
    """
    储存一个个股的数据信息及预测目标，全部使用numpy，日期格式为整数: `YYYYMMDD`.
    """

    def __init__(self,
                 dates: np.ndarray,
                 data: np.ndarray,
                 labels: np.ndarray):
        """
        :param dates: 日期列, 1D numpy array
        :param data: 训练输入的X，2D numpy array, 日期长度 x 特征数量
        :param labels: 训练标签Y, 1D numpy array, 长度与dates相同
        """
        if (not type(dates) is np.ndarray or
                not type(data) is np.ndarray or
                not type(labels) is np.ndarray):
            raise Exception("data should be numpy arrays")
        if len(dates) != len(data) or len(dates) != len(labels):
            raise Exception("Bad data shape")
        if dates.ndim != 1 or data.ndim != 2 or labels.ndim != 1:
            raise Exception("wrong dimensions")
        self.dates = dates.astype(np.int32)
        self.data = data
        self.labels = labels


class TrainValData:
    """
    获取train, validation tensorflow dataset
    """

    def __init__(self,
                 time_series_list: List[TimeSeriesData],
                 train_length=1200,
                 validate_length=300,
                 history_length=30,
                 sample_step=2):
        """
        :param time_series_list: a list of times series
        :param train_length: 训练集天数
        :param validate_length: 验证集天数
        :param history_length: 每个sample的历史天数
        :param sample_step: 采样sample时前进的天数
        """
        # check time series shapes
        if type(time_series_list) is not list:
            raise Exception("time_series_list should be a list")
        if len(time_series_list) == 0:
            raise Exception("Empty time_series_list")
        if type(time_series_list[0]) is not TimeSeriesData:
            raise Exception("time_series_data should be a list "
                            "of TimeSeriesData objects")
        self.__feature_counts = time_series_list[0].data.shape[1]
        for series in time_series_list:
            if series.data.shape[1] != self.__feature_counts:
                raise Exception("time series do not have "
                                "the same number of features")

        # gather distinct dates
        self.__distinct_dates = np.unique([date for stock in time_series_list
                                           for date in stock.dates])
        self.__distinct_dates.sort()

        # initialize rectangular tensor according to dates list
        self.__data = np.empty((len(time_series_list),
                                len(self.__distinct_dates),
                                self.__feature_counts))
        self.__labels = np.empty((len(time_series_list),
                                  len(self.__distinct_dates)))
        self.__data[:] = np.NaN
        self.__labels[:] = np.NaN

        # fill in data for rectangular tensor
        dates_positions = {date: index
                           for index, date in enumerate(self.__distinct_dates)}
        dates_position_mapper = np.vectorize(lambda d: dates_positions[d])
        for i, series in enumerate(time_series_list):
            position_index = dates_position_mapper(series.dates)
            self.__data[i, position_index, :] = series.data
            self.__labels[i, position_index] = series.labels

        # convert to tensor constant
        self.__data = tf.constant(self.__data, dtype=tf.float32)
        self.__labels = tf.constant(self.__labels, dtype=tf.float32)
        self.__train_length = train_length
        self.__validate_length = validate_length
        self.__history_length = history_length
        self.__sample_step = sample_step

    def __get_generator_args__(self, start_index, end_index, order="shuffle"):
        """
        该函数根据每个股票的日期范围给出一个generation_list
        :param start_index: generate 的区间开始的index, inclusive
        :param end_index: generate 的区间结束的index, exclusive
        :param order: 有三种顺序: shuffle, by_date, by_series
        :return: generator
        """
        length = end_index - start_index
        data = self.__data[:, start_index:end_index, :]
        label = self.__labels[:, start_index:end_index]
        generation_list = [(series_i, i)
                           for i in range(0,
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

    def get(self, start_date, order="by_date"):
        """
        :param start_date: 该轮训练开始日期，整数YYYYMMDD
        :param order: 有三种顺序: shuffle, by_date, by_series,
        分别为随机打乱股票和时间，按时间顺序优先，按股票顺序优先
        :return: tensorflow dataset, (train, val)
        """

        if type(start_date) is not int:
            raise Exception("start date should be an integer YYYYMMDD")

        # find the actual starting date
        after_start_date = self.__distinct_dates >= start_date
        if np.sum(after_start_date) < (self.__train_length +
                                       self.__validate_length):
            raise Exception("date range exceeded end of dates")

        # get train, val periods
        first_date = np.min(self.__distinct_dates[after_start_date])
        train_start_index = __first_index__(self.__distinct_dates, first_date)
        train_end_index = train_start_index + self.__train_length
        val_start_index = (train_end_index -
                           self.__history_length +
                           self.__sample_step)
        val_end_index = train_end_index + self.__validate_length

        train_args = self.__get_generator_args__(train_start_index,
                                                 train_end_index,
                                                 order=order)

        val_args = self.__get_generator_args__(val_start_index,
                                               val_end_index,
                                               order=order)

        # get rolling sample generator
        # 通过传递tensor args比直接由generator从环境抓取数据运行速度更快，
        # 只不过args必须是rectangular tensors
        types = (tf.float32, tf.float32)
        shapes = ((self.__history_length, self.__feature_counts), ())
        train_dataset = tf.data.Dataset.from_generator(__generator__,
                                                       args=train_args,
                                                       output_types=types,
                                                       output_shapes=shapes)
        val_dataset = tf.data.Dataset.from_generator(__generator__,
                                                     args=val_args,
                                                     output_types=types,
                                                     output_shapes=shapes)

        dates_info = {
            "training set": {
                "start_date_inc": self.__distinct_dates[train_start_index],
                "end_date_exc": self.__distinct_dates[train_end_index]
            },
            "validation set": {
                "start_date_inc": self.__distinct_dates[val_start_index],
                "end_date_exc": self.__distinct_dates[val_end_index]
            }
        }

        return train_dataset, val_dataset, dates_info


def __generator__(data, label, generation_list, history_length):
    """
    tensorflow dataset 用到的generator
    :param data: tensor, 序列 x 时间 x 特征
    :param label: tensor, 序列 x 时间
    :param generation_list: 生成数据的顺序信息 (序列id, 时间) pairs
    :param history_length: 单个sample的历史长度
    """
    for series_i, i in generation_list:
        x = data[series_i][i: i + history_length]
        y = label[series_i][i + history_length - 1]

        # 如果该序列的历史片段内有缺失数据则跳过该数据
        if (tf.reduce_sum(tf.cast(tf.math.is_nan(x), tf.int64)) == 0 and
                tf.reduce_sum(tf.cast(tf.math.is_nan(y), tf.int64)) == 0):
            yield x, y


def __first_index__(array, element):
    """
    :param array: input array
    :param element: the element in the array to be found
    :return: the index of the element in the array
    """
    return np.min(np.where(array == element))
