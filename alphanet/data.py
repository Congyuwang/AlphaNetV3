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
        if (history_length < 1 or
                validate_length < 1 or
                train_length < history_length):
            raise ValueError("bad argument: either history_length < 1"
                             " or validate_length < 1"
                             " or train_length < history_length")
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

    def get(self,
            start_date,
            order="by_date"):
        """
        该函数首先检测data_folder中是否已经有储存了的tensor文件，如无，
        则生成改tensor。加载快于生成，此外，建议使用时加上cache()函数

        training set 的开始和结束是指其X, label所有数的时间范围(inclusive)。
        validation set 的开始和结束则只是指其label的时间范围，因为
        validation set 可以用到training set内的X历史数据。
        具体时间信息参考返回的第三个元素dates_info

        throws value error if date range exceeded end of dates

        :param start_date: 该轮训练开始日期，整数YYYYMMDD
        :param order: 有三种顺序: shuffle, by_date, by_series,
        分别为随机打乱股票和时间，按时间顺序优先，按股票顺序优先
        """
        train_args, val_args, dates_info = self.__get_period_info__(start_date,
                                                                    order)
        # generate tensors
        tf.print("Generating training data")
        train_x, train_y = __full_tensor_generation__(*train_args)
        tf.print("Generating validation data")
        val_x, val_y = __full_tensor_generation__(*val_args)

        train = tf.data.Dataset.from_tensor_slices((train_x, train_y))
        val = tf.data.Dataset.from_tensor_slices((val_x, val_y))

        return train, val, dates_info

    def get_from_generator(self, start_date, order="by_date"):
        """
        该函数使用generator来生成tensorflow dataset，速度较慢，
        仅建议在数据集过大、内存不足的情况下使用。

        training set 的开始和结束是指其X, label所有数的时间范围(inclusive)。
        validation set 的开始和结束则只是指其label的时间范围，因为
        validation set 可以用到training set内的X历史数据。
        具体时间信息参考返回的第三个元素dates_info

        throws value error if date range exceeded end of dates

        :param start_date: 该轮训练开始日期，整数YYYYMMDD
        :param order: 有三种顺序: shuffle, by_date, by_series,
        分别为随机打乱股票和时间，按时间顺序优先，按股票顺序优先
        :return: tensorflow dataset, (train, val), 以及dates_info
        """
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

    def __get_generator_args__(self, start_index, end_index, order="by_date"):
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
        generation_list = [[series_i, i]
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

    def __get_period_info__(self, start_date, order="by_date"):
        if type(start_date) is not int:
            raise Exception("start date should be an integer YYYYMMDD")

        # find the actual starting date
        after_start_date = self.__distinct_dates >= start_date
        if np.sum(after_start_date) < (self.__train_length +
                                       self.__validate_length):
            raise ValueError("date range exceeded end of dates")

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
        dates_info = self.__compute_dates_info__(train_start_index,
                                                 val_start_index,
                                                 train_args,
                                                 val_args)
        return train_args, val_args, dates_info

    def __compute_dates_info__(self,
                               train_start_index,
                               val_start_index,
                               train_args,
                               val_args):
        train_index = train_start_index + train_args[2][:, 1]
        val_index = val_start_index + val_args[2][:, 1]

        actual_training_beginning = tf.reduce_min(train_index)
        actual_training_end = (tf.reduce_max(train_index) +
                               self.__history_length - 1)
        actual_validation_index = val_index + self.__history_length - 1

        # these are all inclusive
        dates_info = {
            "training": {
                "start_date": int(self.__distinct_dates[
                                      actual_training_beginning
                                  ]),
                "end_date": int(self.__distinct_dates[
                                    actual_training_end
                                ])
            },
            "validation": {
                "start_date": int(self.__distinct_dates[
                                      tf.reduce_min(actual_validation_index)
                                  ]),
                "end_date": int(self.__distinct_dates[
                                    tf.reduce_max(actual_validation_index)
                                ])
            }
        }
        return dates_info

    def __write_data__(self,
                       start_date,
                       data_folder,
                       order="by_date"):
        """
        training set 的开始和结束是指其X, label所有数的时间范围(inclusive)。
        validation set 的开始和结束则只是指其label的时间范围，因为
        validation set 可以用到training set内的X历史数据。
        具体时间信息参考返回的第三个元素dates_info

        throws value error if date range exceeded end of dates

        :param data_folder: 储存该tensor的文件夹
        :param start_date: 该轮训练开始日期，整数YYYYMMDD
        :param order: 有三种顺序: shuffle, by_date, by_series,
        分别为随机打乱股票和时间，按时间顺序优先，按股票顺序优先
        """


def __get_file_names__(dates_info, order):
    """
    :return: train_x_file, train_y_file, val_x_file, val_y_file
    """
    train_start_date = dates_info["training"]["start_date"]
    train_end_date = dates_info["training"]["end_date"]
    val_start_date = dates_info["validation"]["start_date"]
    val_end_date = dates_info["validation"]["end_date"]
    train_x_file = f"train_x_{train_start_date}_{train_end_date}_{order}"
    train_y_file = f"train_y_{train_start_date}_{train_end_date}_{order}"
    val_x_file = f"val_x_{val_start_date}_{val_end_date}_{order}"
    val_y_file = f"val_y_{val_start_date}_{val_end_date}_{order}"
    json_file = f"{train_start_date}_{val_end_date}.json"
    return train_x_file, train_y_file, val_x_file, val_y_file, json_file


def __write_single_tensor__(file, tensor):
    tensor = tf.io.serialize_tensor(tensor)
    tf.io.write_file(file, tensor)


def __load_single_tensor__(file):
    tensor = tf.io.read_file(file)
    tensor = tf.io.parse_tensor(tensor)
    return tensor


def __history_expander__(data_tensor, history_length):
    total_time_length = data_tensor.shape[1]
    data_expanded = tf.stack([data_tensor[:, i: (total_time_length + i
                                                 - history_length + 1), :]
                              for i in tf.range(history_length)])
    # rearrange dimensions to (series, time, history, features)
    return tf.transpose(data_expanded, perm=[1, 2, 0, 3])


def __full_tensor_generation__(data, label, generation_list, history):
    # expanded to history
    # data_expanded : (series, time, history, features)
    data_expanded = __history_expander__(data, history)
    # gather according to the order of generation_list
    # label_all: (sample, )
    label_all = tf.gather_nd(label, generation_list)
    # data_all: (sample, history, features)
    data_all = tf.gather_nd(data_expanded, generation_list)
    # remove nan
    label_nan = tf.cast(tf.math.is_nan(label_all), tf.int64)
    data_nan = tf.cast(tf.math.is_nan(data_all), tf.int64)
    nan_series_time_index = tf.reduce_sum(tf.reduce_sum(data_nan, axis=2), axis=1)
    not_nan = tf.add(nan_series_time_index, label_nan) == 0
    return data_all[not_nan], label_all[not_nan]


@tf.function
def __fast_getter__(data, label, series_i, i, history_length):
    x = data[series_i][i: i + history_length]
    y = label[series_i][i + history_length - 1]

    if (tf.reduce_sum(tf.cast(tf.math.is_nan(x), tf.int64)) == 0 and
            tf.reduce_sum(tf.cast(tf.math.is_nan(y), tf.int64)) == 0):
        y = tf.cast(np.nan, tf.float32)
    return x, y


def __generator__(data, label, generation_list, history_length):
    """
    tensorflow dataset 用到的generator
    :param data: tensor, 序列 x 时间 x 特征
    :param label: tensor, 序列 x 时间
    :param generation_list: 生成数据的顺序信息 (序列id, 时间) pairs
    :param history_length: 单个sample的历史长度
    """
    for series_i, i in generation_list:
        d = __fast_getter__(data, label, series_i, i, history_length)
        x, y = d
        if tf.math.is_nan(y):
            yield x, y


def __first_index__(array, element):
    """
    :param array: input array
    :param element: the element in the array to be found
    :return: the index of the element in the array
    """
    return np.min(np.where(array == element))
