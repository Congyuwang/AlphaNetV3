"""
复现华泰金工 alpha net V3 版本

```
input: (batch_size, history time steps, features)

        +-> expand features (stride = 5)  -> BN -> GRU -> BN -+
input --|                                                     |- concat -> Dense(linear)
        +-> expand features (stride = 10) -> BN -> GRU -> BN -+
```

(BN: batch normalization)

version: 0.2
author: Congyu Wang
date: 2021-07-22
"""
import tensorflow as tf
import tensorflow.keras.layers as tfl
import numpy as np
from tensorflow.keras.layers import Layer
from typing import List


def __get_dimensions__(inputs, stride):
    """
    return time_steps, features, and output_length based on inputs and stride

    :param inputs: pass the inputs of layer to the function
    :param stride: the stride of the custom layer
    :return: (time_steps, features, output_length)
    """
    time_steps = inputs.shape[1]
    features = inputs.shape[2]
    output_length = time_steps // stride

    if time_steps % stride != 0:
        raise Exception("Error, time_steps 应该是 stride的整数倍")

    return time_steps, features, output_length


def __lower_triangle_without_diagonal_mask__(matrix):
    """
    the boolean mask of lower triangular part of the matrix without
    the diagonal elements.

    :param matrix: the input matrix
    :return: boolean mask the the same shape as the input matrix
    """
    ones = tf.ones_like(matrix)
    mask_lower = tf.linalg.band_part(ones, -1, 0)
    mask_diag = tf.linalg.band_part(ones, 0, 0)
    # lower triangle removing the diagonal elements
    mask = tf.cast(mask_lower - mask_diag, dtype=tf.bool)
    return mask


class Std(Layer):
    """
    计算每个feature各个stride的standard deviation
    """

    def __init__(self, stride=10, **kwargs):
        """
        :param stride: time steps需要是stride的整数倍
        """
        if stride <= 1:
            raise Exception("Illegal Argument: stride should be greater than 1")
        super(Std, self).__init__(**kwargs)
        self.stride = stride

    def call(self, inputs, *args, **kwargs):
        """
        :param inputs: 输入dimension为(batch_size, time_steps, features)
        :return: return dimension 为(batch_size, time_steps / stride, features)
        """
        time_steps, features, output_length = __get_dimensions__(inputs, self.stride)

        # compute means for each stride
        means = tf.nn.avg_pool(inputs, ksize=self.stride, strides=self.stride, padding="VALID")
        means = tf.repeat(means, self.stride, axis=1)

        # subtract means for each stride
        squared_diff = tf.square(tf.subtract(inputs, means))
        squared_diff = tf.reshape(squared_diff, (-1, output_length, self.stride, features))

        # compute standard deviation for each stride
        mean_squared_diff = tf.reduce_mean(squared_diff, axis=2)
        std = tf.sqrt(mean_squared_diff)

        return std


class ZScore(Layer):
    """
    并非严格意义上的z-score, 计算公式为每个feature各个stride的mean除以各自的standard deviation
    """

    def __init__(self, stride=10, **kwargs):
        """
        :param stride: time steps需要是stride的整数倍
        """
        if stride <= 1:
            raise Exception("Illegal Argument: stride should be greater than 1")
        super(ZScore, self).__init__(**kwargs)
        self.stride = stride

    def call(self, inputs, *args, **kwargs):
        """
        :param inputs: 输入dimension为(batch_size, time_steps, features)
        :return: return dimension 为(batch_size, time_steps / stride, features)
        """
        time_steps, features, output_length = __get_dimensions__(inputs, self.stride)

        # compute means for each stride
        means = tf.nn.avg_pool(inputs, ksize=self.stride, strides=self.stride, padding="VALID")

        # compute standard deviations for each stride
        squared_diff = tf.square(tf.subtract(inputs, tf.repeat(means, self.stride, axis=1)))
        squared_diff = tf.reshape(squared_diff, (-1, output_length, self.stride, features))
        mean_squared_diff = tf.reduce_mean(squared_diff, axis=2)
        std = tf.sqrt(mean_squared_diff)

        # divide means by standard deviations for each stride
        z_score = tf.math.divide_no_nan(means, std)
        return z_score


class LinearDecay(Layer):
    """
    以线性衰减为权重，计算每个feature各个stride的均值：
    如stride为10，则某feature该stride的权重为(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
    """

    def __init__(self, stride=10, **kwargs):
        """
        :param stride: time steps需要是stride的整数倍
        """
        if stride <= 1:
            raise Exception("Illegal Argument: stride should be greater than 1")
        super(LinearDecay, self).__init__(**kwargs)
        self.stride = stride

    def call(self, inputs, *args, **kwargs):
        """
        :param inputs: 输入dimension为(batch_size, time_steps, features)
        :return: return dimension 为(batch_size, time_steps / stride, features)
        """
        time_steps, features, output_length = __get_dimensions__(inputs, self.stride)

        output_shape = (-1, output_length, features)
        intermediate_shape = (-1, self.stride, features)

        # get linear decay kernel
        single_kernel = tf.linspace(1.0, self.stride, num=self.stride)
        kernel = tf.repeat(single_kernel, intermediate_shape[2])
        kernel = kernel / tf.reduce_sum(single_kernel)

        # reshape tensors into (bash_size * (time_steps / stride), stride, features)
        kernel = tf.reshape(kernel, intermediate_shape[1:])
        inputs = tf.reshape(inputs, intermediate_shape)

        # broadcasting kernel to inputs batch dimension
        linear_decay = tf.reduce_sum(kernel * inputs, axis=1)
        linear_decay = tf.reshape(linear_decay, output_shape)
        return linear_decay


class Return(Layer):
    """
    计算公式为每个stride最后一个数除以第一个数再减去一
    """

    def __init__(self, stride=10, **kwargs):
        """
        :param stride: time steps需要是stride的整数倍
        """
        if stride <= 1:
            raise Exception("Illegal Argument: stride should be greater than 1")
        super(Return, self).__init__(**kwargs)
        self.stride = stride

    def call(self, inputs, *args, **kwargs):
        """
        :param inputs: 输入dimension为(batch_size, time_steps, features)
        :return: return dimension 为(batch_size, time_steps / stride, features)
        """
        time_steps, _, _ = __get_dimensions__(inputs, self.stride)

        # get the endings of each strides as numerators
        numerators = inputs[:, (self.stride - 1)::self.stride, :]

        # get the beginnings of each strides as denominators
        denominators = inputs[:, 0::self.stride, :]

        return tf.math.divide_no_nan(numerators, denominators) - 1.0


class Covariance(Layer):
    """
    计算每个stride每两个feature之间的covariance大小，输出feature数量为features * (features - 1) / 2
    """

    def __init__(self, stride=10, **kwargs):
        """
        :param stride: time steps需要是stride的整数倍
        """
        if stride <= 1:
            raise Exception("Illegal Argument: stride should be greater than 1")
        super(Covariance, self).__init__(**kwargs)
        self.stride = stride

    def call(self, inputs, *args, **kwargs):
        """
        :param inputs: 输入dimension为(batch_size, time_steps, features)
        :return: return dimension 为(batch_size, time_steps / stride, features * (features - 1) / 2)
        """
        time_steps, features, output_length = __get_dimensions__(inputs, self.stride)
        output_features = int(features * (features - 1) / 2)
        output_shape = (-1, output_length, output_features)

        # compute means for each stride
        means = tf.nn.avg_pool(inputs, ksize=self.stride, strides=self.stride, padding="VALID")

        # subtract means for each stride
        means_subtracted = tf.subtract(inputs, tf.repeat(means, self.stride, axis=1))
        means_subtracted = tf.reshape(means_subtracted, (-1, self.stride, features))

        # compute covariance matrix
        covariance_matrix = tf.einsum("ijk,ijm->ikm", means_subtracted, means_subtracted)
        covariance_matrix = covariance_matrix / (self.stride - 1)

        # get the lower part of the covariance matrix without the diagonal elements
        mask = __lower_triangle_without_diagonal_mask__(covariance_matrix)
        covariances = tf.boolean_mask(covariance_matrix, mask)
        covariances = tf.reshape(covariances, output_shape)
        return covariances


class Correlation(Layer):
    """
    计算每个stride每两个feature之间的correlation coefficient，输出feature数量为features * (features - 1) / 2
    """

    def __init__(self, stride=10, **kwargs):
        """
        :param stride: time steps需要是stride的整数倍
        """
        if stride <= 1:
            raise Exception("Illegal Argument: stride should be greater than 1")
        super(Correlation, self).__init__(**kwargs)
        self.stride = stride

    def call(self, inputs, *args, **kwargs):
        """
        :param inputs: 输入dimension为(batch_size, time_steps, features)
        :return: return dimension 为(batch_size, time_steps / stride, features * (features - 1) / 2)
        """
        time_steps, features, output_length = __get_dimensions__(inputs, self.stride)
        output_features = int(features * (features - 1) / 2)
        output_shape = (-1, output_length, output_features)
        intermediate_shape = (-1, self.stride, features)

        # compute means for each stride
        means = tf.nn.avg_pool(inputs, ksize=self.stride, strides=self.stride, padding="VALID")

        # subtract means for each stride
        means_subtracted = tf.subtract(inputs, tf.repeat(means, self.stride, axis=1))
        means_subtracted = tf.reshape(means_subtracted, intermediate_shape)

        # compute standard deviations for each strides
        squared_diff = tf.square(means_subtracted)
        mean_squared_error = tf.reduce_mean(squared_diff, axis=1)
        std = tf.sqrt(mean_squared_error)

        # get denominator of correlation matrix
        denominator_matrix = tf.einsum("ik,im->ikm", std, std)

        # compute covariance matrix
        covariance_matrix = tf.einsum("ijk,ijm->ikm", means_subtracted, means_subtracted)
        covariance_matrix = covariance_matrix / self.stride

        # take the lower triangle of each matrix without diagonal
        mask = __lower_triangle_without_diagonal_mask__(covariance_matrix)
        covariances = tf.reshape(tf.boolean_mask(covariance_matrix, mask), output_shape)
        denominators = tf.reshape(tf.boolean_mask(denominator_matrix, mask), output_shape)

        return tf.math.divide_no_nan(covariances, denominators)


class FeatureExpansion(Layer):
    """
    该层扩张时间序列的feature数量，并通过stride缩短时间序列长度，其包括一下一些feature:
    - standard deviation
    - mean / standard deviation 
    - linear decay average
    - return of each stride
    - covariance of each two features for each stride
    - correlation coefficient of each two features for each stride
    """

    def __init__(self, stride=10, **kwargs):
        """
        :param stride: time steps需要是stride的整数倍
        """
        if stride <= 1:
            raise Exception("Illegal Argument: stride should be greater than 1")
        super(FeatureExpansion, self).__init__(**kwargs)
        self.stride = stride
        self.std = Std(stride=stride)
        self.z_score = ZScore(stride=stride)
        self.linear_decay = LinearDecay(stride=stride)
        self.return_ = Return(stride=stride)
        self.covariance = Covariance(stride=stride)
        self.correlation = Correlation(stride=stride)

    def call(self, inputs, *args, **kwargs):
        """
        :param inputs: 输入dimension为(batch_size, time_steps, features)
        :return: return dimension 为(batch_size, time_steps / stride, features * (features + 3))
        """
        std_output = self.std(inputs)
        z_score_output = self.z_score(inputs)
        decay_linear_output = self.linear_decay(inputs)
        return_output = self.return_(inputs)
        covariance_output = self.covariance(inputs)
        correlation_output = self.correlation(inputs)
        return tf.concat([std_output,
                          z_score_output,
                          decay_linear_output,
                          return_output,
                          covariance_output,
                          correlation_output], axis=2)


class AlphaNetV3:
    """
    复现华泰金工 alpha net V3 版本

    ```
    input: (batch_size, history time steps, features)

            +-> expand features (stride = 5)  -> BN -> GRU -> BN -+
    input --|                                                     |- concat -> Dense(linear)
            +-> expand features (stride = 10) -> BN -> GRU -> BN -+
    ```

    (BN: batch normalization)
    """

    def __init__(self,
                 optimizer=tf.keras.optimizers.Adam,
                 alpha=0.0001,
                 loss="MSE"):
        inputs = tf.keras.Input(shape=(30, 15))
        expanded_10 = FeatureExpansion(stride=10)(inputs)
        expanded_5 = FeatureExpansion(stride=5)(inputs)
        normalized_10 = tfl.BatchNormalization()(expanded_10)
        normalized_5 = tfl.BatchNormalization()(expanded_5)
        gru_10 = tfl.GRU(units=30)(normalized_10)
        gru_5 = tfl.GRU(units=30)(normalized_5)
        normalized_10 = tfl.BatchNormalization()(gru_10)
        normalized_5 = tfl.BatchNormalization()(gru_5)
        concat = tfl.Concatenate(axis=-1)([normalized_10, normalized_5])
        outputs = tfl.Dense(1, activation="linear", kernel_initializer="truncated_normal")(concat)
        self.__model = tf.keras.Model(inputs=inputs, outputs=outputs)
        self.__model.compile(optimizer(alpha), loss=loss)

    def model(self):
        """
        返回model
        :return: tensorflow model
        """
        return self.__model

    def fit(self, *args, **kwargs):
        """
        训练的函数
        :return: history object
        """
        self.__model.fit(*args, **kwargs)

    def predict(self, *args, **kwargs):
        """
        预测的函数
        :return: 预测值
        """
        self.__model.predict(*args, **kwargs)


class UpDownAccuracy(tf.keras.metrics.Metric):

    def __init__(self, name='up_down_accuracy', **kwargs):
        super(UpDownAccuracy, self).__init__(name=name, **kwargs)
        self.up_down_correct_count = self.add_weight(name='ud_count', initializer='zeros', shape=(), dtype=tf.float64)
        self.length = self.add_weight(name='len', initializer='zeros', shape=(), dtype=tf.float64)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true > 0, tf.float64)
        y_pred = tf.cast(y_pred > 0, tf.float64)
        length = tf.cast(len(y_true), tf.float64)
        correct_count = length - tf.reduce_sum(tf.abs(y_true - y_pred))

        self.length.assign_add(length)
        self.up_down_correct_count.assign_add(correct_count)

    def result(self):
        if self.length == 0.0:
            return 0.0
        return self.up_down_correct_count / self.length

    def reset_states(self):
        self.up_down_correct_count.assign(0.0)
        self.length.assign(0.0)


class TimeSeriesData:
    """
    储存一个个股的数据信息及预测目标，全部使用numpy，日期格式为整数: `YYYYMMDD`.
    """

    def __init__(self, dates: np.ndarray, data: np.ndarray, labels: np.ndarray):
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


def __generator__(data, label, generation_list, history_length):
    for series_i, i in generation_list:
        x_data = data[series_i][i: i + history_length]
        y_data = label[series_i][i + history_length - 1]

        # 如果该序列的历史片段内有缺失数据则跳过该数据
        if (tf.reduce_sum(tf.cast(tf.math.is_nan(x_data), tf.int64)) == 0 and
                tf.reduce_sum(tf.cast(tf.math.is_nan(y_data), tf.int64)) == 0):
            x = tf.constant(x_data)
            y = tf.constant(y_data)
            yield x, y


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
        if len(time_series_list) == 0:
            raise Exception("Empty data")
        self.__feature_counts = time_series_list[0].data.shape[1]
        for series in time_series_list:
            if series.data.shape[1] != self.__feature_counts:
                raise Exception("time series do not have the same number of features")

        # gather distinct dates
        self.__distinct_dates = np.unique([date for stock in time_series_list for date in stock.dates])
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
        dates_positions = {date: index for index, date in enumerate(self.__distinct_dates)}
        dates_position_mapper = np.vectorize(lambda d: dates_positions[d])
        for i, series in enumerate(time_series_list):
            position_index = dates_position_mapper(series.dates)
            self.__data[i, position_index, :] = series.data
            self.__labels[i, position_index] = series.labels

        # convert to tensor constant
        self.__data = tf.constant(self.__data)
        self.__labels = tf.constant(self.__labels)
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
                           for i in range(0, length - self.__history_length + 1, self.__sample_step)
                           for series_i in range(len(data))]

        if order == "shuffle":
            np.random.shuffle(generation_list)
        elif order == "by_date":
            pass
        elif order == "by_series":
            generation_list = sorted(generation_list, key=lambda k: k[0])
        else:
            raise Exception("wrong order argument, choose from `shuffle`, `by_date`, and `by_series`")

        return data, label, generation_list, self.__history_length

    def get(self, start_date, order="by_date"):
        """
        :param start_date: 该轮训练开始日期，整数YYYYMMDD
        :param order: 有三种顺序: shuffle, by_date, by_series, 分别为随机打乱股票和时间，按时间顺序优先，按股票顺序优先
        :return: tensorflow dataset, (train, val)
        """

        if type(start_date) is not int:
            raise Exception("start date should be an integer YYYYMMDD")

        # find the actual starting date
        after_start_date = self.__distinct_dates >= start_date
        if np.sum(after_start_date) < self.__train_length + self.__validate_length:
            raise Exception("date range exceeded end of dates")

        # get train, val periods
        train_start_index = np.argmin(self.__distinct_dates[after_start_date])
        train_end_index = train_start_index + self.__train_length
        val_start_index = train_end_index - self.__history_length + self.__sample_step
        val_end_index = train_end_index + self.__validate_length

        train_generator_args = self.__get_generator_args__(train_start_index, train_end_index, order=order)
        val_generator_args = self.__get_generator_args__(val_start_index, val_end_index, order=order)

        # get rolling sample generator
        train_dataset = tf.data.Dataset.from_generator(__generator__,
                                                       args=train_generator_args,
                                                       output_types=(tf.float32, tf.float32),
                                                       output_shapes=((self.__history_length,
                                                                       self.__feature_counts), ()))
        # 通过传递tensor args比直接由generator从环境抓取数据运行速度更快，只不过args必须是rectangular tensors
        val_dataset = tf.data.Dataset.from_generator(__generator__,
                                                     args=val_generator_args,
                                                     output_types=(tf.float32, tf.float32),
                                                     output_shapes=((self.__history_length,
                                                                     self.__feature_counts), ()))

        return train_dataset, val_dataset


# unit test
if __name__ == "__main__":

    import pandas as pd
    from tqdm import tqdm

    # 测试数据准备
    csi = pd.read_csv("./data/CSI500.zip", dtype={"代码": "category", "简称": "category"})
    csi.drop(columns=["简称"], inplace=True)

    # 新增特征
    csi["close/free_turn"] = csi["收盘价(元)"] / csi["换手率(基准.自由流通股本)"]
    csi["open/turn"] = csi["开盘价(元)"] / csi["换手率(%)"]
    csi["volume/low"] = csi["成交量(股)"] / csi["最低价(元)"]
    csi["vwap/high"] = csi["均价"] / csi["最高价(元)"]
    csi["low/high"] = csi["最低价(元)"] / csi["最高价(元)"]
    csi["vwap/close"] = csi["均价"] / csi["收盘价(元)"]

    # 计算十日回报
    trading_dates = csi["日期"].unique()
    trading_dates.sort()
    dates_shift_dictionary = dict(zip(trading_dates[10:], trading_dates[:-10]))
    csi_slice = csi[["代码", "日期", "收盘价(元)"]].copy()
    csi_slice_date_shift = csi[["代码", "日期", "收盘价(元)"]].copy()
    csi_slice_date_shift["日期"] = csi_slice_date_shift["日期"].map(lambda x: dates_shift_dictionary.get(x, None))
    csi_slice_date_shift.rename(columns={"收盘价(元)": "10交易日后收盘价(元)"}, inplace=True)
    csi_slice_date_shift.dropna(inplace=True)
    csi_slice_date_shift["日期"] = [d for d in csi_slice_date_shift["日期"]]
    csi_slice = csi_slice.merge(csi_slice_date_shift, how="inner", left_on=["代码", "日期"], right_on=["代码", "日期"])
    csi_slice["10日回报率"] = csi_slice["10交易日后收盘价(元)"] / csi_slice["收盘价(元)"] - 1
    csi_slice.drop(columns=["收盘价(元)", "10交易日后收盘价(元)"], inplace=True)
    csi = csi_slice.merge(csi, how="inner", left_on=["代码", "日期"], right_on=["代码", "日期"])

    codes = csi.代码.cat.categories
    stock_data = []
    for code in codes:
        table_part = csi.loc[csi.代码 == code, :]
        stock_data.append(TimeSeriesData(dates=table_part["日期"].values,
                                         data=table_part.iloc[:, 3:].values,
                                         labels=table_part["10日回报率"].values))

    test_data = tf.constant([stock_data[0].data[0:30],
                             stock_data[0].data[2:32],
                             stock_data[0].data[4:34]], dtype=tf.float32)

    # 补全全部stock与日期组合，用于手动生成batch对比测试
    trading_dates = csi["日期"].unique()
    trading_dates.sort()
    full_index = pd.DataFrame([[s, d] for s in codes for d in trading_dates])
    full_index.columns = ["代码", "日期"]
    full_csi = full_index.merge(csi, how="left", left_on=["代码", "日期"], right_on=["代码", "日期"])

    def __is_all_close__(data1, data2, **kwargs):
        return np.all(np.isclose(data1, data2, **kwargs))


    def __compute_test_covariance__(data):
        data = data.numpy()
        covariances = []
        cov_mat = np.cov(data.T)
        for i in range(cov_mat.shape[0]):
            for m in range(i):
                covariances.append(cov_mat[i, m])
        return covariances


    def __compute_test_correlation__(data):
        data = data.numpy()
        correlations = []
        corr_coefficient = np.corrcoef(data.T)
        for i in range(corr_coefficient.shape[0]):
            for m in range(i):
                correlations.append(corr_coefficient[i, m])
        return correlations

    def __get_batches__(start_date):
        train_val_generator = TrainValData(stock_data)
        train, val = train_val_generator.get(start_date)
        first_train = next(iter(train.batch(500)))
        first_val = next(iter(val.batch(500)))
        last_train = None
        last_val = None

        for b in iter(train.batch(500)):
            last_train = b

        for b in iter(val.batch(500)):
            last_val = b

        return first_train, first_val, last_train, last_val

    def __get_n_batches__(start_date_index, end_date_index, n=2, step=2):
        data_list = []
        running_index = [(start_date_index + day, end_date_index + day, co)
                         for day in range(0, step * n, step)
                         for co in codes]
        for start, end, co in tqdm(running_index):
            start_date = trading_dates[start]
            end_date = trading_dates[end]
            df = full_csi.loc[np.logical_and(np.logical_and(full_csi["代码"] == co, full_csi["日期"] <= end_date),
                                             full_csi["日期"] >= start_date), :].iloc[:, 3:].values
            if np.sum(pd.isnull(df)) == 0:
                data_list.append(df)

        return data_list

    print("===Testing custom layers===")

    # test std
    s = Std()(test_data)
    test_result = []
    for j in range(len(test_data)):
        test1 = __is_all_close__(s[j][0], np.std(test_data[j][0:10], axis=0))
        test2 = __is_all_close__(s[j][1], np.std(test_data[j][10:20], axis=0))
        test3 = __is_all_close__(s[j][2], np.std(test_data[j][20:30], axis=0))
        test_result.extend([test1, test2, test3])
    if np.all(test_result):
        print("Std: all tests passed")
    else:
        raise Exception("Std incorrect")

    # test z-score
    z = ZScore()(test_data)
    test_result = []
    for j in range(len(test_data)):
        test1 = __is_all_close__(z[j][0], np.mean(test_data[j][0:10], axis=0) / np.std(test_data[j][0:10], axis=0))
        test2 = __is_all_close__(z[j][1], np.mean(test_data[j][10:20], axis=0) / np.std(test_data[j][10:20], axis=0))
        test3 = __is_all_close__(z[j][2], np.mean(test_data[j][20:30], axis=0) / np.std(test_data[j][20:30], axis=0))
        test_result.extend([test1, test2, test3])
    if np.all(test_result):
        print("z-score: all tests passed")
    else:
        raise Exception("z-score incorrect")

    # test linear decay
    d = LinearDecay()(test_data)
    weights = np.linspace(1, 10, 10)
    test_result = []
    for j in range(len(test_data)):
        test1 = __is_all_close__(d[j][0], np.average(test_data[j][0:10], axis=0, weights=weights))
        test2 = __is_all_close__(d[j][1], np.average(test_data[j][10:20], axis=0, weights=weights))
        test3 = __is_all_close__(d[j][2], np.average(test_data[j][20:30], axis=0, weights=weights))
        test_result.extend([test1, test2, test3])
    if np.all(test_result):
        print("linear decay: all tests passed")
    else:
        raise Exception("linear decay incorrect")

    # test return
    r = Return()(test_data)
    test_result = []
    for j in range(len(test_data)):
        test1 = __is_all_close__(r[j][0], test_data[j][10 - 1] / test_data[j][0] - 1)
        test2 = __is_all_close__(r[j][1], test_data[j][20 - 1] / test_data[j][10] - 1)
        test3 = __is_all_close__(r[j][2], test_data[j][30 - 1] / test_data[j][20] - 1)
        test_result.extend([test1, test2, test3])
    if np.all(test_result):
        print("return: all tests passed")
    else:
        raise Exception("return incorrect")

    # test covariances
    c = Covariance()(test_data)
    test_result = []
    for j in range(len(test_data)):
        test1 = __is_all_close__(c[j][0], __compute_test_covariance__(test_data[j][0:10]))
        test2 = __is_all_close__(c[j][1], __compute_test_covariance__(test_data[j][10:20]))
        test3 = __is_all_close__(c[j][2], __compute_test_covariance__(test_data[j][20:30]))
        test_result.extend([test1, test2, test3])
    if np.all(test_result):
        print("covariance: all tests passed")
    else:
        raise Exception("covariance incorrect")

    # test correlation
    c = Correlation()(test_data)
    test_result = []
    for j in range(len(test_data)):
        test1 = __is_all_close__(c[j][0], __compute_test_correlation__(test_data[j][0:10]), atol=1e-5)
        test2 = __is_all_close__(c[j][1], __compute_test_correlation__(test_data[j][10:20]), atol=1e-5)
        test3 = __is_all_close__(c[j][2], __compute_test_correlation__(test_data[j][20:30]), atol=1e-5)
        test_result.extend([test1, test2, test3])
    if np.all(test_result):
        print("correlation: all tests passed")
    else:
        raise Exception("correlation incorrect")

    print("===Testing AlphaNetV3===")

    alpha_net_v3 = AlphaNetV3()
    alpha_net_v3.model().summary()

    print("===Testing data utility===")

    print("Testing data utility")

    print("Computing tensorflow dataset: 20110101")

    first_batch_train, first_batch_val, last_batch_train, last_batch_val = __get_batches__(20110101)

    print("Comparing first batch of training dataset", flush=True)
    first_data_queue = __get_n_batches__(0, 29, 3)
    if __is_all_close__(first_data_queue[:len(first_batch_train[0])], first_batch_train[0]):
        print("passed", flush=True)
    else:
        raise Exception("failure")

    print("Comparing last batch of training dataset", flush=True)
    last_data_queue = __get_n_batches__(1170 - 2, 1199 - 2)
    if __is_all_close__(last_data_queue[-len(last_batch_train[0]):], last_batch_train[0]):
        print("passed", flush=True)
    else:
        raise Exception("failure")

    print("Comparing first batch of validation dataset", flush=True)
    first_val_data_queue = __get_n_batches__(1200 - 30 + 2, 1201)
    if __is_all_close__(first_val_data_queue[:len(first_batch_val[0])], first_batch_val[0]):
        print("passed", flush=True)
    else:
        raise Exception("failure")

    print("Comparing last batch of training dataset", flush=True)
    last_val_data_queue = __get_n_batches__(1470 - 2, 1499 - 2)
    if __is_all_close__(last_val_data_queue[-len(last_batch_val[0]):], last_batch_val[0]):
        print("passed", flush=True)
    else:
        raise Exception("failure")

    start_basis = np.where(trading_dates == 20110601)

    print("Computing tensorflow dataset: 20110601")

    first_batch_train, first_batch_val, last_batch_train, last_batch_val = __get_batches__(20110601)

    print("Comparing first batch of training dataset", flush=True)
    first_data_queue = __get_n_batches__(start_basis + 0, start_basis + 29, 3)
    if __is_all_close__(first_data_queue[:len(first_batch_train[0])], first_batch_train[0]):
        print("passed", flush=True)
    else:
        raise Exception("failure")

    print("Comparing last batch of training dataset", flush=True)
    last_data_queue = __get_n_batches__(start_basis + 1170 - 2, start_basis + 1199 - 2)
    if __is_all_close__(last_data_queue[-len(last_batch_train[0]):], last_batch_train[0]):
        print("passed", flush=True)
    else:
        raise Exception("failure")

    print("Comparing first batch of validation dataset", flush=True)
    first_val_data_queue = __get_n_batches__(start_basis + 1200 - 30 + 2, start_basis + 1201)
    if __is_all_close__(first_val_data_queue[:len(first_batch_val[0])], first_batch_val[0]):
        print("passed", flush=True)
    else:
        raise Exception("failure")

    print("Comparing last batch of training dataset", flush=True)
    last_val_data_queue = __get_n_batches__(start_basis + 1470 - 2, start_basis + 1499 - 2)
    if __is_all_close__(last_val_data_queue[-len(last_batch_val[0]):], last_batch_val[0]):
        print("passed", flush=True)
    else:
        raise Exception("failure")

    print("ALL TESTS PASSED")
