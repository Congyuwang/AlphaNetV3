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
        self.up_down_correct_count = self.add_weight(name='ud_count', initializer='zeros')
        self.length = self.add_weight(name='len', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true > 0, tf.float32)
        y_pred = tf.cast(y_pred > 0, tf.float32)
        length = tf.cast(len(y_true), tf.float32)
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


class TrainValData:
    """
    获取train, validation tensorflow dataset
    """

    def __init__(self,
                 time_series_list: list[TimeSeriesData],
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
        if len(time_series_list) == 0:
            raise Exception("Empty data")
        self.__dates_list = [stock.dates for stock in time_series_list]
        self.__distinct_dates = np.unique([date for dates in self.__dates_list for date in dates])
        self.__distinct_dates.sort()
        self.__data_list = [stock.data for stock in time_series_list]
        self.__labels_list = [stock.labels for stock in time_series_list]
        self.__train_length = train_length
        self.__validate_length = validate_length
        self.__history_length = history_length
        self.__sample_step = sample_step

    def __get_generator__(self, date_index, order="shuffle"):
        """
        该函数根据每个股票的日期范围给出一个generation_list
        :param date_index: 每个时间序列的日期范围, boolean mask
        :param order: 有三种顺序: shuffle, by_date, by_series
        :return: generator
        """
        dates = [date[dates] for date, dates in zip(self.__dates_list, date_index)]
        data = [data[dates] for data, dates in zip(self.__data_list, date_index)]
        label = [labels[dates] for labels, dates in zip(self.__labels_list, date_index)]
        generation_list = [(stock_i, i, dates[stock_i][i]) for stock_i in range(len(data))
                           for i in range(0, len(data[stock_i]) - self.__history_length + 1, self.__sample_step)]

        if order == "shuffle":
            np.random.shuffle(generation_list)
        elif order == "by_date":
            generation_list = sorted(generation_list, key=lambda k: k[2])
        elif order == "by_series":
            pass
        else:
            raise Exception("wrong order argument, choose from `shuffle`, `by_date`, and `by_series`")

        generation_list = [(stock_i, i) for stock_i, i, _ in generation_list]

        def generator():
            for stock_i, i in generation_list:
                x = tf.constant(data[stock_i][i: i + self.__history_length])
                y = tf.constant(label[stock_i][i + self.__history_length - 1])
                yield x, y

        return generator

    def get(self, start_date, order="by_date"):
        """
        :param start_date: 该轮训练开始日期，整数YYYYMMDD
        :param order: 有三种顺序: shuffle, by_date, by_series, 分别为随机打乱股票和时间，按时间顺序优先，按股票顺序优先
        :return: tensorflow dataset, (train, val)
        """

        data_dim = self.__data_list[0].shape[1]
        if type(start_date) is not int:
            raise Exception("start date should be an integer YYYYMMDD")

        # find the actual starting date
        after_start_date = self.__distinct_dates >= start_date
        if np.sum(after_start_date) < self.__train_length + self.__validate_length:
            raise Exception("date range exceeded end of dates")
        train_start_date = np.min(self.__distinct_dates[after_start_date])

        train_start_index = np.where(self.__distinct_dates == train_start_date)[0]
        train_end_index = train_start_index + self.__train_length
        val_start_index = train_end_index - self.__history_length + 1
        val_end_index = train_end_index + self.__validate_length

        # get training, validating data date ranges
        train_start_date = train_start_date
        train_end_date = self.__distinct_dates[train_end_index]
        val_start_index = self.__distinct_dates[val_start_index]
        val_end_date = self.__distinct_dates[val_end_index]

        # generate train_data and val_data
        train_dates_index = [np.logical_and(date < train_end_date, date >= train_start_date)
                             for date in self.__dates_list]
        val_dates_index = [np.logical_and(date < val_end_date, date >= val_start_index)
                           for date in self.__dates_list]

        train_generator = self.__get_generator__(train_dates_index, order=order)
        val_generator = self.__get_generator__(val_dates_index, order=order)

        # get rolling sample generator
        train_dataset = tf.data.Dataset.from_generator(train_generator,
                                                       output_types=(tf.float32, tf.float32),
                                                       output_shapes=((self.__history_length, data_dim), ()))

        val_dataset = tf.data.Dataset.from_generator(val_generator,
                                                     output_types=(tf.float32, tf.float32),
                                                     output_shapes=((self.__history_length, data_dim), ()))

        return train_dataset, val_dataset


# unit test
if __name__ == "__main__":

    import pandas as pd

    # 测试数据准备
    csi = pd.read_csv("./data/CSI500.zip", dtype={"代码": "category", "简称": "category"})
    csi.drop(columns=["简称"], inplace=True)
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

    print("Testing custom layers")

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

    print("Testing AlphaNetV3")

    alpha_net_v3 = AlphaNetV3()
    alpha_net_v3.model().summary()

    print("Testing data utility")

    train_val_generator = TrainValData(stock_data)
    train, val = train_val_generator.get(20110101)
    test_data = next(iter(train.batch(500)))
