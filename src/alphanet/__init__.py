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
import tensorflow as _tf
import tensorflow.keras.layers as _tfl
from tensorflow.keras.layers import Layer as _Layer
from tensorflow.keras.initializers import Initializer as _Initializer
from abc import ABC as _ABC
from abc import abstractmethod as _abstractmethod

__all__ = ["Std",
           "Return",
           "Correlation",
           "LinearDecay",
           "Covariance",
           "ZScore",
           "FeatureExpansion",
           "AlphaNetV3"]


class Std(_Layer):
    """

    Notes:
        计算每个feature各个stride的standard deviation

    """

    def __init__(self, stride: int = 10, **kwargs):
        """

        Args:
            stride (int): time steps需要是stride的整数倍

        """
        if stride <= 1:
            raise ValueError("Illegal Argument: stride should be "
                             "greater than 1")
        super(Std, self).__init__(**kwargs)
        self.stride = stride
        self.intermediate_shape = None

    def build(self, input_shape):
        (features,
         output_length) = __get_dimensions__(input_shape, self.stride)
        self.intermediate_shape = [-1,
                                   output_length,
                                   self.stride,
                                   features]

    def call(self, inputs, *args, **kwargs):
        """

        Args:
            inputs (tensor): 输入dimension为(batch_size, time_steps, features)

        Returns:
            dimension 为(batch_size, time_steps / stride, features)

        """

        # compute means for each stride
        means = _tf.repeat(
            _tf.nn.avg_pool(inputs,
                            ksize=self.stride,
                            strides=self.stride,
                            padding="VALID"),
            self.stride,
            axis=1
        )

        # subtract means for each stride
        squared_diff = _tf.square(_tf.subtract(inputs, means))
        squared_diff = _tf.reshape(squared_diff, self.intermediate_shape)

        # compute standard deviation for each stride
        mean_squared_diff = _tf.reduce_mean(squared_diff, axis=2)
        std = _tf.sqrt(mean_squared_diff)

        return std

    def get_config(self):
        config = super().get_config().copy()
        config.update({'stride': self.stride})
        return config


class ZScore(_Layer):
    """

    Notes:
        并非严格意义上的z-score,
        计算公式为每个feature各个stride的mean除以各自的standard deviation

    """

    def __init__(self, stride: int = 10, **kwargs):
        """

        Args:
            stride (int): time steps需要是stride的整数倍

        """
        if stride <= 1:
            raise ValueError("Illegal Argument: stride should be "
                             "greater than 1")
        super(ZScore, self).__init__(**kwargs)
        self.stride = stride
        self.intermediate_shape = None

    def build(self, input_shape):
        (features,
         output_length) = __get_dimensions__(input_shape, self.stride)
        self.intermediate_shape = [-1,
                                   output_length,
                                   self.stride,
                                   features]

    def call(self, inputs, *args, **kwargs):
        """

        Args:
            inputs (tensor): 输入dimension为(batch_size, time_steps, features)

        Returns:
            dimension 为(batch_size, time_steps / stride, features)

        """

        # compute means for each stride
        means = _tf.nn.avg_pool(inputs,
                                ksize=self.stride,
                                strides=self.stride,
                                padding="VALID")

        # compute standard deviations for each stride
        means_broadcast = _tf.repeat(means, self.stride, axis=1)
        squared_diff = _tf.square(_tf.subtract(inputs, means_broadcast))
        squared_diff = _tf.reshape(squared_diff, self.intermediate_shape)
        mean_squared_diff = _tf.reduce_mean(squared_diff, axis=2)
        std = _tf.sqrt(mean_squared_diff)

        # divide means by standard deviations for each stride
        z_score = _tf.math.divide_no_nan(means, std)
        return z_score

    def get_config(self):
        config = super().get_config().copy()
        config.update({'stride': self.stride})
        return config


class LinearDecay(_Layer):
    """

    Notes:
        以线性衰减为权重，计算每个feature各个stride的均值：
        如stride为10，则某feature该stride的权重为(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)

    """

    def __init__(self, stride=10, **kwargs):
        """

        Args:
            stride (int): time steps需要是stride的整数倍

        """
        if stride <= 1:
            raise ValueError("Illegal Argument: stride should be "
                             "greater than 1")
        super(LinearDecay, self).__init__(**kwargs)
        self.stride = stride
        self.out_shape = None
        self.intermediate_shape = None

    def build(self, input_shape):
        (features,
         output_length) = __get_dimensions__(input_shape, self.stride)
        self.out_shape = [-1, output_length, features]
        self.intermediate_shape = [-1, self.stride, features]

    def call(self, inputs, *args, **kwargs):
        """

        Args:
            inputs (tensor): 输入dimension为(batch_size, time_steps, features)

        Returns:
            dimension 为(batch_size, time_steps / stride, features)

        """

        # get linear decay kernel
        single_kernel = _tf.linspace(1.0, self.stride, num=self.stride)
        kernel = _tf.repeat(single_kernel, self.intermediate_shape[2])
        kernel = kernel / _tf.reduce_sum(single_kernel)

        # reshape tensors into:
        # (bash_size * (time_steps / stride), stride, features)
        kernel = _tf.reshape(kernel, self.intermediate_shape[1:])
        inputs = _tf.reshape(inputs, self.intermediate_shape)

        # broadcasting kernel to inputs batch dimension
        linear_decay = _tf.reduce_sum(kernel * inputs, axis=1)
        linear_decay = _tf.reshape(linear_decay, self.out_shape)
        return linear_decay

    def get_config(self):
        config = super().get_config().copy()
        config.update({'stride': self.stride})
        return config


class Return(_Layer):
    """

    Notes:
        计算公式为每个stride最后一个数除以第一个数再减去一

    """

    def __init__(self, stride=10, **kwargs):
        """

        Args:
            stride (int): time steps需要是stride的整数倍

        """

        if stride <= 1:
            raise ValueError("Illegal Argument: stride should be "
                             "greater than 1")
        super(Return, self).__init__(**kwargs)
        self.stride = stride

    def build(self, input_shape):
        time_steps = input_shape[1]
        if time_steps % self.stride != 0:
            raise ValueError("Error, time_steps 应该是 stride的整数倍")

    def call(self, inputs, *args, **kwargs):
        """

        Args:
            inputs (tensor): 输入dimension为(batch_size, time_steps, features)

        Returns:
            dimension 为(batch_size, time_steps / stride, features)

        """

        # get the endings of each strides as numerators
        numerators = inputs[:, (self.stride - 1)::self.stride, :]

        # get the beginnings of each strides as denominators
        denominators = inputs[:, 0::self.stride, :]

        return _tf.math.divide_no_nan(numerators, denominators) - 1.0

    def get_config(self):
        config = super().get_config().copy()
        config.update({'stride': self.stride})
        return config


class _OuterProductLayer(_Layer, _ABC):

    def __init__(self, stride=10, **kwargs):
        """

        Args:
            stride (int): time steps需要是stride的整数倍

        """
        if stride <= 1:
            raise ValueError("Illegal Argument: stride should be "
                             "greater than 1")
        super(_OuterProductLayer, self).__init__(**kwargs)
        self.stride = stride
        self.intermediate_shape = None
        self.out_shape = None
        self.lower_mask = None

    def build(self, input_shape):
        (features,
         output_length) = __get_dimensions__(input_shape, self.stride)
        self.intermediate_shape = (-1, self.stride, features)
        output_features = int(features * (features - 1) / 2)
        self.out_shape = (-1, output_length, output_features)
        self.lower_mask = LowerNoDiagonalMask()((features, features))

    def get_config(self):
        config = super().get_config().copy()
        config.update({'stride': self.stride})
        return config

    @_abstractmethod
    def call(self, inputs, *args, **kwargs):
        ...


class Covariance(_OuterProductLayer):
    """

    Notes:
        计算每个stride每两个feature之间的covariance大小，
        输出feature数量为features * (features - 1) / 2

    """

    def call(self, inputs, *args, **kwargs):
        """

        Args:
            inputs (tensor): 输入dimension为(batch_size, time_steps, features)

        Returns:
            dimension 为(batch_size, time_steps / stride,
            features * (features - 1) / 2)

        """

        # compute means for each stride
        means = _tf.nn.avg_pool(inputs,
                                ksize=self.stride,
                                strides=self.stride,
                                padding="VALID")

        # subtract means for each stride
        means_broadcast = _tf.repeat(means, self.stride, axis=1)
        means_subtracted = _tf.subtract(inputs, means_broadcast)
        means_subtracted = _tf.reshape(means_subtracted,
                                       self.intermediate_shape)

        # compute covariance matrix
        covariance_matrix = _tf.einsum("ijk,ijm->ikm",
                                       means_subtracted,
                                       means_subtracted)
        covariance_matrix = covariance_matrix / (self.stride - 1)

        # get the lower part of the covariance matrix
        # without the diagonal elements
        covariances = _tf.boolean_mask(covariance_matrix,
                                       self.lower_mask,
                                       axis=1)
        covariances = _tf.reshape(covariances, self.out_shape)
        return covariances


class Correlation(_OuterProductLayer):
    """
    Notes:
        计算每个stride每两个feature之间的correlation coefficient，
        输出feature数量为features * (features - 1) / 2
    """

    def call(self, inputs, *args, **kwargs):
        """

        Args:
            inputs (tensor): 输入dimension为(batch_size, time_steps, features)

        Returns:
            dimension 为(batch_size, time_steps / stride,
            features * (features - 1) / 2)

        """

        # compute means for each stride
        means = _tf.nn.avg_pool(inputs,
                                ksize=self.stride,
                                strides=self.stride,
                                padding="VALID")

        # subtract means for each stride
        means_broadcast = _tf.repeat(means, self.stride, axis=1)
        means_subtracted = _tf.subtract(inputs, means_broadcast)
        means_subtracted = _tf.reshape(means_subtracted,
                                       self.intermediate_shape)

        # compute standard deviations for each strides
        squared_diff = _tf.square(means_subtracted)
        mean_squared_error = _tf.reduce_mean(squared_diff, axis=1)
        std = _tf.sqrt(mean_squared_error)

        # get denominator of correlation matrix
        denominator_matrix = _tf.einsum("ik,im->ikm", std, std)

        # compute covariance matrix
        covariance_matrix = _tf.einsum("ijk,ijm->ikm",
                                       means_subtracted,
                                       means_subtracted)
        covariance_matrix = covariance_matrix / self.stride

        # take the lower triangle of each matrix without diagonal
        covariances = _tf.boolean_mask(covariance_matrix,
                                       self.lower_mask,
                                       axis=1)
        denominators = _tf.boolean_mask(denominator_matrix,
                                        self.lower_mask,
                                        axis=1)
        correlations = _tf.math.divide_no_nan(covariances, denominators)
        correlations = _tf.reshape(correlations, self.out_shape)
        return correlations


class FeatureExpansion(_Layer):
    """

    Notes:
        该层扩张时间序列的feature数量，并通过stride缩短时间序列长度，
        其包括一下一些feature:

            - standard deviation

            - mean / standard deviation

            - linear decay average

            - return of each stride

            - covariance of each two features for each stride

            - correlation coefficient of each two features for each stride

    """

    def __init__(self, stride=10, **kwargs):
        """

        Args:
            stride (int): time steps需要是stride的整数倍

        """
        if type(stride) is not int or stride <= 1:
            raise ValueError("Illegal Argument: stride should be an integer "
                             "greater than 1")
        super(FeatureExpansion, self).__init__(**kwargs)
        self.stride = stride
        self.std = None
        self.z_score = None
        self.linear_decay = None
        self.return_ = None
        self.covariance = None
        self.correlation = None

    def build(self, input_shape):
        self.std = _tf.function(Std(stride=self.stride))
        self.z_score = _tf.function(ZScore(stride=self.stride))
        self.linear_decay = _tf.function(LinearDecay(stride=self.stride))
        self.return_ = _tf.function(Return(stride=self.stride))
        self.covariance = _tf.function(Covariance(stride=self.stride))
        self.correlation = _tf.function(Correlation(stride=self.stride))

    def call(self, inputs, *args, **kwargs):
        """

        Args:
            inputs (tensor): 输入dimension为(batch_size, time_steps, features)

        Returns:
            dimension 为(batch_size, time_steps / stride,
            features * (features + 3))

        """
        std_output = self.std(inputs)
        z_score_output = self.z_score(inputs)
        decay_linear_output = self.linear_decay(inputs)
        return_output = self.return_(inputs)
        covariance_output = self.covariance(inputs)
        correlation_output = self.correlation(inputs)
        return _tf.concat([std_output,
                           z_score_output,
                           decay_linear_output,
                           return_output,
                           covariance_output,
                           correlation_output], axis=2)

    def get_config(self):
        config = super().get_config().copy()
        config.update({'stride': self.stride})
        return config


class AlphaNetV3:
    """

    Notes:
        复现华泰金工 alpha net V3 版本
        ::

            input: (batch_size, history time steps, features)

                            stride = 5
                    +-> expand -> BN -> GRU -> BN -+
            input --|       stride = 10            |- concat -> Dense(linear)
                    +-> expand -> BN -> GRU -> BN -+

        (BN: batch normalization)

    """

    def __init__(self,
                 input_shape: (int, int),
                 optimizer=_tf.keras.optimizers.Adam,
                 learning_rate=0.0001,
                 loss="MSE",
                 dropout=0.0,
                 l2=0.001,
                 metrics=None):
        """alpha net v3

        Notes:
            alpha net v3 版本的全tensorflow实现，结构详见代码展开

        Args:
            input_shape: (int, int), 分别代表 (历史长度, 特征数量)
            optimizer: 优化器
            learning_rate: 优化参数，学习速度
            loss: 损失函数
            dropout: 跟在特征扩张以及Batch Normalization之后的dropout，默认无dropout
            l2: 输出层的l2-regularization参数
            metrics: 训练过程中的metric

        """
        inputs = _tf.keras.Input(shape=input_shape)
        expanded_10 = FeatureExpansion(stride=10)(inputs)
        expanded_5 = FeatureExpansion(stride=5)(inputs)
        normalized_10 = _tfl.BatchNormalization()(expanded_10)
        normalized_5 = _tfl.BatchNormalization()(expanded_5)
        dropout_10 = _tfl.Dropout(dropout)(normalized_10)
        dropout_5 = _tfl.Dropout(dropout)(normalized_5)
        gru_10 = _tfl.GRU(units=30)(dropout_10)
        gru_5 = _tfl.GRU(units=30)(dropout_5)
        normalized_10 = _tfl.BatchNormalization()(gru_10)
        normalized_5 = _tfl.BatchNormalization()(gru_5)
        concat = _tfl.Concatenate(axis=-1)([normalized_10, normalized_5])
        regularize = _tf.keras.regularizers.l2(l2)
        outputs = _tfl.Dense(1, activation="linear",
                             kernel_initializer="truncated_normal",
                             kernel_regularizer=regularize)(concat)
        self.__model = _tf.keras.Model(inputs=inputs, outputs=outputs)
        self.__model.compile(optimizer(learning_rate), loss=loss, metrics=metrics)

    def model(self):
        """返回tensorflow模型

        Returns:
            ``tensorflow.keras.models.Model``

        """
        return self.__model

    def save(self, filepath: str):
        """保存模型参数以及模型的计算graph

        Args:
            filepath: 保存模型的文件路径

        """
        self.__model.save(filepath)

    def save_weights(self, filepath):
        """保存模型参数

        Args:
            filepath: 文件路径

        """
        self.__model.save_weights(filepath)

    def load_weights(self, filepath):
        """载入模型参数

        Args:
            filepath: 参数文件位置

        """
        self.__model.load_weights(filepath)

    def predict(self, x, batch_size=500):
        """批量预测

        Args:
            x: numpy array, tensor, 或者tensorflow dataset
            batch_size: 批处理大小

        Returns:
            预测值

        """
        return self.__model.predict(x, batch_size=batch_size)

    def __call__(self, *args, **kwargs):
        return self.__model(*args, **kwargs)


class LowerNoDiagonalMask(_Initializer):
    """
    Notes:
        Provide a mask giving the lower triangular of a matrix
        without diagonal elements.
    """

    def __init__(self):
        super(LowerNoDiagonalMask, self).__init__()

    def __call__(self, shape, **kwargs):
        ones = _tf.ones(shape)
        mask_lower = _tf.linalg.band_part(ones, -1, 0)
        mask_diag = _tf.linalg.band_part(ones, 0, 0)
        # lower triangle removing the diagonal elements
        mask = _tf.cast(mask_lower - mask_diag, dtype=_tf.bool)
        return mask


def __get_dimensions__(input_shape, stride):
    """

    Notes:
        compute output shapes

    Args:
        input_shape: pass the inputs of layer to the function
        stride (int): the stride of the custom layer

    Returns:
        (features, output_length)

    """
    if type(stride) is not int or stride <= 1:
        raise ValueError("Illegal Argument: stride should be an integer "
                         "greater than 1")
    time_steps = input_shape[1]
    features = input_shape[2]
    output_length = time_steps // stride

    if time_steps % stride != 0:
        raise ValueError("Error, time_steps 应该是 stride的整数倍")

    return features, output_length
