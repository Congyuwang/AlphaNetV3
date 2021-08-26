"""时间序列计算层、神经网络模型定义.

复现华泰金工 alpha net V2、V3 版本.

V2:

```
input: (batch_size, history time steps, features)

                 stride = 5
input -> expand features -> BN -> LSTM -> BN -> Dense(linear)
```

V3:

```
input: (batch_size, history time steps, features)

                stride = 5
        +-> expand features -> BN -> GRU -> BN -+
input --|       stride = 10                     |- concat -> Dense(linear)
        +-> expand features -> BN -> GRU -> BN -+
```

(BN: batch normalization)

version: 0.0.7

author: Congyu Wang

date: 2021-07-29

该module定义了计算不同时间序列特征的层，工程上使用tensorflow
进行高度向量化的计算，训练时较高效。
"""
import tensorflow as _tf
import tensorflow.keras.layers as _tfl
from tensorflow.keras.layers import Layer as _Layer
from tensorflow.keras.initializers import Initializer as _Initializer
from tensorflow.keras import Model as _Model
from .metrics import UpDownAccuracy as _UpDownAccuracy
from abc import ABC as _ABC
from abc import abstractmethod as _abstractmethod

if not "2.3.0" <= _tf.__version__:
    print(f"requires tensorflow version >= 2.3.0, "
          f"current version {_tf.__version__}")
    exit(1)

__all__ = ["Std",
           "Return",
           "Correlation",
           "LinearDecay",
           "Covariance",
           "ZScore",
           "FeatureExpansion",
           "AlphaNetV2",
           "AlphaNetV3",
           "AlphaNetV4",
           "load_model"]


class _StrideLayer(_Layer, _ABC):
    """计算每个stride的统计值的基类."""

    def __init__(self, stride=10, **kwargs):
        """计算每个stride的统计值的基类.

        Args:
            stride (int): time steps需要是stride的整数倍

        """
        if stride <= 1:
            raise ValueError("Illegal Argument: stride should be "
                             "greater than 1")
        super(_StrideLayer, self).__init__(**kwargs)
        self.stride = stride
        self.out_shape = None
        self.intermediate_shape = None

    def build(self, input_shape):
        """构建该层，计算维度信息."""
        (features,
         output_length) = __get_dimensions__(input_shape, self.stride)
        self.out_shape = [-1, output_length, features]
        self.intermediate_shape = [-1, self.stride, features]

    def get_config(self):
        """获取参数，保存模型需要的函数."""
        config = super().get_config().copy()
        config.update({'stride': self.stride})
        return config


class Std(_StrideLayer):
    """计算每个序列各stride的标准差.

    Notes:
        计算每个feature各个stride的standard deviation

    """

    def call(self, inputs, *args, **kwargs):
        """函数主逻辑实现部分.

        Args:
            inputs (tensor): 输入dimension为(batch_size, time_steps, features)

        Returns:
            dimension 为(batch_size, time_steps / stride, features)

        """
        strides = _tf.reshape(inputs, self.intermediate_shape)

        # compute standard deviations for each stride
        std = _tf.math.reduce_std(strides, axis=-2)
        return _tf.reshape(std, self.out_shape)


class ZScore(_StrideLayer):
    """计算每个序列各stride的均值除以其标准差.

    Notes:
        并非严格意义上的z-score,
        计算公式为每个feature各个stride的mean除以各自的standard deviation

    """

    def call(self, inputs, *args, **kwargs):
        """函数主逻辑实现部分.

        Args:
            inputs (tensor): 输入dimension为(batch_size, time_steps, features)

        Returns:
            dimension 为(batch_size, time_steps / stride, features)

        """
        strides = _tf.reshape(inputs, self.intermediate_shape)

        # compute standard deviations for each stride
        std = _tf.math.reduce_std(strides, axis=-2)

        # compute means for each stride
        means = _tf.math.reduce_mean(strides, axis=-2)

        # divide means by standard deviations for each stride
        z_score = _tf.math.divide_no_nan(means, std)
        return _tf.reshape(z_score, self.out_shape)


class LinearDecay(_StrideLayer):
    """计算每个序列各stride的线性衰减加权平均.

    Notes:
        以线性衰减为权重，计算每个feature各个stride的均值：
        如stride为10，则某feature该stride的权重为(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)

    """

    def call(self, inputs, *args, **kwargs):
        """函数主逻辑实现部分.

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


class Return(_Layer):
    """计算每个序列各stride的回报率.

    Notes:
        计算公式为每个stride最后一个数除以第一个数再减去一

    """

    def __init__(self, stride=10, **kwargs):
        """回报率.

        Args:
            stride (int): time steps需要是stride的整数倍

        """
        if stride <= 1:
            raise ValueError("Illegal Argument: stride should be "
                             "greater than 1")
        super(Return, self).__init__(**kwargs)
        self.stride = stride

    def build(self, input_shape):
        """构建该层，计算维度信息."""
        time_steps = input_shape[1]
        if time_steps % self.stride != 0:
            raise ValueError("Error, time_steps 应该是 stride的整数倍")

    def call(self, inputs, *args, **kwargs):
        """函数主逻辑实现部分.

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
        """获取参数，保存模型需要的函数."""
        config = super().get_config().copy()
        config.update({'stride': self.stride})
        return config


class _OuterProductLayer(_Layer, _ABC):

    def __init__(self, stride=10, **kwargs):
        """外乘类的扩张层.

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
        """构建该层，计算维度信息."""
        (features,
         output_length) = __get_dimensions__(input_shape, self.stride)
        self.intermediate_shape = (-1, self.stride, features)
        output_features = int(features * (features - 1) / 2)
        self.out_shape = (-1, output_length, output_features)
        self.lower_mask = _LowerNoDiagonalMask()((features, features))

    def get_config(self):
        """获取参数，保存模型需要的函数."""
        config = super().get_config().copy()
        config.update({'stride': self.stride})
        return config

    @_abstractmethod
    def call(self, inputs, *args, **kwargs):
        """逻辑实现部分."""
        ...


class Covariance(_OuterProductLayer):
    """计算每个stride各时间序列片段的covariance.

    Notes:
        计算每个stride每两个feature之间的covariance大小，
        输出feature数量为features * (features - 1) / 2

    """

    def call(self, inputs, *args, **kwargs):
        """函数主逻辑实现部分.

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
    """计算每个stride各时间序列的相关系数.

    Notes:
        计算每个stride每两个feature之间的correlation coefficient，
        输出feature数量为features * (features - 1) / 2

    """

    def call(self, inputs, *args, **kwargs):
        """函数主逻辑实现部分.

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
    """计算时间序列特征扩张层，汇总6个计算层.

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
        """时间序列特征扩张.

        Args:
            stride (int): time steps需要是stride的整数倍

        """
        if type(stride) is not int or stride <= 1:
            raise ValueError("Illegal Argument: stride should be an integer "
                             "greater than 1")
        super(FeatureExpansion, self).__init__(**kwargs)
        self.stride = stride
        self.std = _tf.function(Std(stride=self.stride))
        self.z_score = _tf.function(ZScore(stride=self.stride))
        self.linear_decay = _tf.function(LinearDecay(stride=self.stride))
        self.return_ = _tf.function(Return(stride=self.stride))
        self.covariance = _tf.function(Covariance(stride=self.stride))
        self.correlation = _tf.function(Correlation(stride=self.stride))

    def call(self, inputs, *args, **kwargs):
        """函数主逻辑实现部分.

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
        """获取参数，保存模型需要的函数."""
        config = super().get_config().copy()
        config.update({'stride': self.stride})
        return config


class AlphaNetV2(_Model):
    """神经网络模型，继承``keras.Model``类.

    alpha net v2版本模型.

    Notes:
        复现华泰金工 alpha net V2 版本

        ``input: (batch_size, history time steps, features)``

    """

    def __init__(self,
                 dropout=0.0,
                 l2=0.001,
                 stride=10,
                 classification=False,
                 categories=0,
                 *args,
                 **kwargs):
        """Alpha net v3.

        Notes:
            alpha net v2 版本的全tensorflow实现，结构详见代码展开

        Args:
            dropout: 跟在特征扩张以及Batch Normalization之后的dropout，默认无dropout
            l2: 输出层的l2-regularization参数

        """
        super(AlphaNetV2, self).__init__(*args, **kwargs)
        self.l2 = l2
        self.dropout = dropout
        self.stride = stride
        self.expanded = FeatureExpansion(stride=self.stride)
        self.normalized = _tfl.BatchNormalization()
        self.dropout = _tfl.Dropout(self.dropout)
        self.lstm = _tfl.LSTM(units=30)
        self.normalized_2 = _tfl.BatchNormalization()
        self.regularizer = _tf.keras.regularizers.l2(self.l2)
        if classification:
            if categories < 1:
                raise ValueError("categories should be at least 1")
            elif categories == 1:
                self.outputs = _tfl.Dense(1, activation="sigmoid",
                                          kernel_initializer="truncated_normal",
                                          kernel_regularizer=self.regularizer)
            else:
                self.outputs = _tfl.Dense(categories, activation="softmax",
                                          kernel_initializer="truncated_normal",
                                          kernel_regularizer=self.regularizer)
        else:
            self.outputs = _tfl.Dense(1, activation="linear",
                                      kernel_initializer="truncated_normal",
                                      kernel_regularizer=self.regularizer)

    @_tf.function
    def call(self, inputs, training=None, mask=None):
        """计算逻辑实现."""
        expanded = self.expanded(inputs)
        normalized = self.normalized(expanded, training=training)
        lstm = self.lstm(normalized)
        normalized2 = self.normalized_2(lstm, training=training)
        dropout = self.dropout(normalized2, training=training)
        output = self.outputs(dropout)
        return output

    def compile(self,
                optimizer=_tf.keras.optimizers.Adam(0.0001),
                loss="MSE",
                metrics=None,
                loss_weights=None,
                weighted_metrics=None,
                run_eagerly=None,
                **kwargs):
        """设置优化器、loss、metric等."""
        super().compile(optimizer=optimizer,
                        loss=loss,
                        metrics=metrics,
                        loss_weights=loss_weights,
                        weighted_metrics=weighted_metrics,
                        run_eagerly=run_eagerly)

    def get_config(self):
        """获取参数，保存模型需要的函数."""
        config = super().get_config().copy()
        config.update({'dropout': self.dropout,
                       'l2': self.l2,
                       'stride': self.stride})
        return config


class AlphaNetV3(_Model):
    """神经网络模型，继承``keras.Model``类.

    alpha net v3版本模型.

    Notes:
        复现华泰金工 alpha net V3 版本

        ``input: (batch_size, history time steps, features)``

    """

    def __init__(self,
                 dropout=0.0,
                 l2=0.001,
                 classification=False,
                 categories=0,
                 recurrent_unit="GRU",
                 *args,
                 **kwargs):
        """Alpha net v3.

        Notes:
            alpha net v3 版本的全tensorflow实现，结构详见代码展开

        Args:
            dropout: 跟在特征扩张以及Batch Normalization之后的dropout，默认无dropout
            l2: 输出层的l2-regularization参数
            classification: 是否为分类问题
            categories: 分类问题的类别数量
            recurrent_unit (str): 该参数可以为"GRU"或"LSTM"

        """
        super(AlphaNetV3, self).__init__(*args, **kwargs)
        self.l2 = l2
        self.dropout = dropout
        self.expanded10 = FeatureExpansion(stride=10)
        self.expanded5 = FeatureExpansion(stride=5)
        self.normalized10 = _tfl.BatchNormalization()
        self.normalized5 = _tfl.BatchNormalization()
        self.dropout_layer = _tfl.Dropout(self.dropout)
        if recurrent_unit == "GRU":
            self.recurrent10 = _tfl.GRU(units=30)
            self.recurrent5 = _tfl.GRU(units=30)
        elif recurrent_unit == "LSTM":
            self.recurrent10 = _tfl.LSTM(units=30)
            self.recurrent5 = _tfl.LSTM(units=30)
        else:
            raise ValueError("Unknown recurrent_unit")
        self.normalized10_2 = _tfl.BatchNormalization()
        self.normalized5_2 = _tfl.BatchNormalization()
        self.concat = _tfl.Concatenate(axis=-1)
        self.regularizer = _tf.keras.regularizers.l2(self.l2)
        if classification:
            if categories < 1:
                raise ValueError("categories should be at least 1")
            elif categories == 1:
                self.outputs = _tfl.Dense(1, activation="sigmoid",
                                          kernel_initializer="truncated_normal",
                                          kernel_regularizer=self.regularizer)
            else:
                self.outputs = _tfl.Dense(categories, activation="softmax",
                                          kernel_initializer="truncated_normal",
                                          kernel_regularizer=self.regularizer)
        else:
            self.outputs = _tfl.Dense(1, activation="linear",
                                      kernel_initializer="truncated_normal",
                                      kernel_regularizer=self.regularizer)

    @_tf.function
    def call(self, inputs, training=None, mask=None):
        """计算逻辑实现."""
        expanded10 = self.expanded10(inputs)
        expanded5 = self.expanded5(inputs)
        normalized10 = self.normalized10(expanded10, training=training)
        normalized5 = self.normalized5(expanded5, training=training)
        recurrent10 = self.recurrent10(normalized10)
        recurrent5 = self.recurrent5(normalized5)
        normalized10_2 = self.normalized10_2(recurrent10, training=training)
        normalized5_2 = self.normalized5_2(recurrent5, training=training)
        concat = self.concat([normalized10_2, normalized5_2])
        dropout = self.dropout_layer(concat, training=training)
        output = self.outputs(dropout)
        return output

    def compile(self,
                optimizer=_tf.keras.optimizers.Adam(0.0001),
                loss="MSE",
                metrics=None,
                loss_weights=None,
                weighted_metrics=None,
                run_eagerly=None,
                **kwargs):
        """设置优化器、loss、metric等."""
        super().compile(optimizer=optimizer,
                        loss=loss,
                        metrics=metrics,
                        loss_weights=loss_weights,
                        weighted_metrics=weighted_metrics,
                        run_eagerly=run_eagerly)

    def get_config(self):
        """获取参数，保存模型需要的函数."""
        config = super().get_config().copy()
        config.update({'dropout': self.dropout,
                       'l2': self.l2})
        return config


class AlphaNetV4(_Model):
    """神经网络模型，继承``keras.Model``类.

    Notes:
        ``input: (batch_size, history time steps, features)``

    """

    def __init__(self,
                 dropout=0.0,
                 l2=0.001,
                 classification=False,
                 categories=0,
                 recurrent_unit="GRU",
                 *args,
                 **kwargs):
        """Alpha net v4.

        Notes:
            去掉了batch normalization的模型，
            训练需要使用data模块的normalization
            或其他自定义normalization.

        Args:
            dropout: 跟在特征扩张以及Batch Normalization之后的dropout，默认无dropout
            l2: 输出层的l2-regularization参数
            classification: 是否为分类问题
            categories: 分类问题的类别数量
            recurrent_unit (str): 该参数可以为"GRU"或"LSTM"

        """
        super(AlphaNetV4, self).__init__(*args, **kwargs)
        self.l2 = l2
        self.dropout = dropout
        self.expanded10 = FeatureExpansion(stride=10)
        self.expanded5 = FeatureExpansion(stride=5)
        self.dropout_layer = _tfl.Dropout(self.dropout)
        if recurrent_unit == "GRU":
            self.recurrent10 = _tfl.GRU(units=30)
            self.recurrent5 = _tfl.GRU(units=30)
        elif recurrent_unit == "LSTM":
            self.recurrent10 = _tfl.LSTM(units=30)
            self.recurrent5 = _tfl.LSTM(units=30)
        else:
            raise ValueError("Unknown recurrent_unit")
        self.concat = _tfl.Concatenate(axis=-1)
        self.regularizer = _tf.keras.regularizers.l2(self.l2)
        self.dense = _tfl.Dense(units=30,
                                activation="relu",
                                kernel_initializer="he_normal",
                                kernel_regularizer=self.regularizer)
        if classification:
            if categories < 1:
                raise ValueError("categories should be at least 1")
            elif categories == 1:
                self.outputs = _tfl.Dense(1, activation="sigmoid",
                                          kernel_initializer="truncated_normal",
                                          kernel_regularizer=self.regularizer)
            else:
                self.outputs = _tfl.Dense(categories, activation="softmax",
                                          kernel_initializer="truncated_normal",
                                          kernel_regularizer=self.regularizer)
        else:
            self.outputs = _tfl.Dense(1, activation="linear",
                                      kernel_initializer="truncated_normal",
                                      kernel_regularizer=self.regularizer)

    @_tf.function
    def call(self, inputs, training=None, mask=None):
        """计算逻辑实现."""
        expanded10 = self.expanded10(inputs)
        expanded5 = self.expanded5(inputs)
        recurrent10 = self.recurrent10(expanded10)
        recurrent5 = self.recurrent5(expanded5)
        concat = self.concat([recurrent10, recurrent5])
        dropout = self.dropout_layer(concat, training=training)
        dense = self.dense(dropout)
        output = self.outputs(dense)
        return output

    def compile(self,
                optimizer=_tf.keras.optimizers.Adam(0.0001),
                loss="MSE",
                metrics=None,
                loss_weights=None,
                weighted_metrics=None,
                run_eagerly=None,
                **kwargs):
        """设置优化器、loss、metric等."""
        super().compile(optimizer=optimizer,
                        loss=loss,
                        metrics=metrics,
                        loss_weights=loss_weights,
                        weighted_metrics=weighted_metrics,
                        run_eagerly=run_eagerly)

    def get_config(self):
        """获取参数，保存模型需要的函数."""
        config = super().get_config().copy()
        config.update({'dropout': self.dropout,
                       'l2': self.l2})
        return config


def load_model(filepath,
               custom_objects: dict = None,
               compile: bool = True,
               options=None):
    """用于读取已存储的模型，可识别自定义metric: UpDownAccuracy.

    Notes:
        包装``tf.keras``的``load_model``函数，添加``UpDownAccuracy``.

    Args:
        filepath: 文件路径:
            - String or `pathlib.Path` object, path to the saved model
            - `h5py.File` object from which to load the model
        custom_objects: 自定义类的识别，从类或函数名到类或函数的映射字典.
        compile: Boolean, 是否compile model.
        options: 其他 `tf.saved_model.LoadOptions`.

    Returns:
        Keras model instance.

    Raises:
        ImportError: if loading from an hdf5 file and h5py is not available.
        IOError: In case of an invalid savefile
    """
    object_dict = {"UpDownAccuracy": _UpDownAccuracy}
    if custom_objects is not None:
        object_dict.update(custom_objects)
    return _tf.keras.models.load_model(filepath,
                                       custom_objects=object_dict,
                                       compile=compile,
                                       options=options)


class _LowerNoDiagonalMask(_Initializer):
    """获取不含对角元素的矩阵下三角mask.

    Notes:
        Provide a mask giving the lower triangular of a matrix
        without diagonal elements.

    """

    def __init__(self):
        super(_LowerNoDiagonalMask, self).__init__()

    def __call__(self, shape, **kwargs):
        """计算逻辑."""
        ones = _tf.ones(shape)
        mask_lower = _tf.linalg.band_part(ones, -1, 0)
        mask_diag = _tf.linalg.band_part(ones, 0, 0)
        # lower triangle removing the diagonal elements
        mask = _tf.cast(mask_lower - mask_diag, dtype=_tf.bool)
        return mask


def __get_dimensions__(input_shape, stride):
    """计算相关维度长度.

    Notes:
        output_length = 原来的时间长度 / stride的长度

    Args:
        input_shape: pass the inputs of layer to the function
        stride (int): the stride of the custom layer

    Returns:
        (features, output_length)

    Raises:
        ValueError: 如果历史长度不是stride的整数倍

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
