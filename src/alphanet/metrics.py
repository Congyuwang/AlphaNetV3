"""训练中的辅助准确率信息.

该module包含涨跌精度信息的计算类。
使用到该类的模型在保存后重新读取时需要添加``custom_objects``参数，
或者使用``alphanet.load_model()``函数。

"""
import tensorflow as _tf


__all__ = ["UpDownAccuracy"]


class UpDownAccuracy(_tf.keras.metrics.Metric):
    """通过对return的预测来计算涨跌准确率."""

    def __init__(self, name='up_down_accuracy', **kwargs):
        """涨跌准确率."""
        super(UpDownAccuracy, self).__init__(name=name, **kwargs)
        self.up_down_correct_count = self.add_weight(name='ud_count',
                                                     initializer='zeros',
                                                     shape=(),
                                                     dtype=_tf.float32)
        self.length = self.add_weight(name='len',
                                      initializer='zeros',
                                      shape=(),
                                      dtype=_tf.float32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        """加入新的预测更新精度值."""
        y_true = _tf.cast(y_true > 0.0, _tf.float32)
        y_pred = _tf.cast(y_pred > 0.0, _tf.float32)
        length = _tf.cast(len(y_true), _tf.float32)
        correct_count = length - _tf.reduce_sum(_tf.abs(y_true - y_pred))

        self.length.assign_add(length)
        self.up_down_correct_count.assign_add(correct_count)

    def result(self):
        """获取结果."""
        if self.length == 0.0:
            return 0.0
        return self.up_down_correct_count / self.length

    def reset_state(self):
        """在train、val的epoch末尾重置精度."""
        self.up_down_correct_count.assign(0.0)
        self.length.assign(0.0)
