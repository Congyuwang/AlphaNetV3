import tensorflow as tf


class UpDownAccuracy(tf.keras.metrics.Metric):
    """
    通过对return的预测来计算涨跌准确率
    """

    def __init__(self, name='up_down_accuracy', **kwargs):
        super(UpDownAccuracy, self).__init__(name=name, **kwargs)
        self.up_down_correct_count = self.add_weight(name='ud_count',
                                                     initializer='zeros',
                                                     shape=(),
                                                     dtype=tf.float64)
        self.length = self.add_weight(name='len',
                                      initializer='zeros',
                                      shape=(),
                                      dtype=tf.float64)

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
