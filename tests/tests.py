import unittest
import tensorflow as tf
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import shutil
from src.alphanet import *
from src.alphanet.data import *
from src.alphanet.metrics import *


class TestLayers(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data = np.maximum(np.abs(np.random.rand(3, 30, 15)), 0.1)

    def test_std(self):
        s = Std()(tf.constant(self.data, tf.float32))
        test_result = []
        for j in range(len(self.data)):
            test1 = __is_all_close__(s[j][0],
                                     np.std(self.data[j][0:10], axis=0))
            test2 = __is_all_close__(s[j][1],
                                     np.std(self.data[j][10:20], axis=0))
            test3 = __is_all_close__(s[j][2],
                                     np.std(self.data[j][20:30], axis=0))
            test_result.extend([test1, test2, test3])
        self.assertTrue(np.all(test_result), "std incorrect")

    def test_zscore(self):
        z = ZScore()(tf.constant(self.data, tf.float32))
        test_result = []
        for j in range(len(self.data)):
            test1 = __is_all_close__(z[j][0],
                                     np.mean(self.data[j][0:10], axis=0) /
                                     np.std(self.data[j][0:10], axis=0))
            test2 = __is_all_close__(z[j][1],
                                     np.mean(self.data[j][10:20], axis=0) /
                                     np.std(self.data[j][10:20], axis=0))
            test3 = __is_all_close__(z[j][2],
                                     np.mean(self.data[j][20:30], axis=0) /
                                     np.std(self.data[j][20:30], axis=0))
            test_result.extend([test1, test2, test3])
        self.assertTrue(np.all(test_result), "z-score incorrect")

    def test_return(self):
        r = Return()(tf.constant(self.data, tf.float32))
        test_result = []
        for j in range(len(self.data)):
            test1 = __is_all_close__(r[j][0],
                                     (self.data[j][10 - 1] /
                                      self.data[j][0]) - 1, atol=1e-5)
            test2 = __is_all_close__(r[j][1],
                                     (self.data[j][20 - 1] /
                                      self.data[j][10]) - 1, atol=1e-5)
            test3 = __is_all_close__(r[j][2],
                                     (self.data[j][30 - 1] /
                                      self.data[j][20]) - 1, atol=1e-5)
            test_result.extend([test1, test2, test3])
        self.assertTrue(np.all(test_result), "return incorrect")

    def test_linear_decay(self):
        d = LinearDecay()(tf.constant(self.data, tf.float32))
        weights = np.linspace(1, 10, 10)
        test_result = []
        for j in range(len(self.data)):
            test1 = __is_all_close__(d[j][0], np.average(self.data[j][0:10],
                                                         axis=0,
                                                         weights=weights))
            test2 = __is_all_close__(d[j][1], np.average(self.data[j][10:20],
                                                         axis=0,
                                                         weights=weights))
            test3 = __is_all_close__(d[j][2], np.average(self.data[j][20:30],
                                                         axis=0,
                                                         weights=weights))
            test_result.extend([test1, test2, test3])
        self.assertTrue(np.all(test_result), "linear decay incorrect")

    def test_covariance(self):
        c = Covariance()(tf.constant(self.data, tf.float32))
        test_result = []
        for j in range(len(self.data)):
            test1 = __is_all_close__(c[j][0],
                                     self.__covariance__(self.data[j][0:10]))
            test2 = __is_all_close__(c[j][1],
                                     self.__covariance__(self.data[j][10:20]))
            test3 = __is_all_close__(c[j][2],
                                     self.__covariance__(self.data[j][20:30]))
            test_result.extend([test1, test2, test3])
        self.assertTrue(np.all(test_result), "covariance incorrect")

    def test_correlation(self):
        c = Correlation()(tf.constant(self.data, tf.float32))
        test_result = []
        for j in range(len(self.data)):
            test1 = __is_all_close__(c[j][0],
                                     self.__correlation__(self.data[j][0:10]),
                                     atol=1e-5)
            test2 = __is_all_close__(c[j][1],
                                     self.__correlation__(self.data[j][10:20]),
                                     atol=1e-5)
            test3 = __is_all_close__(c[j][2],
                                     self.__correlation__(self.data[j][20:30]),
                                     atol=1e-5)
            test_result.extend([test1, test2, test3])
        self.assertTrue(np.all(test_result), "correlation incorrect")

    @classmethod
    def __covariance__(cls, data):
        covariances = []
        cov_mat = np.cov(data.T)
        for i in range(cov_mat.shape[0]):
            for m in range(i):
                covariances.append(cov_mat[i, m])
        return covariances

    @classmethod
    def __correlation__(cls, data):
        correlations = []
        corr_coefficient = np.corrcoef(data.T)
        for i in range(corr_coefficient.shape[0]):
            for m in range(i):
                correlations.append(corr_coefficient[i, m])
        return correlations


class TestAlphaNetV2(unittest.TestCase):
    test_dir = "./.test_alpha_net_save/"

    @classmethod
    def setUpClass(cls):
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)
        os.mkdir(cls.test_dir)
        cls.wrong_shape = tf.constant(np.random.rand(500, 45, 15),
                                      dtype=tf.float32)
        cls.random_test = tf.constant(np.random.rand(500, 30, 15),
                                      dtype=tf.float32)
        cls.random_label = tf.constant(np.random.rand(500, ),
                                       dtype=tf.float32)

    def test_save_weights(self):

        # save weights
        alpha_net = AlphaNetV2()
        alpha_net.compile(optimizer=tf.keras.optimizers.Adam(0.0001),
                          loss="MSE",
                          metrics=[tf.keras.metrics.RootMeanSquaredError(),
                                   UpDownAccuracy()])
        alpha_net.fit(self.random_test, self.random_label, batch_size=20)
        output = alpha_net.predict(self.random_test)
        alpha_net.save_weights("./.test_alpha_net_save/weights")

        # load weights
        alpha_net = AlphaNetV2()
        alpha_net.compile(metrics=[tf.keras.metrics.RootMeanSquaredError(),
                                   UpDownAccuracy()])
        alpha_net.load_weights("./.test_alpha_net_save/weights")
        output_2 = alpha_net.predict(self.random_test)

        self.assertTrue(__is_all_close__(output, output_2),
                        "save and load weights failed")

    def test_save_model(self):

        # save models
        alpha_net = AlphaNetV2()
        alpha_net.compile(metrics=[tf.keras.metrics.RootMeanSquaredError(),
                                   UpDownAccuracy()])
        alpha_net.fit(self.random_test, self.random_label, batch_size=20)
        output = alpha_net.predict(self.random_test)
        alpha_net.save("./.test_alpha_net_save/model")

        # load models
        model = load_model("./.test_alpha_net_save/model")
        output_2 = model.predict(self.random_test, batch_size=500)
        self.assertTrue(__is_all_close__(output, output_2, atol=1e-5),
                        "save and load model failed")

    def test_wrong_shape(self):
        with self.assertRaises(ValueError) as context:
            alpha_net = AlphaNetV2()
            alpha_net.compile()
            alpha_net.fit(self.wrong_shape,
                          self.random_label,
                          batch_size=20)
        self.assertTrue("stride的整数倍" in str(context.exception))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.test_dir, ignore_errors=True)


class TestAlphaNetV3(unittest.TestCase):
    test_dir = "./.test_alpha_net_save/"

    @classmethod
    def setUpClass(cls):
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)
        os.mkdir(cls.test_dir)
        cls.wrong_shape = tf.constant(np.random.rand(500, 45, 15),
                                      dtype=tf.float32)
        cls.random_test = tf.constant(np.random.rand(500, 30, 15),
                                      dtype=tf.float32)
        cls.random_label = tf.constant(np.random.rand(500, ),
                                       dtype=tf.float32)

    def test_save_weights(self):

        # save weights
        alpha_net_v3 = AlphaNetV3()
        alpha_net_v3.compile(optimizer=tf.keras.optimizers.Adam(0.0001),
                             loss="MSE",
                             metrics=[tf.keras.metrics.RootMeanSquaredError(),
                                      UpDownAccuracy()])
        alpha_net_v3.fit(self.random_test, self.random_label, batch_size=20)
        output = alpha_net_v3.predict(self.random_test)
        alpha_net_v3.save_weights("./.test_alpha_net_save/weights")

        # load weights
        alpha_net_v3 = AlphaNetV3()
        alpha_net_v3.compile(metrics=[tf.keras.metrics.RootMeanSquaredError(),
                                      UpDownAccuracy()])
        alpha_net_v3.load_weights("./.test_alpha_net_save/weights")
        output_2 = alpha_net_v3.predict(self.random_test)

        self.assertTrue(__is_all_close__(output, output_2),
                        "save and load weights failed")

    def test_save_model(self):

        # save models
        alpha_net_v3 = AlphaNetV3()
        alpha_net_v3.compile(metrics=[tf.keras.metrics.RootMeanSquaredError(),
                                      UpDownAccuracy()])
        alpha_net_v3.fit(self.random_test, self.random_label, batch_size=20)
        output = alpha_net_v3.predict(self.random_test)
        alpha_net_v3.save("./.test_alpha_net_save/model")

        # load models
        model = load_model("./.test_alpha_net_save/model")
        output_2 = model.predict(self.random_test, batch_size=500)
        self.assertTrue(__is_all_close__(output, output_2, atol=1e-5),
                        "save and load model failed")

    def test_wrong_shape(self):
        with self.assertRaises(ValueError) as context:
            alpha_net_v3 = AlphaNetV3()
            alpha_net_v3.compile()
            alpha_net_v3.fit(self.wrong_shape,
                             self.random_label,
                             batch_size=20)
        self.assertTrue("stride的整数倍" in str(context.exception))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.test_dir, ignore_errors=True)


class TestAlphaNetV4(unittest.TestCase):
    test_dir = "./.test_alpha_net_save/"

    @classmethod
    def setUpClass(cls):
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)
        os.mkdir(cls.test_dir)
        cls.wrong_shape = tf.constant(np.random.rand(500, 45, 15),
                                      dtype=tf.float32)
        cls.random_test = tf.constant(np.random.rand(500, 30, 15),
                                      dtype=tf.float32)
        cls.random_label = tf.constant(np.random.rand(500, ),
                                       dtype=tf.float32)

    def test_save_weights(self):

        # save weights
        alpha_net_v4 = AlphaNetV4()
        alpha_net_v4.compile(optimizer=tf.keras.optimizers.Adam(0.0001),
                             loss="MSE",
                             metrics=[tf.keras.metrics.RootMeanSquaredError(),
                                      UpDownAccuracy()])
        alpha_net_v4.fit(self.random_test, self.random_label, batch_size=20)
        output = alpha_net_v4.predict(self.random_test)
        alpha_net_v4.save_weights("./.test_alpha_net_save/weights")

        # load weights
        alpha_net_v4 = AlphaNetV4()
        alpha_net_v4.compile(metrics=[tf.keras.metrics.RootMeanSquaredError(),
                                      UpDownAccuracy()])
        alpha_net_v4.load_weights("./.test_alpha_net_save/weights")
        output_2 = alpha_net_v4.predict(self.random_test)

        self.assertTrue(__is_all_close__(output, output_2),
                        "save and load weights failed")

    def test_save_model(self):

        # save models
        alpha_net_v4 = AlphaNetV4()
        alpha_net_v4.compile(metrics=[tf.keras.metrics.RootMeanSquaredError(),
                                      UpDownAccuracy()])
        alpha_net_v4.fit(self.random_test, self.random_label, batch_size=20)
        output = alpha_net_v4.predict(self.random_test)
        alpha_net_v4.save("./.test_alpha_net_save/model")

        # load models
        model = load_model("./.test_alpha_net_save/model")
        output_2 = model.predict(self.random_test, batch_size=500)
        self.assertTrue(__is_all_close__(output, output_2, atol=1e-5),
                        "save and load model failed")

    def test_wrong_shape(self):
        with self.assertRaises(ValueError) as context:
            alpha_net_v4 = AlphaNetV4()
            alpha_net_v4.compile()
            alpha_net_v4.fit(self.wrong_shape,
                             self.random_label,
                             batch_size=20)
        self.assertTrue("stride的整数倍" in str(context.exception))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.test_dir, ignore_errors=True)


class TestDataModule(unittest.TestCase):
    """
    Test the data ranges of data and label for `alphanet.data`
    """

    @classmethod
    def setUpClass(cls):
        (cls.data,
         cls.full_data,
         cls.codes,
         cls.trading_dates) = __test_data__()
        cls.test_date = np.random.randint(20110101, 20121231)
        print("getting batches for {}".format(cls.test_date))
        (cls.first_batch_train,
         cls.first_batch_val,
         cls.last_batch_train,
         cls.last_batch_val,
         cls.dates_info) = cls.__get_batches__(cls.data, cls.test_date)
        cls.start_basis = np.min(np.where(cls.trading_dates >= cls.test_date))

    def test_dates_info(self):
        self.assertEqual(
            min(self.dates_info["validation"]["dates_list"]),
            self.dates_info["validation"]["start_date"],
            "validation dates_list incorrect"
        )
        self.assertEqual(
            max(self.dates_info["validation"]["dates_list"]),
            self.dates_info["validation"]["end_date"],
            "validation dates_list incorrect"
        )

    def test_first_batch_of_training_dataset(self):
        data_label = self.__get_first_batches__(self.start_basis, 0, 120)
        for k, name in enumerate(["data", "label"]):
            self.assertTrue(__is_all_close__(
                data_label[k][:len(self.first_batch_train[k])],
                self.first_batch_train[k]
            ), "first batch of training {} "
               "(start {}): failure".format(name, self.test_date))

    def test_last_batch_of_training_dataset(self):
        data_label = self.__get_last_batches__(self.start_basis, 1200, 120)
        for k, name in enumerate(["data", "label"]):
            self.assertTrue(__is_all_close__(
                data_label[k][-len(self.last_batch_train[0]):],
                self.last_batch_train[k]
            ), "last batch of training {} "
               "(start {}): failure".format(name, self.test_date))

    def test_first_batch_of_validation_dataset(self):
        data_label = self.__get_first_batches__(self.start_basis, 1210 - 29, 120)
        for k, name in enumerate(["data", "label"]):
            self.assertTrue(__is_all_close__(
                data_label[k][:len(self.first_batch_val[k])],
                self.first_batch_val[k]
            ), "first batch of validation {} "
               "(start {}): failure".format(name, self.test_date))

    def test_last_batch_of_validation_dataset(self):
        data_label = self.__get_last_batches__(self.start_basis, 1510 - 1, 120)
        for k, name in enumerate(["data", "label"]):
            self.assertTrue(__is_all_close__(
                data_label[k][-len(self.last_batch_val[0]):],
                self.last_batch_val[k]
            ), "last batch of validation {} "
               "(start {}): failure".format(name, self.test_date))

    @classmethod
    def __get_batches__(cls, data, start_date):
        train_val_generator = TrainValData(data)
        train, val, dates_info = train_val_generator.get(start_date)
        first_train = next(iter(train.batch(500)))
        first_val = next(iter(val.batch(500)))
        last_train = None
        last_val = None

        for b in iter(train.batch(500)):
            last_train = b

        for b in iter(val.batch(500)):
            last_val = b

        return first_train, first_val, last_train, last_val, dates_info

    def __get_n_batches__(self,
                          start_date_index,
                          end_date_index,
                          n=2,
                          step=2):
        data_list = []
        label_list = []
        running_index = [(start_date_index + day, end_date_index + day, co)
                         for day in range(0, step * n, step)
                         for co in self.codes]
        for start, end, co in tqdm(running_index):
            start_date = self.trading_dates[start]
            end_date = self.trading_dates[end]
            df = self.full_data.loc[np.logical_and(
                np.logical_and(
                    self.full_data["代码"] == co,
                    self.full_data["日期"] <= end_date
                ),
                self.full_data["日期"] >= start_date
            ), :]
            dt = df.iloc[:, 3:].values
            lb = df["10日回报率"].iloc[-1]
            if np.sum(pd.isnull(dt)) == 0:
                data_list.append(dt)
                label_list.append(lb)

        return data_list, label_list

    def __get_first_batches__(self, start_basis, start, n, history=30, step=2):
        return self.__get_n_batches__(start_basis + start,
                                      start_basis + start + history - 1,
                                      n=n,
                                      step=step)

    def __get_last_batches__(self, start_basis, end, n, history=30, step=2):
        """
        :param end: exclusive
        """
        return self.__get_n_batches__(start_basis + end - history - step * (n - 1),
                                      start_basis + end - 1 - step * (n - 1),
                                      n=n,
                                      step=step)


class TestDataModuleClassification(unittest.TestCase):
    """
    Test the data ranges of data and label for `alphanet.data` for classification task
    """

    @classmethod
    def setUpClass(cls):
        (cls.data,
         cls.full_data,
         cls.codes,
         cls.trading_dates) = __test_data_classification__()
        cls.test_date = np.random.randint(20110101, 20121231)
        print("getting batches for {} (classification)".format(cls.test_date))
        (cls.first_batch_train,
         cls.first_batch_val,
         cls.last_batch_train,
         cls.last_batch_val,
         cls.dates_info) = cls.__get_batches__(cls.data, cls.test_date)
        cls.start_basis = np.min(np.where(cls.trading_dates >= cls.test_date))

    def test_dates_info(self):
        self.assertEqual(
            min(self.dates_info["validation"]["dates_list"]),
            self.dates_info["validation"]["start_date"],
            "validation dates_list incorrect"
        )
        self.assertEqual(
            max(self.dates_info["validation"]["dates_list"]),
            self.dates_info["validation"]["end_date"],
            "validation dates_list incorrect"
        )

    def test_first_batch_of_training_dataset(self):
        data_label = self.__get_first_batches__(self.start_basis, 0, 120)
        for k, name in enumerate(["data", "label"]):
            self.assertTrue(__is_all_close__(
                data_label[k][:len(self.first_batch_train[k])],
                self.first_batch_train[k]
            ), "first batch of training {} "
               "(start {}): failure".format(name, self.test_date))

    def test_last_batch_of_training_dataset(self):
        data_label = self.__get_last_batches__(self.start_basis, 1200, 120)
        for k, name in enumerate(["data", "label"]):
            self.assertTrue(__is_all_close__(
                data_label[k][-len(self.last_batch_train[0]):],
                self.last_batch_train[k]
            ), "last batch of training {} "
               "(start {}): failure".format(name, self.test_date))

    def test_first_batch_of_validation_dataset(self):
        data_label = self.__get_first_batches__(self.start_basis, 1210 - 29, 120)
        for k, name in enumerate(["data", "label"]):
            self.assertTrue(__is_all_close__(
                data_label[k][:len(self.first_batch_val[k])],
                self.first_batch_val[k]
            ), "first batch of validation {} "
               "(start {}): failure".format(name, self.test_date))

    def test_last_batch_of_validation_dataset(self):
        data_label = self.__get_last_batches__(self.start_basis, 1510 - 1, 120)
        for k, name in enumerate(["data", "label"]):
            self.assertTrue(__is_all_close__(
                data_label[k][-len(self.last_batch_val[0]):],
                self.last_batch_val[k]
            ), "last batch of validation {} "
               "(start {}): failure".format(name, self.test_date))

    @classmethod
    def __get_batches__(cls, data, start_date):
        train_val_generator = TrainValData(data)
        train, val, dates_info = train_val_generator.get(start_date)
        first_train = next(iter(train.batch(500)))
        first_val = next(iter(val.batch(500)))
        last_train = None
        last_val = None

        for b in iter(train.batch(500)):
            last_train = b

        for b in iter(val.batch(500)):
            last_val = b

        return first_train, first_val, last_train, last_val, dates_info

    def __get_n_batches__(self,
                          start_date_index,
                          end_date_index,
                          n=2,
                          step=2,
                          class_count=3):
        data_list = []
        label_list = []
        running_index = [(start_date_index + day, end_date_index + day, co)
                         for day in range(0, step * n, step)
                         for co in self.codes]
        for start, end, co in tqdm(running_index):
            start_date = self.trading_dates[start]
            end_date = self.trading_dates[end]
            df_label = pd.get_dummies(self.full_data["10日回报率"])
            which_ones = np.logical_and(
                np.logical_and(
                    self.full_data["代码"] == co,
                    self.full_data["日期"] <= end_date
                ),
                self.full_data["日期"] >= start_date
            )
            df = self.full_data.loc[which_ones, :]
            labels = df_label.loc[which_ones, :]
            dt = df.iloc[:, 3:].values
            lb = labels.iloc[-1].values
            if np.sum(pd.isnull(dt)) == 0:
                data_list.append(dt)
                label_list.append(lb)

        return data_list, label_list

    def __get_first_batches__(self, start_basis, start, n, history=30, step=2):
        return self.__get_n_batches__(start_basis + start,
                                      start_basis + start + history - 1,
                                      n=n,
                                      step=step)

    def __get_last_batches__(self, start_basis, end, n, history=30, step=2):
        """
        :param end: exclusive
        """
        return self.__get_n_batches__(start_basis + end - history - step * (n - 1),
                                      start_basis + end - 1 - step * (n - 1),
                                      n=n,
                                      step=step)


class TestMetrics(unittest.TestCase):

    def test_up_down_accuracy(self):
        upd = UpDownAccuracy()
        upd.update_state(tf.constant([2.3, -1.3, 2.0]),
                         tf.constant([2.0, -1.2, -0.3]))
        self.assertTrue(np.isclose(upd.result(), 2 / 3), "accuracy incorrect")
        upd.update_state(tf.constant([2.3, -1.3]),
                         tf.constant([2.0, -1.2]))
        self.assertTrue(np.isclose(upd.result(), 4 / 5), "accuracy incorrect")
        upd.reset_states()
        self.assertTrue(np.isclose(upd.result(), 0.0), "reset failure")


def __test_data__():
    # 测试数据准备
    path_1 = "./tests/test_data/test_data.zip"
    path_2 = "./test_data/test_data.zip"
    if os.path.exists(path_1):
        df = pd.read_csv(path_1, dtype={"代码": "category"})
    elif os.path.exists(path_2):
        df = pd.read_csv(path_2, dtype={"代码": "category"})
    else:
        raise FileNotFoundError("test data missing")
    codes = df.代码.cat.categories
    df_parts = [df.loc[df.代码 == code, :] for code in codes]
    stock_data = [TimeSeriesData(dates=p["日期"].values,
                                 data=p.iloc[:, 3:].values,
                                 labels=p["10日回报率"].values)
                  for p in df_parts]
    # 补全全部stock与日期组合，用于手动生成batch对比测试
    trading_dates = df["日期"].unique()
    trading_dates.sort()
    full_index = pd.DataFrame([[s, d] for s in codes for d in trading_dates])
    full_index.columns = ["代码", "日期"]
    full_csi = full_index.merge(df,
                                how="left",
                                left_on=["代码", "日期"],
                                right_on=["代码", "日期"])
    return stock_data, full_csi, codes, trading_dates


def __test_data_classification__(class_count=3):
    # 测试数据准备
    path_1 = "./tests/test_data/test_data.zip"
    path_2 = "./test_data/test_data.zip"
    if os.path.exists(path_1):
        df = pd.read_csv(path_1, dtype={"代码": "category"})
    elif os.path.exists(path_2):
        df = pd.read_csv(path_2, dtype={"代码": "category"})
    else:
        raise FileNotFoundError("test data missing")
    codes = df.代码.cat.categories
    df["10日回报率"] = df.groupby("日期")["10日回报率"].transform(lambda x: pd.qcut(x, class_count, labels=False))
    csi_label = df["10日回报率"]
    csi_label = pd.get_dummies(csi_label)
    df_parts = [(csi_label.loc[df.代码 == code, :], df.loc[df.代码 == code, :]) for code in codes]
    stock_data = [TimeSeriesData(dates=p["日期"].values,
                                 data=p.iloc[:, 3:].values,
                                 labels=l.values)
                  for l, p in df_parts]
    # 补全全部stock与日期组合，用于手动生成batch对比测试
    trading_dates = df["日期"].unique()
    trading_dates.sort()
    full_index = pd.DataFrame([[s, d] for s in codes for d in trading_dates])
    full_index.columns = ["代码", "日期"]
    full_csi = full_index.merge(df,
                                how="left",
                                left_on=["代码", "日期"],
                                right_on=["代码", "日期"])
    return stock_data, full_csi, codes, trading_dates


def __is_all_close__(data1, data2, **kwargs):
    return np.all(np.isclose(data1, data2, **kwargs))


# unit test
if __name__ == "__main__":
    unittest.main(verbosity=2)
