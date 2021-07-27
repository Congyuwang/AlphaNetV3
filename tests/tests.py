import unittest
import tensorflow as tf
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import shutil
from alphanet import *
from alphanet.data import *
from alphanet.metrics import *
from preprocessing import process_df, build_data_list


def prepare_test_data():
    # 测试数据准备
    csi = process_df("../data/CSI500.zip")

    codes = csi.代码.cat.categories
    stock_data = build_data_list(csi)

    test_data = tf.constant([stock_data[0].data[0:30],
                             stock_data[0].data[2:32],
                             stock_data[0].data[4:34]], dtype=tf.float32)

    # 补全全部stock与日期组合，用于手动生成batch对比测试
    trading_dates = csi["日期"].unique()
    trading_dates.sort()
    full_index = pd.DataFrame([[s, d] for s in codes for d in trading_dates])
    full_index.columns = ["代码", "日期"]
    full_csi = full_index.merge(csi,
                                how="left",
                                left_on=["代码", "日期"],
                                right_on=["代码", "日期"])
    return test_data, stock_data, full_csi, codes, trading_dates


class TestLayers(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data, _, _, _, _ = prepare_test_data()

    def test_std(self):
        s = Std()(self.data)
        test_result = []
        for j in range(len(self.data)):
            test1 = __is_all_close__(s[j][0], np.std(self.data[j][0:10], axis=0))
            test2 = __is_all_close__(s[j][1], np.std(self.data[j][10:20], axis=0))
            test3 = __is_all_close__(s[j][2], np.std(self.data[j][20:30], axis=0))
            test_result.extend([test1, test2, test3])
        self.assertTrue(np.all(test_result), "std incorrect")

    def test_zscore(self):
        z = ZScore()(self.data)
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
        r = Return()(self.data)
        test_result = []
        for j in range(len(self.data)):
            test1 = __is_all_close__(r[j][0], self.data[j][10 - 1] /
                                     self.data[j][0] - 1)
            test2 = __is_all_close__(r[j][1], self.data[j][20 - 1] /
                                     self.data[j][10] - 1)
            test3 = __is_all_close__(r[j][2], self.data[j][30 - 1] /
                                     self.data[j][20] - 1)
            test_result.extend([test1, test2, test3])
        self.assertTrue(np.all(test_result), "return incorrect")

    def test_linear_decay(self):
        d = LinearDecay()(self.data)
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
        c = Covariance()(self.data)
        test_result = []
        for j in range(len(self.data)):
            test1 = __is_all_close__(c[j][0],
                                     __test_covariance__(self.data[j][0:10]))
            test2 = __is_all_close__(c[j][1],
                                     __test_covariance__(self.data[j][10:20]))
            test3 = __is_all_close__(c[j][2],
                                     __test_covariance__(self.data[j][20:30]))
            test_result.extend([test1, test2, test3])
        self.assertTrue(np.all(test_result), "covariance incorrect")

    def test_correlation(self):
        c = Correlation()(self.data)
        test_result = []
        for j in range(len(self.data)):
            test1 = __is_all_close__(c[j][0],
                                     __test_correlation__(self.data[j][0:10]),
                                     atol=1e-5)
            test2 = __is_all_close__(c[j][1],
                                     __test_correlation__(self.data[j][10:20]),
                                     atol=1e-5)
            test3 = __is_all_close__(c[j][2],
                                     __test_correlation__(self.data[j][20:30]),
                                     atol=1e-5)
            test_result.extend([test1, test2, test3])
        self.assertTrue(np.all(test_result), "correlation incorrect")


class TestAlphaNet(unittest.TestCase):

    def test_save_model(self):
        alpha_net_v3 = AlphaNetV3()
        test_dir = "./.test_alpha_net_save/"
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
        os.mkdir(test_dir)
        alpha_net_v3.save_weights("./.test_alpha_net_save/test")
        random_test = tf.constant(np.random.rand(500, 30, 15))
        output = alpha_net_v3(random_test)
        alpha_net_v3 = AlphaNetV3()
        alpha_net_v3.load_weights("./.test_alpha_net_save/test")
        output_2 = alpha_net_v3(random_test)
        self.assertTrue(__is_all_close__(output, output_2),
                        "save and load failed")
        shutil.rmtree(test_dir)


class TestDataModuleCrossChecking(unittest.TestCase):

    _, data, full_data, codes, trading_dates = prepare_test_data()
    producer_1 = TrainValData(data)
    producer_2 = TrainValData(data)
    train_1, val_1, _ = producer_1.get(20120301)
    train_2, val_2, _ = producer_2.get(20120301, mode="generator")

    def test_compare_train(self):
        for d1, d2 in zip(iter(self.train_1.batch(500)),
                          iter(self.train_2.batch(500))):
            self.assertTrue(__is_all_close__(d1[0], d2[0]))
            self.assertTrue(__is_all_close__(d1[1], d2[1]))

    def test_compare_val(self):
        for d in zip(iter(self.val_1.batch(500)),
                     iter(self.val_2.batch(500))):
            self.assertTrue(__is_all_close__(d[0][0], d[1][0]))
            self.assertTrue(__is_all_close__(d[0][1], d[1][1]))


class TestDataModule(unittest.TestCase):

    _, data, full_data, codes, trading_dates = prepare_test_data()

    @classmethod
    def setUpClass(cls):
        cls.test_date_1 = 20110131
        cls.test_date_2 = 20120201
        print("getting batches for {}".format(cls.test_date_1))
        (cls.first_batch_train_1,
         cls.first_batch_val_1,
         cls.last_batch_train_1,
         cls.last_batch_val_1) = cls.__get_batches__(cls.test_date_1)
        print("getting batches for {}".format(cls.test_date_2))
        (cls.first_batch_train_2,
         cls.first_batch_val_2,
         cls.last_batch_train_2,
         cls.last_batch_val_2) = cls.__get_batches__(cls.test_date_2)
        cls.start_basis_1 = np.min(np.where(cls.trading_dates == cls.test_date_1))
        cls.start_basis_2 = np.min(np.where(cls.trading_dates == cls.test_date_2))

    @classmethod
    def __get_batches__(cls, start_date):
        train_val_generator = TrainValData(cls.data)
        train, val, _ = train_val_generator.get(start_date)
        first_train = next(iter(train.batch(500)))
        first_val = next(iter(val.batch(500)))
        last_train = None
        last_val = None

        for b in iter(train.batch(500)):
            last_train = b

        for b in iter(val.batch(500)):
            last_val = b

        return first_train, first_val, last_train, last_val

    @classmethod
    def __get_n_batches__(cls,
                          start_date_index,
                          end_date_index,
                          n=2,
                          step=2):
        data_list = []
        label_list = []
        running_index = [(start_date_index + day, end_date_index + day, co)
                         for day in range(0, step * n, step)
                         for co in cls.codes]
        for start, end, co in tqdm(running_index):
            start_date = cls.trading_dates[start]
            end_date = cls.trading_dates[end]
            df = cls.full_data.loc[np.logical_and(
                np.logical_and(
                    cls.full_data["代码"] == co,
                    cls.full_data["日期"] <= end_date
                ),
                cls.full_data["日期"] >= start_date
            ), :]
            dt = df.iloc[:, 3:].values
            lb = df["10日回报率"].iloc[-1]
            if np.sum(pd.isnull(dt)) == 0:
                data_list.append(dt)
                label_list.append(lb)

        return data_list, label_list

    def test_first_batch_of_training_dataset_1(self):
        first_data_queue = self.__get_n_batches__(self.start_basis_1 + 0,
                                                  self.start_basis_1 + 29, 3)
        data, label = first_data_queue
        self.assertTrue(__is_all_close__(
            data[:len(self.first_batch_train_1[0])],
            self.first_batch_train_1[0]
        ), "first batch of training dataset (start {}): failure".format(self.test_date_1))
        self.assertTrue(__is_all_close__(
            label[:len(self.first_batch_train_1[0])],
            self.first_batch_train_1[1]
        ), "first batch of training dataset (start {}): failure".format(self.test_date_1))

    def test_last_batch_of_training_dataset_1(self):
        last_data_queue = self.__get_n_batches__(self.start_basis_1 + 1170 - 2,
                                                 self.start_basis_1 + 1199 - 2)
        data, label = last_data_queue
        self.assertTrue(__is_all_close__(
            data[-len(self.last_batch_train_1[0]):],
            self.last_batch_train_1[0]
        ), "last batch of training dataset (start {}): failure".format(self.test_date_1))
        self.assertTrue(__is_all_close__(
            label[-len(self.last_batch_train_1[0]):],
            self.last_batch_train_1[1]
        ), "last batch of training dataset (start {}): failure".format(self.test_date_1))

    def test_first_batch_of_validation_dataset_1(self):
        first_val_data_queue = self.__get_n_batches__(self.start_basis_1 + 1200 - 29 + 10,
                                                      self.start_basis_1 + 1210)
        data, label = first_val_data_queue
        self.assertTrue(__is_all_close__(
            data[:len(self.first_batch_val_1[0])],
            self.first_batch_val_1[0]
        ), "first batch of validation dataset (start {}): failure".format(self.test_date_1))
        self.assertTrue(__is_all_close__(
            label[:len(self.first_batch_val_1[0])],
            self.first_batch_val_1[1]
        ), "first batch of validation dataset (start {}): failure".format(self.test_date_1))

    def test_last_batch_of_validation_dataset_1(self):
        last_val_data_queue = self.__get_n_batches__(self.start_basis_1 + 1470 + 7,
                                                     self.start_basis_1 + 1499 + 9 - 2)
        data, label = last_val_data_queue
        self.assertTrue(__is_all_close__(
            data[-len(self.last_batch_val_1[0]):],
            self.last_batch_val_1[0]
        ), "last batch of training dataset (start {}): failure".format(self.test_date_1))
        self.assertTrue(__is_all_close__(
            label[-len(self.last_batch_val_1[0]):],
            self.last_batch_val_1[1]
        ), "last batch of training dataset (start {}): failure".format(self.test_date_1))

    def test_first_batch_of_training_dataset_2(self):
        first_data_queue = self.__get_n_batches__(self.start_basis_2 + 0,
                                                  self.start_basis_2 + 29, 3)
        data, label = first_data_queue
        self.assertTrue(__is_all_close__(
            data[:len(self.first_batch_train_2[0])],
            self.first_batch_train_2[0]
        ), "first batch of training dataset (start {}): failure".format(self.test_date_2))
        self.assertTrue(__is_all_close__(
            label[:len(self.first_batch_train_2[0])],
            self.first_batch_train_2[1]
        ), "first batch of training dataset (start {}): failure".format(self.test_date_2))

    def test_last_batch_of_training_dataset_2(self):
        last_data_queue = self.__get_n_batches__(self.start_basis_2 + 1170 - 2,
                                                 self.start_basis_2 + 1199 - 2)
        data, label = last_data_queue
        self.assertTrue(__is_all_close__(
            data[-len(self.last_batch_train_2[0]):],
            self.last_batch_train_2[0]
        ), "last batch of training dataset (start {}): failure".format(self.test_date_2))
        self.assertTrue(__is_all_close__(
            label[-len(self.last_batch_train_2[0]):],
            self.last_batch_train_2[1]
        ), "last batch of training dataset (start {}): failure".format(self.test_date_2))

    def test_first_batch_of_validation_dataset_2(self):
        first_val_data_queue = self.__get_n_batches__(self.start_basis_2 + 1200 - 30 + 1 + 10,
                                                      self.start_basis_2 + 1210)
        data, label = first_val_data_queue
        self.assertTrue(__is_all_close__(
            data[:len(self.first_batch_val_2[0])],
            self.first_batch_val_2[0]
        ), "first batch of validation dataset (start {}): failure".format(self.test_date_2))
        self.assertTrue(__is_all_close__(
            label[:len(self.first_batch_val_2[0])],
            self.first_batch_val_2[1]
        ), "first batch of validation dataset (start {}): failure".format(self.test_date_2))

    def test_last_batch_of_validation_dataset_2(self):
        last_val_data_queue = self.__get_n_batches__(self.start_basis_2 + 1470 + 9 - 2,
                                                     self.start_basis_2 + 1499 + 9 - 2)
        data, label = last_val_data_queue
        self.assertTrue(__is_all_close__(
            data[-len(self.last_batch_val_2[0]):],
            self.last_batch_val_2[0]
        ), "last batch of training dataset (start {}): failure".format(self.test_date_2))
        self.assertTrue(__is_all_close__(
            label[-len(self.last_batch_val_2[0]):],
            self.last_batch_val_2[1]
        ), "last batch of training dataset (start {}): failure".format(self.test_date_2))


class TestMetrics(unittest.TestCase):

    def test_up_down_accuracy(self):
        upd = UpDownAccuracy()
        upd.update_state(tf.constant([2.3, -1.3, 2.0]),
                         tf.constant([2.0, -1.2, -0.3]))
        self.assertTrue(np.isclose(upd.result(), 2/3), "accuracy incorrect")
        upd.update_state(tf.constant([2.3, -1.3]),
                         tf.constant([2.0, -1.2]))
        self.assertTrue(np.isclose(upd.result(), 4/5), "accuracy incorrect")
        upd.reset_states()
        self.assertTrue(np.isclose(upd.result(), 0.0), "reset failure")


def __is_all_close__(data1, data2, **kwargs):
    return np.all(np.isclose(data1, data2, **kwargs))


def __test_covariance__(data):
    data = data.numpy()
    covariances = []
    cov_mat = np.cov(data.T)
    for i in range(cov_mat.shape[0]):
        for m in range(i):
            covariances.append(cov_mat[i, m])
    return covariances


def __test_correlation__(data):
    data = data.numpy()
    correlations = []
    corr_coefficient = np.corrcoef(data.T)
    for i in range(corr_coefficient.shape[0]):
        for m in range(i):
            correlations.append(corr_coefficient[i, m])
    return correlations


# unit test
if __name__ == "__main__":
    unittest.main(verbosity=2)
