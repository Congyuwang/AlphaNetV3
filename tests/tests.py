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


class TestLayers(unittest.TestCase):

    def setUp(self):
        self.data, _, _, _, _ = prepare_test_data()

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
        alpha_net_v3.save("./.test_alpha_net_save/test")
        random_test = tf.constant(np.random.rand(500, 30, 15))
        output = alpha_net_v3(random_test)
        alpha_net_v3 = AlphaNetV3()
        alpha_net_v3.load("./.test_alpha_net_save/test")
        output_2 = alpha_net_v3(random_test)
        self.assertTrue(__is_all_close__(output, output_2),
                        "save and load failed")
        shutil.rmtree(test_dir)


class TestDataModule(unittest.TestCase):

    def setUp(self):
        _, data, full_data, codes, trading_dates = prepare_test_data()
        self.data = data
        self.full_data = full_data
        self.codes = codes
        self.trading_dates = trading_dates
        self.test_date_1 = 20110131
        self.test_date_2 = 20120201
        print("getting batches for {}".format(self.test_date_1))
        (self.first_batch_train_1,
         self.first_batch_val_1,
         self.last_batch_train_1,
         self.last_batch_val_1) = self.__get_batches__(self.test_date_1)
        print("getting batches for {}".format(self.test_date_2))
        (self.first_batch_train_2,
         self.first_batch_val_2,
         self.last_batch_train_2,
         self.last_batch_val_2) = self.__get_batches__(self.test_date_2)
        self.start_basis_1 = np.min(np.where(trading_dates == self.test_date_1))
        self.start_basis_2 = np.min(np.where(trading_dates == self.test_date_2))

    def __get_batches__(self, start_date):
        train_val_generator = TrainValData(self.data)
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

    def __get_n_batches__(self,
                          start_date_index,
                          end_date_index,
                          n=2,
                          step=2):
        data_list = []
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
            ), :].iloc[:, 3:].values
            if np.sum(pd.isnull(df)) == 0:
                data_list.append(df)

        return data_list

    def test_first_batch_of_training_dataset_1(self):
        first_data_queue = self.__get_n_batches__(self.start_basis_1 + 0,
                                                  self.start_basis_1 + 29, 3)
        self.assertTrue(__is_all_close__(
            first_data_queue[:len(self.first_batch_train_1[0])],
            self.first_batch_train_1[0]
        ), "first batch of training dataset (start {}): failure".format(self.test_date_1))

    def test_last_batch_of_training_dataset_1(self):
        last_data_queue = self.__get_n_batches__(self.start_basis_1 + 1170 - 2,
                                                 self.start_basis_1 + 1199 - 2)
        self.assertTrue(__is_all_close__(
            last_data_queue[-len(self.last_batch_train_1[0]):],
            self.last_batch_train_1[0]
        ), "last batch of training dataset (start {}): failure".format(self.test_date_1))

    def test_first_batch_of_validation_dataset_1(self):
        first_val_data_queue = self.__get_n_batches__(self.start_basis_1 + 1200 - 30 + 2,
                                                      self.start_basis_1 + 1201)
        self.assertTrue(__is_all_close__(
            first_val_data_queue[:len(self.first_batch_val_1[0])],
            self.first_batch_val_1[0]
        ), "first batch of validation dataset (start {}): failure".format(self.test_date_1))

    def test_last_batch_of_validation_dataset_1(self):
        last_val_data_queue = self.__get_n_batches__(self.start_basis_1 + 1470 - 2,
                                                     self.start_basis_1 + 1499 - 2)
        self.assertTrue(__is_all_close__(
            last_val_data_queue[-len(self.last_batch_val_1[0]):],
            self.last_batch_val_1[0]
        ), "last batch of training dataset (start {}): failure".format(self.test_date_1))

    def test_first_batch_of_training_dataset_2(self):
        first_data_queue = self.__get_n_batches__(self.start_basis_2 + 0,
                                                  self.start_basis_2 + 29, 3)
        self.assertTrue(__is_all_close__(
            first_data_queue[:len(self.first_batch_train_2[0])],
            self.first_batch_train_2[0]
        ), "first batch of training dataset (start {}): failure".format(self.test_date_2))

    def test_last_batch_of_training_dataset_2(self):
        last_data_queue = self.__get_n_batches__(self.start_basis_2 + 1170 - 2,
                                                 self.start_basis_2 + 1199 - 2)
        self.assertTrue(__is_all_close__(
            last_data_queue[-len(self.last_batch_train_2[0]):],
            self.last_batch_train_2[0]
        ), "last batch of training dataset (start {}): failure".format(self.test_date_2))

    def test_first_batch_of_validation_dataset_2(self):
        first_val_data_queue = self.__get_n_batches__(self.start_basis_2 + 1200 - 30 + 2,
                                                      self.start_basis_2 + 1201)
        self.assertTrue(__is_all_close__(
            first_val_data_queue[:len(self.first_batch_val_2[0])],
            self.first_batch_val_2[0]
        ), "first batch of validation dataset (start {}): failure".format(self.test_date_2))

    def test_last_batch_of_validation_dataset_2(self):
        last_val_data_queue = self.__get_n_batches__(self.start_basis_2 + 1470 - 2,
                                                     self.start_basis_2 + 1499 - 2)
        self.assertTrue(__is_all_close__(
            last_val_data_queue[-len(self.last_batch_val_2[0]):],
            self.last_batch_val_2[0]
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


def prepare_test_data():
    # 测试数据准备
    csi = pd.read_csv("../data/CSI500.zip", dtype={"代码": "category",
                                                   "简称": "category"})
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
    csi_slice_date_shift["日期"] = csi_slice_date_shift["日期"] \
        .map(lambda x: dates_shift_dictionary.get(x, None))
    csi_slice_date_shift.rename(columns={"收盘价(元)": "10交易日后收盘价(元)"},
                                inplace=True)
    csi_slice_date_shift.dropna(inplace=True)
    csi_slice_date_shift["日期"] = [d for d in csi_slice_date_shift["日期"]]
    csi_slice = csi_slice.merge(csi_slice_date_shift,
                                how="inner",
                                left_on=["代码", "日期"],
                                right_on=["代码", "日期"])
    close_price = csi_slice["收盘价(元)"]
    future_close_price = csi_slice["10交易日后收盘价(元)"]
    csi_slice["10日回报率"] = future_close_price / close_price - 1
    csi_slice.drop(columns=["收盘价(元)", "10交易日后收盘价(元)"], inplace=True)
    csi = csi_slice.merge(csi,
                          how="inner",
                          left_on=["代码", "日期"],
                          right_on=["代码", "日期"])

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
    full_csi = full_index.merge(csi,
                                how="left",
                                left_on=["代码", "日期"],
                                right_on=["代码", "日期"])
    return test_data, stock_data, full_csi, codes, trading_dates


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
