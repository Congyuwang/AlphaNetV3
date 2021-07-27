import pandas as pd

from alphanet.data import TimeSeriesData


def process_df(data):
    """
    新增特征以及计算十日累计回报
    :param data: csv或压缩包路径
    :return: dataframe
    """
    csi = pd.read_csv(data, dtype={"代码": "category",
                                   "简称": "category"})
    csi.drop(columns=["简称"], inplace=True)

    # 新增特征
    csi["close/free_turn"] = csi["收盘价(元)"] / csi["换手率(基准.自由流通股本)"]
    csi["open/turn"] = csi["开盘价(元)"] / csi["换手率(%)"]
    csi["volume/low"] = csi["成交量(股)"] / csi["最低价(元)"]
    csi["vwap/high"] = csi["均价"] / csi["最高价(元)"]
    csi["low/high"] = csi["最低价(元)"] / csi["最高价(元)"]
    csi["vwap/close"] = csi["均价"] / csi["收盘价(元)"]

    # 计算t(1)至t(11)回报
    trading_dates = csi["日期"].unique()
    trading_dates.sort()
    dates_shift_dictionary_1 = dict(zip(trading_dates[1:], trading_dates[:-1]))
    dates_shift_dictionary_11 = dict(zip(trading_dates[11:], trading_dates[:-11]))
    csi_slice = csi[["代码", "日期", "收盘价(元)"]].copy()
    csi_slice_date_shift_1 = csi[["代码", "日期", "收盘价(元)"]].copy()
    csi_slice_date_shift_11 = csi[["代码", "日期", "收盘价(元)"]].copy()
    csi_slice_date_shift_1["日期"] = csi_slice_date_shift_11["日期"]\
        .map(lambda x: dates_shift_dictionary_1.get(x, None))
    csi_slice_date_shift_11["日期"] = csi_slice_date_shift_11["日期"]\
        .map(lambda x: dates_shift_dictionary_11.get(x, None))
    csi_slice_date_shift_11.rename(columns={"收盘价(元)": "11交易日后收盘价(元)"},
                                   inplace=True)
    csi_slice_date_shift_11.dropna(inplace=True)
    csi_slice_date_shift_1.rename(columns={"收盘价(元)": "1交易日后收盘价(元)"},
                                  inplace=True)
    csi_slice_date_shift_1.dropna(inplace=True)
    csi_slice = csi_slice.merge(csi_slice_date_shift_1,
                                how="inner",
                                left_on=["代码", "日期"],
                                right_on=["代码", "日期"])
    csi_slice = csi_slice.merge(csi_slice_date_shift_11,
                                how="inner",
                                left_on=["代码", "日期"],
                                right_on=["代码", "日期"])
    future_close_price_1 = csi_slice["1交易日后收盘价(元)"]
    future_close_price_11 = csi_slice["11交易日后收盘价(元)"]
    csi_slice["10日回报率"] = future_close_price_11 / future_close_price_1 - 1
    csi_slice.drop(columns=["收盘价(元)", "11交易日后收盘价(元)", "1交易日后收盘价(元)"], inplace=True)
    csi = csi_slice.merge(csi,
                          how="inner",
                          left_on=["代码", "日期"],
                          right_on=["代码", "日期"])
    return csi


def build_data_list(csi):
    """
    根据上面的dataframe结果构建TimeSeriesData list
    :param csi: process_df返回的结果
    :return:
    """
    codes = csi.代码.cat.categories
    stock_data_list = []
    for code in codes:
        table_part = csi.loc[csi.代码 == code, :]
        stock_data_list.append(TimeSeriesData(dates=table_part["日期"].values,
                                              data=table_part.iloc[:, 3:].values,
                                              labels=table_part["10日回报率"].values))
    return stock_data_list
