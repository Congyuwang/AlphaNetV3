#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
import pandas as pd
import os
import json
import matplotlib.pyplot as plt
from alphanet import AlphaNetV3
from alphanet.data import TrainValData
from alphanet.metrics import UpDownAccuracy
from preprocessing import process_df, build_data_list
from config import *


def try_mkdirs(path):
    """新建路径"""
    if not os.path.exists(path):
        os.makedirs(path)


def print_info(dates_info):
    """显示信息"""
    print(f"Start_training for period "
          f"{dates_info['training']['start_date']} "
          f"to {dates_info['validation']['end_date']}:\n"
          f"\t the training set is "
          f"{dates_info['training']['start_date']}:"
          f"{dates_info['training']['end_date']} (inclusive)\n"
          f"\t the validation set is "
          f"{dates_info['validation']['start_date']}:"
          f"{dates_info['validation']['end_date']} (inclusive)")


def plot_history(his, path):
    """绘图"""
    h = pd.DataFrame(his.history)
    h[["loss", "val_loss"]].plot()
    plt.savefig(path + "loss.png")
    plt.show()
    plt.close()
    h[["root_mean_squared_error", "val_root_mean_squared_error"]].plot()
    plt.savefig(path + "root_mean_squared_error.png")
    plt.show()
    plt.close()
    h[["up_down_accuracy", "val_up_down_accuracy"]].plot()
    plt.savefig(path + "accuracy.png")
    plt.show()
    plt.close()


def do_training(beginning_date,
                training_id,
                feature_count,
                data_producer,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                early_stopping_patience=EARLY_STOPPING_PATIENCE):
    """训练"""
    # get training data starting from beginning_date
    try:
        train, val, dates_info = data_producer.get(beginning_date)
    except ValueError as e:
        print(beginning_date, e)
        return None

    # print dates information
    print_info(dates_info)

    # build model
    net = AlphaNetV3(metrics=[tf.keras.metrics.RootMeanSquaredError(),
                              UpDownAccuracy()],
                     input_shape=(HISTORY_LENGTH, feature_count),
                     l2=L2,
                     dropout=DROPOUT,
                     alpha=LEARNING_RATE)
    m = net.model()

    # create a folder to save information for this model
    path_str = "./models/{training_id:02d}/{beginning_date:08d}/"
    folder_path = path_str.format(training_id=training_id,
                                  beginning_date=beginning_date)
    try_mkdirs(folder_path)

    # writes one model
    model_path = folder_path + "model"
    net.save(model_path)

    # write dates information
    json_path = folder_path + "dates_info.json"
    with open(json_path, "w") as fp:
        json.dump(dates_info, fp)

    # save model weights per epoch in folders
    file_path = folder_path + "{epoch:04d}-{val_loss:.6f}.hdf5"
    ckp = tf.keras.callbacks.ModelCheckpoint(filepath=file_path,
                                             save_freq="epoch",
                                             save_weights_only=True)

    # early stopping
    es = tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                          patience=early_stopping_patience,
                                          restore_best_weights=True)

    # fit model
    h = m.fit(train.batch(batch_size).cache(),
              validation_data=val.batch(batch_size).cache(),
              epochs=epochs,
              callbacks=[es, ckp])

    # save weights of the best model
    net.save_weights(folder_path + "best.hdf5")

    # plot loss
    plot_history(h, folder_path)

    # write history
    history_path = folder_path + "history.json"
    with open(history_path, "w") as fp:
        json.dump(h.history, fp)


if __name__ == "__main__":
    csi = process_df(DATA)
    stock_data_list = build_data_list(csi)
    feature_number = stock_data_list[0].data.shape[1]

    try_mkdirs("./models")

    data = TrainValData(time_series_list=stock_data_list,
                        train_length=TRAIN_LENGTH,
                        validate_length=VALIDATE_LENGTH,
                        history_length=HISTORY_LENGTH,
                        sample_step=SAMPLE_STEP,
                        train_val_gap=10)

    for tid in TRAINING_ID:
        for beginning in ROLLING_BEGINNING_LIST:
            do_training(beginning_date=beginning,
                        training_id=tid,
                        data_producer=data,
                        feature_count=feature_number)
