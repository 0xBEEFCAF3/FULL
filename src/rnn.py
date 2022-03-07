#!/usr/bin/env python3

import os
import sys
import datetime

import math
import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from scipy import stats
from datetime import datetime
from minmax import min_max, reverse_min_max
from clean import clean_dataset
from window_generator import WindowGenerator

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

MAX_EPOCHS = 20
CHECK_POINT_PATH = './checkpoints/'
FINAL_MODEL_PATH = '../models/rnn_7200'


def plot(window, model=None, plot_col='mempoolsize'):
    inputs, labels = window.example
    fig = plt.figure(figsize=(12, 8))
    plot_col_index = window.column_indices[plot_col]

    plt.ylabel(f'{plot_col} [normed]')
    plt.plot(window.input_indices, inputs[0, :, plot_col_index],
             label='Inputs', marker='.', zorder=-10)

    if window.label_columns:
        label_col_index = window.label_columns_indices.get(plot_col, None)
    else:
        label_col_index = plot_col_index

    plt.scatter(window.label_indices, labels[0, :, label_col_index],
                edgecolors='k', label='Labels', c='#2ca02c', s=64)
    if model is not None:
        predictions = model(inputs)
        plt.scatter(window.label_indices, predictions[0, :, label_col_index],
                    marker='X', edgecolors='k', label='Predictions',
                    c='#ff7f0e', s=64)

        plt.legend()

    plt.xlabel('15 sec')

    fig.show()


def main():
    new_data_df = clean_dataset('../data/new_test_data.json')

    df = clean_dataset('../data/split_data.json')
    n = len(df)
    train_df = df[0:int(n*0.7)]
    val_df = df[int(n*0.7):int(n*0.9)]
    test_df = df[int(n*0.9):]

    num_features = df.shape[1]

    # Normalize the dataset with min / max

    test_mean = train_df.mean()
    test_std = train_df.std()

    train_df = min_max(train_df)
    val_df = min_max(val_df)
    test_df = min_max(test_df)

    new_test_df_mean = new_data_df.mean()
    new_test_df_std = new_data_df.std()
    new_test_df = min_max(new_data_df)

    test_mempool_size_mean = test_mean['mempoolsize']
    test_mempool_size_std = test_std['mempoolsize']

    new_test_mempool_size_mean = new_test_df_mean['mempoolsize']
    new_test_mempool_size_std = new_test_df_std['mempoolsize']

    # 1 time unit = 15 seconds
    # wide_window = WindowGenerator(
    #     input_width=1200, label_width=1200, shift=1,
    #     label_columns=['mempoolsize'],
    #     train_df=train_df,
    #     val_df=val_df,
    #     test_df=test_df
    # )

    test_data_window = WindowGenerator(
        input_width=1200, label_width=1200, shift=1,
        label_columns=['mempoolsize'],
        train_df=new_test_df,
        val_df=new_test_df,
        test_df=new_test_df
    )

    model = tf.keras.models.Sequential([
        # Shape [batch, time, features] => [batch, time, lstm_units]
        tf.keras.layers.LSTM(32, return_sequences=True),
        # Shape => [batch, time, features]
        tf.keras.layers.Dense(units=1)
    ])

    #  checkpoint_dir = os.path.dirname(CHECK_POINT_PATH)

  # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=CHECK_POINT_PATH,
                                                     save_weights_only=True,
                                                     verbose=1)

    def compile_and_fit(model, window, patience=2):
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                          patience=patience,
                                                          mode='min')

        model.compile(loss=tf.losses.MeanSquaredError(),
                      optimizer=tf.optimizers.Adam(),
                      metrics=[tf.metrics.MeanAbsoluteError()])

        history = model.fit(window.train, epochs=MAX_EPOCHS,
                            validation_data=window.val,
                            callbacks=[early_stopping, cp_callback])
        return history

    # history = compile_and_fit(model, wide_window)

    # Load model and new testing data and calculate MAE
    model = tf.keras.models.load_model(FINAL_MODEL_PATH)
    predictions = model.predict(list(test_data_window.test)[0][0])
    labels = list(test_data_window.test)[0][1]

    count = 0
    diff = []

    # Calculate MAE
    for batch_index, prediction_batch in enumerate(predictions):
        for prediction_index, prediction in enumerate(prediction_batch):
            label = reverse_min_max(
                labels[batch_index][prediction_index], test_mempool_size_mean, test_mempool_size_std)
            prediction = reverse_min_max(
                prediction, new_test_mempool_size_mean, new_test_mempool_size_std)

            diff.append(
                float(abs(label - prediction)))
            count += 1

    print("new data MAE: " + str(sum(diff) / count))

    plot(test_data_window, model)

    # Evaluate the model
    # loss, acc = model.evaluate(wide_window.test, verbose=2)
    # print("test data MAE: " + str(acc))


if __name__ == "__main__":
    main()
