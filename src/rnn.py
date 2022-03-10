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
FINAL_MODEL_PATH = '../models/rnn_240_multi_lstm'


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
    # new_data_df = clean_dataset('../data/new_test_data.json')

    df = clean_dataset('../data/split_data.json')
    n = len(df)
    train_df = df[0:int(n*0.7)]
    val_df = df[int(n*0.7):int(n*0.9)]
    test_df = df[int(n*0.9):]

    num_features = df.shape[1]

    # Normalize the dataset with min / max

    train_df = min_max(train_df, train_df.mean(), train_df.std())
    val_df = min_max(val_df, train_df.mean(), train_df.std())
    test_df = min_max(test_df, train_df.mean(), train_df.std())

    # 1 time unit = 15 seconds
    wide_window = WindowGenerator(
        input_width=240, label_width=240, shift=1,
        label_columns=['mempoolsize'],
        train_df=train_df,
        val_df=val_df,
        test_df=test_df
    )

    model = tf.keras.models.Sequential([
        # Shape [batch, time, features] => [batch, time, lstm_units]
        # tf.keras.layers.LSTM(32, return_sequences=True),
        # Shape => [batch, time, features]
        # tf.keras.layers.Dense(units=1)
        tf.keras.layers.LSTM(256, return_sequences=True),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.LSTM(256, return_sequences=False),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(num_features,
                              kernel_initializer=tf.initializers.zeros()),
        tf.keras.layers.Reshape([1, num_features])
    ])

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

    history = compile_and_fit(model, wide_window)
    model.save(FINAL_MODEL_PATH)

    # Evaluate the model
    loss, acc = model.evaluate(wide_window.test, verbose=2)
    print("MAE: " + str(acc))


if __name__ == "__main__":
    main()
