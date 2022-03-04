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


mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

MAX_EPOCHS = 20
CHECK_POINT_PATH = './checkpoints/'
FINAL_MODEL_PATH = '../models/rnn_7200'


class WindowGenerator():
    def __init__(self, input_width, label_width, shift,
                 train_df, val_df, test_df,
                 label_columns=None):
        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                          enumerate(label_columns)}
        self.column_indices = {name: i for i, name in
                               enumerate(train_df.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[
            self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[
            self.labels_slice]

    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]]
                    for name in self.label_columns],
                axis=-1)

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    def plot(self, model=None, plot_col='mempoolsize', max_subplots=3):
        inputs, labels = self.example
        plt.figure(figsize=(12, 8))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))
        for n in range(max_n):
            plt.subplot(max_n, 1, n+1)
            plt.ylabel(f'{plot_col} [normed]')
            plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                     label='Inputs', marker='.', zorder=-10)

            if self.label_columns:
                label_col_index = self.label_columns_indices.get(
                    plot_col, None)
            else:
                label_col_index = plot_col_index

            if label_col_index is None:
                continue

            plt.scatter(self.label_indices, labels[n, :, label_col_index],
                        edgecolors='k', label='Labels', c='#2ca02c', s=64)
            if model is not None:
                predictions = model(inputs)
                plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                            marker='X', edgecolors='k', label='Predictions',
                            c='#ff7f0e', s=64)

            if n == 0:
                plt.legend()

        plt.xlabel('Time [h]')

    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=32,)

        ds = ds.map(self.split_window)

        return ds

    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)

    @property
    def example(self):
        """Get and cache an example batch of `inputs, labels` for plotting."""
        result = getattr(self, '_example', None)
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            result = next(iter(self.train))
            # And cache it for next time
            self._example = result
        return result

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])


def as_date(ts):
    return(datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S'))


def clean_dataset(dataset_path):
    df = pd.read_json(dataset_path)
    cols = list(df.columns)

    # Drop meta data fields
    df.drop('txid', axis=1, inplace=True)
    df.drop('hash', axis=1, inplace=True)
    df.drop('version', axis=1, inplace=True)
    df.drop('locktime', axis=1, inplace=True)
    df.drop('vsize', axis=1, inplace=True)
    if 'conf' in cols:
        df.drop('conf', axis=1, inplace=True)
    # TODO do we really need to drop net difficulty
    df.drop(columns='networkdifficulty', inplace=True)
    # Sort by date value
    df = df.sort_values(by='mempooldate')
    # Datify mempool date
    df.mempooldate = pd.to_datetime(df.mempooldate.apply(as_date))

    df.set_index('mempooldate', drop=True, inplace=True)
    # group by each unique timestamp
    df = df.reset_index().groupby('mempooldate').mean()

    # resample to 15 sec intervals and foward fill when NA;s get created
    # pad -> forward fill
    # iloc -> first row is all na's skip that boi
    df = df.resample('15S').pad().iloc[1:, :]
    # Split data

    return df


def min_max(df):
    mean = df.mean()
    std = df.std()

    return (df - mean) / std


def reverse_min_max(value, mean, std):
    return (value * std) + mean


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
    train_mean = train_df.mean()
    train_std = train_df.std()

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

    print('new test describe: ', new_test_mempool_size_mean,
          new_test_mempool_size_std)
    print('test describe: ', test_mempool_size_mean, test_mempool_size_std)

# 1 time unit = 15 seconds
    wide_window = WindowGenerator(
        input_width=1200, label_width=1200, shift=1,
        label_columns=['mempoolsize'],
        train_df=train_df,
        val_df=val_df,
        test_df=test_df
    )

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
