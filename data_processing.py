from pathlib import Path
import os
import traceback
import tkinter as tk
import numpy as np
import spm1d


# -----------------------------------------------------------------------------
# A collection of functions used as first aid to ensure baseline and potentiated measurement files
#  intended for use with SPM have the same number of rows and columns
# -----------------------------------------------------------------------------
def match_rows(baseline_data, active_data, base_rows, active_rows):
    """
    If there are more potentiated rows than baseline rows, trims number of rows in potentiated array
     to match number of rows in baseline array
    And vice versa for opposite case

    :param baseline_data: 2D numpy array containing baseline measurement data
    :param active_data: 2D numpy array containing "active" measurement data---either potentiated or atrophied
    :param base_rows:
    :param active_rows:
    :return:
    """
    if base_rows < active_rows:  # more potentiated rows; trim potentiated to match baseline
        active_data = active_data[0:base_rows, :]

    elif base_rows > active_rows:  # more baseline rows; trim baseline to match active
        baseline_data = baseline_data[0:active_rows, :]

    return baseline_data, active_data


def match_cols(baseline_data, active_data, base_rows, base_cols, active_rows, active_cols):
    """
    If there are more potentiated columns than baseline columns, adds more columns to baseline array until the
     number of columns in baseline and potentiated match.
    And vice versa for opposite case

    Extra columns are found by taking the average of the existing columns, and then adding noise to each datapoint;
     the noise size is in the interval of +/1 0.1 of each data point's absolute value.

    :param baseline_data: 2D numpy array containing baseline measurement data
    :param active_data: 2D numpy array containing "active" measurement data---either potentiated or atrophied
    :param base_rows
    :param base_cols:
    :param active_rows:
    :param active_cols:
    :return:
    """
    if base_cols < active_cols:  # more potentiated columns; add more noisy averaged baseline columns
        temp_baseline_data = np.zeros(
            shape=(active_rows, active_cols))  # declare empty array with proper dimensions (more columns)
        col_avg = baseline_data.mean(axis=1)  # get column average

        for i, col in enumerate(baseline_data.T):  # fill expanded array's first columns with existing baseline data
            temp_baseline_data[:, i] = col
        for j in range(base_cols, active_cols):
            temp_baseline_data[:, j] = col_avg + 0.1 * np.random.uniform(-np.abs(col_avg), abs(
                col_avg))  # add a noisy average of original columns. adds in range of \pm 10 percent of each data point
        baseline_data = temp_baseline_data  # overwrite old data with correctly sized array

    elif base_cols > active_cols:  # more baseline columns; add more noisy averaged potentiated columns

        temp_active_data = np.zeros(
            shape=(base_rows, base_cols))  # declare empty array with proper dimensions (more columns)
        col_avg = active_data.mean(axis=1)  # get column average

        for i, col in enumerate(active_data.T):  # fill expanded array's first columns with existing potentiated data
            temp_active_data[:, i] = col
        for j in range(active_cols, base_cols):
            temp_active_data[:, j] = col_avg + 0.1 * np.random.uniform(-np.abs(col_avg), abs(
                col_avg))  # add a noisy average of original columns. adds in range of \pm 10 percent of each data point

        active_data = temp_active_data  # overwrite old data with correctly sized array

    return baseline_data, active_data


def increase_cols(baseline_data, active_data, base_rows, base_cols, active_rows, active_cols):
    """
    Extra columns are found by taking the average of the existing columns, and then adding noise to each datapoint;
     the noise size is in the interval of +/1 0.1 of each data point's absolute value.

    :param baseline_data: 2D numpy array containing baseline measurement data
    :param active_data: 2D numpy array containing "active" measurement data---either potentiated or atrophied
    :param base_rows
    :param base_cols:
    :param active_rows:
    :param active_cols:
    :return:
    """
    temp_baseline_data = np.zeros(shape=(active_rows, 5))  # declare empty array with 5 columns
    col_avg = baseline_data.mean(axis=1)  # get column average
    for i, col in enumerate(baseline_data.T):  # fill expanded array's first columns with existing baseline data
        temp_baseline_data[:, i] = col
    for j in range(base_cols, active_cols):
        temp_baseline_data[:, j] = col_avg + 0.1 * np.random.uniform(-np.abs(col_avg), abs(
            col_avg))  # add a noisy average of original columns. adds in range of \pm 10 percent of each data point
    baseline_data = temp_baseline_data # overwrite old data with correctly sized array

    temp_active_data = np.zeros(shape=(base_rows, 5))  # declare empty array with 5 columns
    col_avg = active_data.mean(axis=1)  # get column average
    for i, col in enumerate(active_data.T):  # fill expanded array's first columns with existing potentiated data
        temp_active_data[:, i] = col
    for j in range(active_cols, base_cols):
        temp_active_data[:, j] = col_avg + 0.1 * np.random.uniform(-np.abs(col_avg), abs(
            col_avg))  # add a noisy average of original columns. adds in range of \pm 10 percent of each data point
    active_data = temp_active_data  # overwrite old data with correctly sized array

    return baseline_data, active_data
# -----------------------------------------------------------------------------
# END DATA SHAPE ACCOMODATION FUNCTIONS
# -----------------------------------------------------------------------------
