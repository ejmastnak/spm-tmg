import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import plotting
import analysis

# A script for development/testing/debuggin code not intendend for deployment


def start_spike_param_analysis():
    """ Just my own testing code for figuring out the initial SPM spike """

    # Loop through many measurement folders
    # Loop through *params.csv files in each measurement folder
    # Use pandas to read file, convert to numpy. Skip rows 1 to 7. Begin reading line 8.
    # If there are multiple columns, the first is a false significance region
    # If there is one column---assume true column.
    # Store various significance region params

    parent_dir = "/Users/ejmastnak/Documents/Media/tmg-bmc-media/measurements/analysis-spm-potenciacija-19-03-2021/"
    directories = ["MM20210319105636_1/", "EP20210319102018/", "MM20210319115856/", "NF20210319122049/", "ZI20210319123747/"]

    # start time row 2
    # end time row 3
    # centroid time row 4

    false_spike = []  # store start time and end time
    true_spikes = []   # store start time and end time

    for meas_dir in directories:
        for filename in os.listdir(parent_dir + meas_dir):
            if "-params.csv" in filename:
                print(filename)
                filepath = parent_dir + meas_dir + filename
                # skip rows to skip header; drop first column; convert to numpy
                data = pd.read_csv(filepath, header=None, skiprows=7).drop(columns=[0]).to_numpy()

                num_rows, num_cols = np.shape(data)

                if num_cols == 1:  # a single significance region. I assume no intial spike and one true significance region.
                    # true_spikes.append((data[2][0], data[3][0], data[4][0], filename))
                    true_spikes.append((data[2][0], data[3][0], data[4][0]))
                else:
                    # true_spikes.append((data[2][0], data[3][0], data[4][0], filename))
                    false_spike.append((data[2][0], data[3][0], data[4][0]))
                    for i in range(1, num_cols):
                        # true_spikes.append((data[2][i], data[3][i], data[4][i], filename))
                        true_spikes.append((data[2][i], data[3][i], data[4][i]))

    false_spike = np.array(false_spike)  # columns are start time, end time, centroid time
    true_spikes = np.array(true_spikes)
    print(false_spike)
    print(true_spikes)

    true_centroid = true_spikes[:, 2]
    false_centroid = false_spike[:, 2]

    a = pd.DataFrame(true_centroid).describe()
    b = pd.DataFrame(false_centroid).describe()
    print(a)
    print(b)

    plt.hist(false_spike[:, 2])
    plt.hist(true_spikes[:, 2])
    plt.show()

    # use centroid greater than 10 as a cut-off


def plot_test_results(t, ti, baseline_data, active_data, figure_output_path, time_offset=0, mode_name="Potentiated", show_plot=True, save_figures=True):
    """
    Plots the baseline and active data's mean and standard deviation clouds on axis one
    Plots the SPM t-test between the baseline and active data on axis two

    :param t: An SPM T object TODO what this is
    :param ti: An SPM TI inference object
    :param active_data: 2D numpy array containing "active" measurement data---either potentiated or atrophied
    :param baseline_data: 2D numpy array containing baseline measurement data
    :param figure_output_path: path at which to save output plot
    :param time_offset: integer [milliseconds] to correct for potentially changing SPM start time
    :param mode_name: either BASE_POT or BASE_ATRO (to accomodate different colors in "potentiated" and "atrophied" (injured) modes)
    :param mode_name: either "Potentiated" or "Atrophied"
    :param show_plot: whether or not to show matplotlib plot. Generally turned off for automated processes.
    :param save_figures: whether or not to automatically save figure before plotting.
    """
    num_points = np.shape(active_data)[0]  # could also use base_data
    time = np.linspace(0, num_points - 1, num_points) + time_offset  # include time offset

    active_mean = np.mean(active_data, axis=1)
    base_mean = np.mean(baseline_data, axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    # plot TMG measurement
    ax = axes[0]
    plotting.remove_spines(ax)
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Position [mm]")

    lw = 1.0
    ax.plot(time, base_mean, color=plotting.base_color, linewidth=lw, marker='o', label="Baseline", zorder=4)  # plot in front
    ax.plot(time, active_mean, color=plotting.pot_color, linewidth=lw, marker='o', label=mode_name, zorder=3)

    ax.axhline(y=0, color='k', linestyle=':')  # dashed line at y = 0
    ax.legend()

    # plot SPM results:
    ax = axes[1]
    plotting.remove_spines(ax)
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("SPM $t$ Statistic", labelpad=-0.1)

    ax.plot(time, t.z, color=plotting.tline_color)  # plot t-curve
    ax.axhline(y=0, color='#000000', linestyle=':')  # dashed line at y = 0
    ax.axhline(y=ti.zstar, color='#000000', linestyle='--')  # dashed line at t threshold
    ax.text(73, ti.zstar + 0.4, "$\\alpha = {:.2f}$\n$t^* = {:.2f}$".format(ti.alpha, ti.zstar),
            va='bottom', ha='left', bbox=dict(facecolor='#FFFFFF', edgecolor='#222222', boxstyle='round,pad=0.3'))

    ax.fill_between(time, t.z, ti.zstar, where=t.z >= ti.zstar, interpolate=True, color=plotting.tpotfill_color)  # shade between curve and threshold

    if save_figures: plt.savefig(figure_output_path, dpi=150)

    if show_plot:  # either show plot...
        plt.show()
    else:  # or clear plot during automated batch tasks to clear memory
        plt.close(fig)


def plot_test():
    start_row = 1  # csv data file row at which to begin reading data (0-indexed)
    max_rows = 11  # number of rows to read after start_row is reached

    parent_dir = "/Users/ejmastnak/Documents/Media/tmg-bmc-media/measurements/analysis-spm-potenciacija-19-03-2021/EP20210319102018/"
    base_filename = parent_dir + "EP20210319102018-base-1.csv"
    pot_filename = parent_dir + "EP20210319102018-pot-1.csv"

    figure_output_path = parent_dir + "testfig.png"
    base_data = np.loadtxt(base_filename, delimiter=",", skiprows=start_row, max_rows=max_rows)  # load data
    pot_data = np.loadtxt(pot_filename, delimiter=",", skiprows=start_row, max_rows=max_rows)  # load data

    base_mean = np.mean(base_data[0:1, :], axis=1)[0]
    pot_mean = np.mean(pot_data[0:1, :], axis=1)[0]
    if pot_mean > base_mean:
        pot_data -= pot_mean - base_mean

    t, ti = analysis.get_spm_ti(base_data, pot_data)
    plot_test_results(t, ti, base_data, pot_data, figure_output_path, time_offset=start_row, mode_name="Potentiated", show_plot=True, save_figures=False)


def start_spike_plot_analysis(parent_dir, file_basename, sets_to_convert):
    """
    :param parent_dir: full path to a directory containing data files

     The files should be named in the form
        EM1234-base-1.csv
        EM1234-base-2.csv
        EM1234-base-3.csv
        EM1234-pot-1.csv
        EM1234-pot-2.csv
        EM1234-pot-3.csv
    :param file_basename: data file basename e.g. "EM1234"
    :param sets_to_convert: list of the set numbers of convert e.g. [1, 2, 3]
    :return:
    """

    start_row = 1  # csv data file row at which to begin reading data (0-indexed)
    max_rows = 11  # number of rows to read after start_row is reached

    for set_num in sets_to_convert:
        base_filename = parent_dir + file_basename + "-base-{}.csv".format(set_num)
        pot_filename = parent_dir + file_basename + "-pot-{}.csv".format(set_num)

        figure_output_path = "/Users/ejmastnak/spm-figs/" + "{}-{}.png".format(file_basename, set_num)
        base_data = np.loadtxt(base_filename, delimiter=",", skiprows=start_row, max_rows=max_rows)  # load data
        pot_data = np.loadtxt(pot_filename, delimiter=",", skiprows=start_row, max_rows=max_rows)  # load data

        avg_rows = 3
        amp_factor = 1.0
        base_mean = np.mean(np.mean(base_data[0:avg_rows, :], axis=1))  # the average of the average baseline signal over the first avg_row data points
        pot_mean = np.mean(np.mean(pot_data[0:avg_rows, :], axis=1))
        if pot_mean > base_mean:
            pot_data -= amp_factor * np.mean(pot_mean - base_mean)

        t, ti = analysis.get_spm_ti(base_data, pot_data)
        plot_test_results(t, ti, base_data, pot_data, figure_output_path, time_offset=start_row, mode_name="Potentiated", show_plot=False, save_figures=True)


def start_spike_plot_wrapper():
    """
    Wrapper method for performing SPM analysis where each set is separately analyzed.
    How to use:
        - Define parent directory containing meeasurement directories of individual athletes
        - Define a list of sets to convert for each individual athlete
    """
    # parent_dir = "/Users/ejmastnak/Documents/Media/tmg-bmc-media/measurements/analysis-spm-potenciacija-19-03-2021/"
    # directories = ["MM20210319105636_1/", "EP20210319102018/", "MM20210319115856/", "NF20210319122049/", "NF20210319113716/", "ZI20210319123747/"]
    # sets_to_convert = [list(range(2, 8)), list(range(1, 9)), list(range(1, 5)), list(range(1, 5)), list(range(1, 6)), list(range(1, 6))]

    parent_dir = "/Users/ejmastnak/Documents/Media/tmg-bmc-media/measurements/analysis-spm-potenciacija-19-03-2021/"
    directories = ["EP20210319102018/"]
    sets_to_convert = [list(range(1, 9))]

    for i, dir in enumerate(directories):
        print(dir)
        file_basename = dir[0:-1]  # drop backslash
        start_spike_plot_analysis(parent_dir + dir, file_basename, sets_to_convert[i])
        print()


if __name__ == "__main__":
    # start_spike_param_analysis()
    start_spike_plot_wrapper()
    # plot_test()
