from pathlib import Path
import traceback
from tkinter import filedialog
import numpy as np
import spm1d

import plotting
import data_processing


def get_spm_ti_string_description(ti, time_offset=0):
    """
    Returns a string description of an SPM TI inference object's important parameters, e.g. start time, centroid, etc...
    Used to provide a string description of SMP inference results in the GUI
    :param ti: An SPM TI inference object
    :param time_offset: integer [milliseconds] to correct for potentially changing SPM start time
    """

    # Start header
    # Potentiated file,"filename"
    # Potentiated file,"filename"
    # Alpha,
    # T-star,
    # # End header
    # Parameter,Cluster,Cluster

    analysis_string = "Alpha: {:.2f}".format(ti.alpha)  # alpha value
    analysis_string += "\nThreshold: {:.2f}".format(ti.zstar)  # threshold t-statistic value
    clusters = ti.clusters  # portions of curve above threshold value
    threshold = ti.zstar
    if clusters is not None:
        for i, cluster in enumerate(clusters):
            tstart, tend = cluster.endpoints  # start and end time of each cluster
            tstart += time_offset  # add potential time offset
            tend += time_offset
            x, z = cluster._X, cluster._Z  # x and z (time and t-statistic) coordinates of the cluster
            z_max = np.max(z)  # max value of t-statistic in this cluster
            N = len(x)  # number of points in this cluster
            A = 0.0  # area under curve
            for k in range(1, N, 1):  # i = 1, 2, ..., N
                A += np.abs(0.5*(z[k] + z[k-1]))*(x[k] - x[k-1])  # midpoint formula
            A_threshold = A - (threshold * (x[-1] - x[0]))  # subtract area below threshold (threshold * interval length)

            cluster_string = "\n" + 50*"-"  # draws a bunch of dashes i.e. ----------
            cluster_string += "\nSignificance Region {}".format(i+1)  # include a newline character
            cluster_string += "\nProbability: {:.2e}".format(cluster.P)
            cluster_string += "\nProbability (decimal): {:.4f}".format(cluster.P)
            cluster_string += "\nStart: {:.2f}\t End: {:.2f}".format(tstart, tend)
            cluster_string += "\nCentroid: ({:.2f}, {:.2f})".format(cluster.centroid[0] + time_offset, cluster.centroid[1])
            cluster_string += "\nMaximum: {:.2f}".format(z_max)
            cluster_string += "\nArea Above Threshold: {:.2f}".format(A_threshold)
            cluster_string += "\nArea Above x Axis: {:.2f}".format(A)
            analysis_string += cluster_string
    return analysis_string


def export_t_curve(t, time_offset=0):
    """
    Exports the SPM t statistic as a function of time to a local csv file chosen by the user
    Assumes 1 kHz sampling of data! (as is standard for TMG measurements)
    :param t: 1D numpy array containing an SPM t statistic
    :param time_offset: integer [milliseconds] to correct for potentially changing SPM start time
    """
    try:
        filename = filedialog.asksaveasfilename(filetypes=[("CSV files", "*.csv")])
        if filename is None or filename == "":  # in case export was cancelled
            return
        if not filename.endswith(".csv"):  # append .csv extension, unless user has done so manually
            filename += ".csv"
        time = np.arange(0, len(t.z), 1)  # assumes 1 kHz sampling, i.e. 1ms per sample. Time reads 0, 1, 2, ...
        time += time_offset  # add potential time offset
        header = "Time [ms], SPM t-statistic"
        np.savetxt(filename, np.column_stack([time, t.z]), delimiter=',', header=header)

    except Exception as e:
        print("Error performing exporting SPM data: " + str(e))
        return


def export_ti_parameters(ti, time_offset=0, baseline_filename="", active_filename="", mode_name="Potentiated", output_file_path=None):
    """
    Exports various ti object parameters to a local csv file in tabular format
    :param ti: An SPM TI inference object
    :param time_offset: integer [milliseconds] to correct for potentially changing SPM start time
    :param baseline_filename: name of baseline data file, just for reference in the exported information
    :param active_filename: name of active data file, just for reference in the exported information
    :param mode_name: either "Potentiated" or "Atrophied"
    :param output_file_path: full path for output file. If None, a GUI file chooser is used instead.
    """
    try:
        if output_file_path is None:  # use GUI file chooser to determine output file path if no path specified
            output_file_path = filedialog.asksaveasfilename(filetypes=[("Text files", "*.csv")])
            if output_file_path is None or output_file_path == "":  # in case export was cancelled
                return
            if not output_file_path.endswith(".csv"):  # append .csv extension, unless user has done so manually
                output_file_path += ".csv"

        # Print file header
        metadata = "# START HEADER\n"
        metadata += "# Baseline file,{}\n".format(Path(baseline_filename).name)
        metadata += "# {} file,{}\n".format(mode_name, Path(active_filename).name)
        metadata += "# Alpha,{:.2f}\n".format(ti.alpha)  # alpha value
        metadata += "# Threshold,{:.2f}\n".format(ti.zstar)  # threshold t-statistic value
        metadata += "# END HEADER\n"

        with open(output_file_path, 'w') as output:  # open file for writing
            output.write(metadata)  # write metadata

            clusters = ti.clusters  # portions of curve above threshold value
            threshold = ti.zstar

            if clusters is None:  # catch possibility that threshold is not exceeded
                output.write("Significance threshold not exceeded.")
            else:

                # Create header string of the form "# Parameter,Cluster 1,Cluster 2, ..."
                header = "# Parameter"
                for i in range(len(clusters)):
                    header += ",Cluster {}".format(i + 1)
                header += "\n"  # add new line
                output.write(header)  # write header

                # Assign each outputted parameter a row; pack into an array for easier printing to file
                param_strs = ["Probability",  # probability for threshold in exponent (scientific) notation
                              "Probability (decimal)",  # probability as a float
                              "Start Time [ms]",
                              "End Time [ms]",
                              "Centroid Time [ms]",
                              "Centroid t-value",
                              "Maximum",
                              "Area Above Threshold",
                              "Area Above x Axis"]

                for i, cluster in enumerate(clusters):  # loop through significance clusters
                    tstart, tend = cluster.endpoints  # start and end time of each cluster
                    tstart += time_offset  # add potential time offset
                    tend += time_offset

                    x, z = cluster._X, cluster._Z  # x and z (time and t-statistic) coordinates of the cluster
                    z_max = np.max(z)  # max value of t-statistic in this cluster
                    N = len(x)  # number of points in this cluster
                    A = 0.0  # area under curve
                    for k in range(1, N, 1):  # i = 1, 2, ..., N
                        A += np.abs(0.5*(z[k] + z[k-1]))*(x[k] - x[k-1])  # midpoint formula
                    A_threshold = A - (threshold * (x[-1] - x[0]))  # subtract area below threshold (threshold * interval length)

                    param_strs[0] += ",{:.2e}".format(cluster.P)
                    param_strs[1] += ",{:.4f}".format(cluster.P)
                    param_strs[2] += ",{:.2f}".format(tstart)
                    param_strs[3] += ",{:.2f}".format(tend)
                    param_strs[4] += ",{:.2f}".format(cluster.centroid[0] + time_offset)
                    param_strs[5] += ",{:.2f}".format(cluster.centroid[1])
                    param_strs[6] += ",{:.2f}".format(z_max)
                    param_strs[7] += ",{:.2f}".format(A_threshold)
                    param_strs[8] += ",{:.2f}".format(A)

                # print parameter strings---this is where it's useful the strings are in an array
                for i, param_str in enumerate(param_strs):
                    output.write(param_str)  # write header
                    if i < len(param_strs):  # don't print new line for last string at end of file
                        output.write("\n")

    except Exception as e:
        print("Error exporting SPM parameters: " + str(e))
        traceback.print_exception(type(e), e, e.__traceback__)
        return


def get_spm_t(baseline_data, active_data):
    """
    Returns the spm.t object resulting from an SMP t test between the inputted baseline and active data
    :param active_data: 2D numpy array containing "active" measurement data---either potentiated or atrophied
    :param baseline_data: 2D numpy array containing baseline measurement data
    """
    try:
        # t = spm1d.stats.ttest2(self.baseline_data.T, self.pot_data.T, equal_var=False)
        t = spm1d.stats.ttest2(active_data.T, baseline_data.T, equal_var=False)
        return t
    except Exception as e:
        print("Error performing SPM analysis: " + str(e))
        return None


def get_spm_ti(baseline_data, active_data):
    """
    Returns the spm.t and spm.ti objects resulting from an SMP t test between the inputted baseline and active data
    :param active_data: 2D numpy array containing "active" measurement data---either potentiated or atrophied
    :param baseline_data: 2D numpy array containing baseline measurement data
    """
    try:
        # t = spm1d.stats.ttest2(self.baseline_data.T, self.pot_data.T, equal_var=False)
        t = spm1d.stats.ttest2(active_data.T, baseline_data.T, equal_var=False)
        ti = t.inference(alpha=0.05, two_tailed=False, interp=True)
        return t, ti
    except Exception as e:
        print("Error performing SPM analysis: " + str(e))
        return None


def set_file_smp_analysis(parent_dir, file_basename, sets_to_convert):
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
    """

    start_row = 1  # csv data file row at which to begin reading data (0-indexed)
    max_rows = 100  # number of rows to read after start_row is reached

    for set_num in sets_to_convert:
        base_filename = parent_dir + file_basename + "-base-{}.csv".format(set_num)
        pot_filename = parent_dir + file_basename + "-pot-{}.csv".format(set_num)

        params_output_path = parent_dir + file_basename + "-set{}-params.csv".format(set_num)
        figure_output_path = parent_dir + file_basename + "-set{}.png".format(set_num)
        base_data = np.loadtxt(base_filename, delimiter=",", skiprows=start_row, max_rows=max_rows)  # load data
        pot_data = np.loadtxt(pot_filename, delimiter=",", skiprows=start_row, max_rows=max_rows)  # load data
        base_data, pot_data = data_processing.fix_false_spm_significance(base_data, pot_data, mode=data_processing.BASE_POT)

        t, ti = get_spm_ti(base_data, pot_data)
        export_ti_parameters(ti, time_offset=start_row, baseline_filename=base_filename, active_filename=pot_filename, mode_name="Potentiated", output_file_path=params_output_path)
        plotting.plot_test_results(t, ti, base_data, pot_data, figure_output_path, time_offset=start_row, mode=plotting.BASE_POT, mode_name="Potentiated", show_plot=False)


def set_file_analysis_wrapper():
    """
    Wrapper method for performing SPM analysis where each set is separately analyzed.
    How to use:
        - Define parent directory containing meeasurement directories of individual athletes
        - Define a list of sets to convert for each individual athlete
    """
    parent_dir = "/Users/ejmastnak/Documents/Media/tmg-bmc-media/measurements/analysis-spm-potenciacija-19-03-2021/"
    directories = ["MM20210319105636_1/", "EP20210319102018/", "MM20210319115856/", "NF20210319122049/", "NF20210319113716/", "ZI20210319123747/"]
    sets_to_convert = [list(range(2, 8)), list(range(1, 9)), list(range(1, 5)), list(range(1, 5)), list(range(1, 6)), list(range(1, 6))]

    for i, dir in enumerate(directories):
        print(dir)
        file_basename = dir[0:-1]  # drop backslash
        set_file_smp_analysis(parent_dir + dir, file_basename, sets_to_convert[i])
        print()


if __name__ == "__main__":
    set_file_analysis_wrapper()
