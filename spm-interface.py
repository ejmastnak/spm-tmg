import matplotlib
matplotlib.use("TkAgg")  # set tk backend
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from pathlib import Path
import os
import traceback
from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
import numpy as np
import spm1d


class MCModulationInterface:

    def __init__(self, base_filename=None, pot_filename=None):
        """
        For now includes filename parameters as a hacky way to load data automatically on startup for testing
        :param base_filename:
        :param pot_filename:
        """

        self.baseline_filename = ""  # name of file holding baseline data
        self.potentiated_filename = ""  # name of file holding potentiated data
        self.baseline_data = np.zeros(shape=(0, 0))  # e.g. 1000 x 10
        self.pot_data = np.zeros(shape=(0, 0))  # e.g. 1000 x 10

        self.new_data = True
        self.start_row = 1  # csv data file row at which to begin reading data (0-indexed)
        self.max_rows = 100  # number of rows to read after start_row is reached
        self.position_offset = 0.0
        self.normalize = False

        self.root = Tk()
        self.root.title("SPM 1D Analysis Interface")

        # create frames here
        self.rootframe = ttk.Frame(self.root, padding=(3, 12, 3, 12))
        self.controlframe = ttk.Frame(self.rootframe) # panel to hold controls
        self.graphframe = ttk.Frame(self.rootframe) # panel to hold graph

        # create control widgets
        self.import_potentiated_button = ttk.Button(self.controlframe, text="Import Potentiated Data", command=self.import_potentiated)
        self.import_baseline_button = ttk.Button(self.controlframe, text="Import Baseline Data", command=self.import_baseline)
        self.compare_button = ttk.Button(self.controlframe, text="Compare", command=self.compare)
        self.export_curve_button = ttk.Button(self.controlframe, text="Export t Curve", command=self.export_tcurve)
        self.export_params_button = ttk.Button(self.controlframe, text="Export t Parameters", command=self.export_tparams)
        self.close_button = ttk.Button(self.controlframe, text="Exit", command=self.close)

        # Baseline data window
        self.baseline_label = ttk.Label(self.controlframe, text="Baseline Data")
        self.baseline_scroll = ttk.Scrollbar(self.controlframe)
        self.baseline_text_area = Text(self.controlframe, height=5, width=52)
        self.baseline_scroll.config(command=self.baseline_text_area.yview)
        self.baseline_text_area.config(yscrollcommand=self.baseline_scroll.set)
        self.baseline_text_area.insert(END, "No data imported")
        self.baseline_text_area.configure(state='disabled')

        # Potentiated data window
        self.potentiated_label = ttk.Label(self.controlframe, text="Potentiated Data")
        self.potentiated_scroll = ttk.Scrollbar(self.controlframe)
        self.potentiated_text_area = Text(self.controlframe, height=5, width=52)
        self.potentiated_scroll.config(command=self.potentiated_text_area.yview)
        self.potentiated_text_area.config(yscrollcommand=self.potentiated_scroll.set)
        self.potentiated_text_area.insert(END, "No data imported")
        self.potentiated_text_area.configure(state='disabled')

        # SPM analysis results window
        self.spm_label = ttk.Label(self.controlframe, text="SPM Analysis Results")
        self.spm_scroll = ttk.Scrollbar(self.controlframe)
        self.spm_text_area = Text(self.controlframe, height=7, width=52)
        self.spm_scroll.config(command=self.spm_text_area.yview)
        self.spm_text_area.config(yscrollcommand=self.spm_scroll.set)
        self.spm_text_area.insert(END, "No analysis results")
        self.spm_text_area.configure(state='disabled')

        # # create signal graph
        # self.signal_fig = Figure(figsize=(6, 2), dpi=100) # figure to hold graph of muscle signals
        # t = np.arange(0, 3, .01)
        # self.signal_subplot = self.signal_fig.add_subplot(111)
        # self.signal_subplot.plot(t, 2 * np.sin(2 * np.pi * t))
        # self.signal_canvas = FigureCanvasTkAgg(self.signal_fig, master=self.graphframe)
        # self.signal_canvas.draw()
        #
        # # create spm graph
        # self.spm_fig = Figure(figsize=(6, 2), dpi=100) # figure to hold spm graph
        # # change sine to an hline, add nice labels and things
        # t = np.arange(0, 3, .01)
        # self.spm_subplot = self.spm_fig.add_subplot(111)
        # self.spm_subplot.plot(t, 2 * np.sin(2 * np.pi * t))
        # self.spm_canvas = FigureCanvasTkAgg(self.spm_fig, master=self.graphframe)
        # self.spm_canvas.draw()

        # gridding widgets
        self.rootframe.grid(column=0, row=0, sticky=(N, S, E, W)) # place the root frame

        self.graphframe.grid(column=0, row=0, sticky=(N, S, E, W)) # place graph frame
        self.controlframe.grid(column=1, row=0, sticky=(N, S, E, W)) # place control frame

        # gridding control panel
        self.baseline_label.grid(column=0, row=1)
        self.baseline_scroll.grid(column=1, row=2, sticky=W)
        self.baseline_text_area.grid(column=0, row=2, sticky=W)

        self.potentiated_label.grid(column=0, row=3)
        self.potentiated_scroll.grid(column=1, row=4, sticky=W)
        self.potentiated_text_area.grid(column=0, row=4, sticky=W)

        self.spm_label.grid(column=0, row=5)
        self.spm_scroll.grid(column=1, row=6, sticky=W)
        self.spm_text_area.grid(column=0, row=6, sticky=W)

        self.import_baseline_button.grid(column=0, row=7, sticky=W)
        self.import_potentiated_button.grid(column=0, row=8, sticky=W)
        self.compare_button.grid(column=0, row=9, sticky=W)
        self.export_curve_button.grid(column=0, row=10, sticky=W)
        self.export_params_button.grid(column=0, row=11, sticky=W)
        self.close_button.grid(column=0, row=12, sticky=W)

        # gridding graphs
        # self.signal_canvas.get_tk_widget().grid(column=0, row=0, sticky=(N, S, E, W))
        # self.spm_canvas.get_tk_widget().grid(column=0, row=1, sticky=(N, S, E, W))

        # configure weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        self.rootframe.columnconfigure(0, weight=1)  # control frame
        self.rootframe.columnconfigure(1, weight=2)  # graph frame
        self.rootframe.rowconfigure(0, weight=1)

        # for loading data programtically (and not via gui) when testing
        if base_filename is not None and pot_filename is not None:
            self.set_baseline_data(base_filename)
            self.set_potentiated_data(pot_filename)

        self.root.mainloop()

    @ staticmethod
    def get_data_description(filename, data):
        """
        Used to get a string description of imported baseline/potentiated data
        to display in the GUI, so the user knows what they've imported

        :param filename:
        :param data:
        :return:
        """

        file_string = "Filename: " + Path(filename).name
        dim_string_1 = "Dimensions: {} rows by {} columns".format(data.shape[0], data.shape[1])
        dim_string_2 = "({} measurements of {} points per measurement)".format(data.shape[1], data.shape[0])
        path_string = "Location: " + os.path.dirname(filename)
        return file_string + "\n" + dim_string_1 + "\n" + dim_string_2 + "\n" + path_string

    @ staticmethod
    def export_spm_params(ti):
        """
        Used to get a string description of an SPM two-sample t-test

        :param ti: An SPM TI inference object
        :return:
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
                cluster_string += "\nCentroid: ({:.2f}, {:.2f})".format(cluster.centroid[0], cluster.centroid[1])
                cluster_string += "\nMaximum: {:.2f}".format(z_max)
                cluster_string += "\nArea Above Threshold: {:.2f}".format(A_threshold)
                cluster_string += "\nArea Above x Axis: {:.2f}".format(A)
                analysis_string += cluster_string
        return analysis_string

    @ staticmethod
    def set_imported_data_description(text_widget, text_content):
        """
        Wrapper method for protocol of setting data text widget info content
        Used with to give the user a description of the data they've imported
        And to give an overview of SPM analysis results
        :param text_widget:
        :param text_content:
        :return:
        """
        text_widget.configure(state='normal')
        text_widget.delete('1.0', END)
        text_widget.insert(END, text_content)
        text_widget.configure(state='disabled')

    # -----------------------------------------------------------------------------
    # START GUI WIDGET ACTION FUNCTIONS
    # -----------------------------------------------------------------------------
    def import_baseline(self):
        """
        Action for the import_baseline_button widget.
        Implements the protocol for importing baseline data.
        """
        filename = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])  # get csv files
        if filename:  # only called if a file is chosen and avoids null filename on cancel click
            try:
                self.baseline_data = np.loadtxt(filename, delimiter=",", skiprows=self.start_row, max_rows=self.max_rows)  # load data
                self.process_baseline_data()  # process imported data
                self.set_imported_data_description(self.baseline_text_area, self.get_data_description(filename, self.baseline_data))
                self.baseline_filename = filename  # if import is successful, set baseline filename

            except Exception as e:
                print("Error importing data: " + str(e))
                traceback.print_tb(e.__traceback__)
                return

    def import_potentiated(self):
        """
        Action for the import_baseline_button widget.
        Implements protocol for importing potentiated data
        """
        filename = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])  # get csv files
        if filename:  # only called if a file is chosen and avoids null filename on cancel click
            try:
                self.pot_data = np.loadtxt(filename, delimiter=",", skiprows=self.start_row, max_rows=self.max_rows)  # load data
                self.process_potentiated_data()  # process imported data
                self.set_imported_data_description(self.potentiated_text_area, self.get_data_description(filename, self.pot_data))
                self.potentiated_filename = filename  # if import is successful, set potentiated filename

            except Exception as e:
                print("Error importing data: " + str(e))
                return

    def compare(self):
        """
        Action for the "Compare" button. Runs an SPM analysis between the imported baseline and potentiated data.
        """
        if self.is_import_data_null():  # null data check
            return

        base_shape = self.baseline_data.shape
        pot_shape = self.pot_data.shape

        base_rows = base_shape[0]
        base_cols = base_shape[1]
        pot_rows = pot_shape[0]
        pot_cols = pot_shape[1]

        if base_rows != pot_rows and base_cols == pot_cols: # same number columns, different number of rows
            self.match_rows(base_rows, pot_rows)

        elif base_rows == pot_rows and base_cols != pot_cols: # same number rows, different number columns
            self.match_cols(base_rows, base_cols, pot_rows, pot_cols)

        elif base_shape != pot_shape: # different number of rows AND different number of columns
            self.match_rows(base_rows, pot_rows) # first match rows
            self.match_cols(base_rows, base_cols, pot_rows, pot_cols) # then match columns

        if base_cols == 1 and pot_cols == 1:
            self.increase_cols(base_rows, base_cols, pot_rows, pot_cols)

        self.run_two_sample_test(self.baseline_data.T, self.pot_data.T, x_label="Time [ms]", y_label="Position [mm]")
        # print("Hi!")
        # try:
        #     t = spm1d.stats.ttest2(self.potentiated_data.T, self.baseline_data.T, equal_var=False)
        #     ti = t.inference(alpha=0.05, two_tailed=False, interp=True)
        #     self.update_signal_graph(self.baseline_data, self.potentiated_data)
        #     self.update_spm_graph(t, ti)
        #
        # except Exception as e:
        #     print("Error performing SPM analysis: " + str(e))
        #     return

    def export_tcurve(self, *args):  # used to write an output file
        """
        Action for the export t-curve button widget.
        Export the t-statistic as a 2D array to a local CSV file.
        First column is time, second is t-statistic
        """
        if self.is_import_data_null():  # null data check
            return

        try:
            t = spm1d.stats.ttest2(self.pot_data.T, self.baseline_data.T, equal_var=False)
            filename = filedialog.asksaveasfilename(filetypes=[("CSV files", "*.csv")])
            if filename is None or filename == "":  # in case export was cancelled
                return
            if not filename.endswith(".csv"):  # append .csv extension, unless user has done so manually
                filename += ".csv"
            time = np.linspace(0, len(t.z)-1, len(t.z)) + self.start_row  # assumes 1 kHz sampling, i.e. 1ms per sample
            header = "Time [ms], SPM t-statistic"
            np.savetxt(filename, np.column_stack([time, t.z]), delimiter=',', header=header)
        except Exception as e:
            print("Error performing SPM analysis: " + str(e))
            return

    def export_tparams(self, *args):  # used to write an output file
        """
        Action for the export parameters button widget.
        Exports various ti object parameters to a local text file
        """
        if self.is_import_data_null():  # null data check
            return

        try:
            t = spm1d.stats.ttest2(self.pot_data.T, self.baseline_data.T, equal_var=False)
            ti = t.inference(alpha=0.05, two_tailed=False, interp=True)
            filename = filedialog.asksaveasfilename(filetypes=[("Text files", "*.csv")])
            if filename is None or filename == "":  # in case export was cancelled
                return
            if not filename.endswith(".csv"):  # append .csv extension, unless user has done so manually
                filename += ".csv"

            # Print file header
            metadata = "# Start header\n"
            metadata += "Baseline file,{}\n".format(Path(self.baseline_filename).name)
            metadata += "Potentiated file,{}\n".format(Path(self.potentiated_filename).name)
            metadata += "Alpha,{:.2f}\n".format(ti.alpha)  # alpha value
            metadata += "Threshold,{:.2f}\n".format(ti.zstar)  # threshold t-statistic value
            metadata += "# End header\n"

            with open(filename, 'w') as output:  # open file for writing
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
                                  "Start",
                                  "End",
                                  "Centroid Time",
                                  "Centroid t-value",
                                  "Maximum",
                                  "Area Above Threshold",
                                  "Area Above x Axis"]

                    for i, cluster in enumerate(clusters):  # loop through significance clusters
                        tstart, tend = cluster.endpoints  # start and end time of each cluster
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
                        param_strs[4] += ",{:.2f}".format(cluster.centroid[0])
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
            return

    def close(self):
        """
        Action for the close button widget.
        Closes the current SPM window and exits the program
        """
        self.root.destroy()
        sys.exit()
    # -----------------------------------------------------------------------------
    # END GUI WIDGET ACTION FUNCTIONS
    # -----------------------------------------------------------------------------

    def set_baseline_data(self, base_filename):
        """
        Action to programatically set baseline data
        Implements the protocol for importing baseline data.
        """
        if base_filename:  # only called if a file is chosen and avoids null filename on cancel click
            try:
                self.baseline_data = np.loadtxt(base_filename, delimiter=",", skiprows=self.start_row, max_rows=self.max_rows)  # load data
                self.process_baseline_data()  # process imported data
                self.set_imported_data_description(self.baseline_text_area, self.get_data_description(base_filename, self.baseline_data))
                self.baseline_filename = base_filename  # if import is successful, set baseline filename

            except Exception as e:
                print("Error importing baseline data: " + str(e))
                traceback.print_tb(e.__traceback__)
                return

    def set_potentiated_data(self, pot_filename):
        """
        Action to programatically set potentiated data
        Implements the protocol for importing potentiated data.
        """
        if pot_filename:  # only called if a file is chosen and avoids null filename on cancel click
            try:
                self.pot_data = np.loadtxt(pot_filename, delimiter=",", skiprows=self.start_row, max_rows=self.max_rows)  # load data
                self.process_potentiated_data()  # process imported data
                self.set_imported_data_description(self.potentiated_text_area, self.get_data_description(pot_filename, self.pot_data))
                self.potentiated_filename = pot_filename  # if import is successful, set potentiated filename

            except Exception as e:
                print("Error importing potentiated data: " + str(e))
                traceback.print_tb(e.__traceback__)
                return

    # -----------------------------------------------------------------------------
    # START SPM SIGNAL PROCESSING FUNCTIONS
    # -----------------------------------------------------------------------------
    def process_baseline_data(self):
        if self.baseline_data is None:  # null check
            return

        if self.normalize: self.baseline_data = self.baseline_data / self.baseline_data.max(axis=0)  # normalize
        self.baseline_data = self.baseline_data + self.position_offset  # add vertical offset

        self.baseline_data = self.baseline_data + self.position_offset

        # if data is a single column (1D array) reshape into a 2D array (matrix with one column)
        if len(self.baseline_data.shape) == 1:
            self.baseline_data = self.baseline_data.reshape(-1, 1)

    def process_potentiated_data(self):
        if self.pot_data is None:  # null check
            return

        if self.normalize: self.pot_data = self.pot_data / self.pot_data.max(axis=0)  # normalize
        self.pot_data = self.pot_data + self.position_offset  # add vertical offset

        self.pot_data = self.pot_data + self.position_offset

        # if data is a single column (1D array) reshape into a 2D array (matrix with one column)
        if len(self.pot_data.shape) == 1:
            self.pot_data = self.pot_data.reshape(-1, 1)

    # -----------------------------------------------------------------------------
    # END SPM SIGNAL PROCESSING FUNCTIONS
    # -----------------------------------------------------------------------------

    # -----------------------------------------------------------------------------
    # START DATA SHAPE ACCOMODATION FUNCTIONS
    # -----------------------------------------------------------------------------
    def match_rows(self, base_rows, pot_rows):
        """
        If there are more potentiated rows than baseline rows, trims number of rows in potentiated array
         to match number of rows in baseline array
        And vice versa for opposite case

        :param base_rows:
        :param pot_rows:
        :return:
        """
        if base_rows < pot_rows:  # more potentiated rows; trim potentiated to match baseline
            self.pot_data = self.pot_data[0:base_rows, :]

        elif base_rows > pot_rows:  # more baseline rows; trim baseline to match potentiated
            self.baseline_data = self.baseline_data[0:pot_rows, :]

    def match_cols(self, base_rows, base_cols, pot_rows, pot_cols):
        """
        If there are more potentiated columns than baseline columns, adds more columns to baseline array until the
         number of columns in baseline and potentiated match.
        And vice versa for opposite case

        Extra columns are found by taking the average of the existing columns, and then adding noise to each datapoint;
         the noise size is in the interval of +/1 0.1 of each data point's absolute value.

        :param base_cols:
        :param pot_cols:
        :return:
        """
        if base_cols < pot_cols:  # more potentiated columns; add more noisy averaged baseline columns
            temp_baseline_data = np.zeros(shape=(pot_rows, pot_cols))  # declare empty array with proper dimensions (more columns)
            col_avg = self.baseline_data.mean(axis=1)  # get column average

            for i, col in enumerate(self.baseline_data.T):  # fill expanded array's first columns with existing baseline data
                temp_baseline_data[:,i] = col
            for j in range(base_cols, pot_cols):
                temp_baseline_data[:,j] = col_avg + 0.1 * np.random.uniform(-np.abs(col_avg), abs(col_avg))  # add a noisy average of original columns. adds in range of \pm 10 percent of each data point

            self.baseline_data = temp_baseline_data # overwrite old data with correctly sized array

        elif base_cols > pot_cols:  # more baseline columns; add more noisy averaged potentiated columns
            temp_potentiated_data = np.zeros(shape=(base_rows, base_cols))  # declare empty array with proper dimensions (more columns)
            col_avg = self.pot_data.mean(axis=1)  # get column average

            for i, col in enumerate(self.pot_data.T):  # fill expanded array's first columns with existing potentiated data
                temp_potentiated_data[:,i] = col
            for j in range(pot_cols, base_cols):
                temp_potentiated_data[:,j] = col_avg + 0.1 * np.random.uniform(-np.abs(col_avg), abs(col_avg))  # add a noisy average of original columns. adds in range of \pm 10 percent of each data point

            self.pot_data = temp_potentiated_data # overwrite old data with correctly sized array

    def increase_cols(self, base_rows, base_cols, pot_rows, pot_cols):
        """
        TODO documentation of what happens if there is only one column of data

        Extra columns are found by taking the average of the existing columns, and then adding noise to each datapoint;
         the noise size is in the interval of +/1 0.1 of each data point's absolute value.

        :param base_cols:
        :param pot_cols:
        :return:
        """
        temp_baseline_data = np.zeros(shape=(pot_rows, 5))  # declare empty array with 5 columns
        col_avg = self.baseline_data.mean(axis=1)  # get column average

        for i, col in enumerate(self.baseline_data.T):  # fill expanded array's first columns with existing baseline data
            temp_baseline_data[:,i] = col
        for j in range(base_cols, pot_cols):
            temp_baseline_data[:,j] = col_avg + 0.1 * np.random.uniform(-np.abs(col_avg), abs(col_avg))  # add a noisy average of original columns. adds in range of \pm 10 percent of each data point

        self.baseline_data = temp_baseline_data # overwrite old data with correctly sized array

        temp_potentiated_data = np.zeros(shape=(base_rows, 5))  # declare empty array with 5 columns
        col_avg = self.pot_data.mean(axis=1)  # get column average

        for i, col in enumerate(self.pot_data.T):  # fill expanded array's first columns with existing potentiated data
            temp_potentiated_data[:,i] = col
        for j in range(pot_cols, base_cols):
            temp_potentiated_data[:,j] = col_avg + 0.1 * np.random.uniform(-np.abs(col_avg), abs(col_avg))  # add a noisy average of original columns. adds in range of \pm 10 percent of each data point

        self.pot_data = temp_potentiated_data # overwrite old data with correctly sized array
    # -----------------------------------------------------------------------------
    # END DATA SHAPE ACCOMODATION FUNCTIONS
    # -----------------------------------------------------------------------------

    def is_import_data_null(self):
        """
        Used as a null check for imported data
        Returns true (data is null) if there are either no rows or
         no columns in either of baseline and potentiated data arrays.

        e.g. base_shape.shape = (0, 100) and pot_shape.shape = (10, 100) returns true (data is null)
        """
        base_shape = self.baseline_data.shape
        pot_shape = self.pot_data.shape

        base_rows = base_shape[0]
        base_cols = base_shape[1]
        pot_rows = pot_shape[0]
        pot_cols = pot_shape[1]

        if base_rows*base_cols*pot_rows*pot_cols == 0: # quick way to check if any of the values are zero
            return True
        else:
            return False

    def run_two_sample_test(self, base_data, pot_data, x_label="Independent Variable", y_label="Dependent Variable"):
        """
        Runs and plots the results of a two-sample SPM t test between base_data and pot_data.
        Plot results in an external window using SPM's plotting utilities
        :param base_data: n x m double (n number of trials, m number of data points per trial, usually m \gg n)
        :param pot_data: n x m double
        :param x_label: label for x axis
        :param y_label: label for x axis
        :return:
        """
        try:
            t = spm1d.stats.ttest2(pot_data, base_data, equal_var=False)
            ti = t.inference(alpha=0.05, two_tailed=False, interp=True)
        except Exception as e:
            print("Error performing SPM analysis: " + str(e))
            return

        self.set_imported_data_description(self.spm_text_area, self.export_spm_params(ti))

        #  Plot:
        plt.close('all')
        # plot mean and SD:
        plt.figure(figsize=(8, 3.5))
        ax = plt.axes((0.1, 0.15, 0.35, 0.8))

        spm1d.plot.plot_mean_sd(pot_data, label="Potentiated", linecolor='r', facecolor='r')
        spm1d.plot.plot_mean_sd(base_data, label="Baseline")

        ax.axhline(y=0, color='k', linestyle=':')
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        plt.legend()

        # plot SPM results:
        ax = plt.axes((0.55, 0.15, 0.35, 0.8))
        ti.plot()
        ax.text(73, ti.zstar + 0.4, "$\\alpha = {:.2f}$\n$t^* = {:.2f}$".format(ti.alpha, ti.zstar),
                va='bottom', ha='left', bbox=dict(facecolor='#FFFFFF', edgecolor='#222222', boxstyle='round,pad=0.3'))
        # ti.plot_threshold_label(fontsize=10, color='black')
        # ti.plot_p_values(size=10)  # offsets=[(0, 0.3)]
        ax.set_xlabel(x_label)
        plt.show()

    # -----------------------------------------------------------------------------
    # START GRAPH UPDATE FUNCTIONS
    # -----------------------------------------------------------------------------
    def plot_threshold(self, ax, threshold, alpha):
        """
        Plots threshold t-statistic value and corresponding alpha value on the inputted axis
        :param ax: matplotlib.axes._axes.Axes object
        :param threshold: threshold t-statistic value
        :param alpha: alpha value for t test
        """

    def update_spm_graph(self, t, ti):
        t_star = ti.zstar
        p = ti.p

        self.spm_subplot.clear()  # clear data
        self.spm_subplot.plot(t.z)  # plot data
        self.spm_fig.canvas.draw()  # updates the graph

    def update_signal_graph(self, base_data, pot_data):
        baseline_mean = np.mean(base_data, axis=1) # column average of baseline data
        potentiated_mean = np.mean(pot_data, axis=1)  # column average of baseline data

        self.signal_subplot.clear()  # clear data
        self.signal_subplot.plot(baseline_mean, label='baseline')  # plot data
        self.signal_subplot.plot(potentiated_mean, label='potentiated')  # plot data
        self.signal_fig.canvas.draw()  # updates the graph
    # -----------------------------------------------------------------------------
    # END GRAPH UPDATE FUNCTIONS
    # -----------------------------------------------------------------------------


def practice():
    filename = "test.txt"
    with open(filename, 'w') as output:
        output.write("Line 1")
        output.write("Line 2")


def gui_launch():
    interface = MCModulationInterface()


def development_launch():
    # load data programatically for development use
    data_dir = "/Users/ejmastnak/Documents/Dropbox/projects-and-products/tmg-bmc/spm/spm-measurements/spm_1_9_2020/em/"
    base_filename = data_dir + "em_base.csv"
    pot_filename = data_dir + "em_pot.csv"
    interface = MCModulationInterface(base_filename=base_filename, pot_filename=pot_filename)


if __name__ == "__main__":
    gui_launch()
    # development_launch()
    # practice()
