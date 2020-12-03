import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from pathlib import Path
import os
from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
import numpy as np
import spm1d


class MCModulationInterface:

    def __init__(self):

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
        self.export_button = ttk.Button(self.controlframe, text="Export SPM", command=self.export)
        self.close_button = ttk.Button(self.controlframe, text="Exit", command=self.close)

        self.baseline_label = ttk.Label(self.controlframe, text="Baseline Data")
        self.baseline_scroll = ttk.Scrollbar(self.controlframe)
        self.baseline_text = Text(self.controlframe, height=5, width=52)
        self.baseline_scroll.config(command=self.baseline_text.yview)
        self.baseline_text.config(yscrollcommand=self.baseline_scroll.set)
        self.baseline_text.insert(END, "No data imported")
        self.baseline_text.configure(state='disabled')

        self.potentiated_label = ttk.Label(self.controlframe, text="Potentiated Data")
        self.potentiated_scroll = ttk.Scrollbar(self.controlframe)
        self.potentiated_text = Text(self.controlframe, height=5, width=52)
        self.potentiated_scroll.config(command=self.potentiated_text.yview)
        self.potentiated_text.config(yscrollcommand=self.potentiated_scroll.set)
        self.potentiated_text.insert(END, "No data imported")
        self.potentiated_text.configure(state='disabled')

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
        self.baseline_text.grid(column=0, row=2, sticky=W)

        self.potentiated_label.grid(column=0, row=3)
        self.potentiated_scroll.grid(column=1, row=4, sticky=W)
        self.potentiated_text.grid(column=0, row=4, sticky=W)

        self.import_baseline_button.grid(column=0, row=5, sticky=W)
        self.import_potentiated_button.grid(column=0, row=6, sticky=W)
        self.compare_button.grid(column=0, row=7, sticky=W)
        self.export_button.grid(column=0, row=8, sticky=W)
        self.close_button.grid(column=0, row=9, sticky=W)

        # gridding graphs
        # self.signal_canvas.get_tk_widget().grid(column=0, row=0, sticky=(N, S, E, W))
        # self.spm_canvas.get_tk_widget().grid(column=0, row=1, sticky=(N, S, E, W))

        # configure weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        self.rootframe.columnconfigure(0, weight=1)  # control frame
        self.rootframe.columnconfigure(1, weight=2)  # graph frame
        self.rootframe.rowconfigure(0, weight=1)

        self.root.mainloop()

    @ staticmethod
    def get_data_description_string(filename, data):
        """
        Used to get a string description of imported baseline/potentiated data
        to display in the GUI, so the user knows what they've imported

        :param filename:
        :param data:
        :return:
        """

        if len(data.shape) == 2:  # a matrix
            file_string = "Data: " + Path(filename).name
            dim_string_1 = str("Dimensions: (" + str(data.shape[0]) + " x " + str(data.shape[1]) + ")")
            dim_string_2 = "(" + str(data.shape[1]) + " measurements of " + str(data.shape[0]) + " points per measurement)"
            path_string = "Location: " + os.path.dirname(filename)
            return file_string + "\n" + dim_string_1 + "\n" + dim_string_2 + "\n" + path_string

        elif len(data.shape) == 1:  # a single column
            file_string = "Data: " + Path(filename).name
            dim_string_1 = str("Dimensions: (" + str(data.shape[0]) + " x " + str(1) + ")")
            dim_string_2 = "(1 measurements of " + str(data.shape[0]) + " points per measurement)"
            path_string = "Location: " + os.path.dirname(filename)
            return file_string + "\n" + dim_string_1 + "\n" + dim_string_2 + "\n" + path_string

        else: return "Error importing data.\nData has wrong shape: " + str(data.shape)

    @ staticmethod
    def set_text_content_description_after_import(text_widget, text_content):
        """
        Wrapper method for protocol of setting data text widget info content
        The idea is to give the user a description of the data they've imported
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
                self.baseline_data = self.baseline_data + self.position_offset
                if self.normalize: self.baseline_data = self.baseline_data / self.baseline_data.max(axis=0)

                self.set_text_content_description_after_import(self.baseline_text, self.get_data_description_string(filename, self.baseline_data))
                if len(self.baseline_data.shape) == 1: # turn a 1D array into a 2D matrix with one column
                    self.baseline_data = self.baseline_data.reshape(-1, 1)
            except Exception as e:
                print("Error importing data: " + str(e))
                return

    def import_potentiated(self):
        """
        Action for the import_baseline_button widget.
        Implements protocol for importing potentiated data
        """
        filename = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])  # get csv files
        if filename: # only called if a file is chosen and avoids null filename on cancel click
            try:
                self.pot_data = np.loadtxt(filename, delimiter=",", skiprows=self.start_row, max_rows=self.max_rows)  # load data
                self.pot_data = self.pot_data + self.position_offset
                if self.normalize: self.pot_data = self.pot_data / self.pot_data.max(axis=0)

                self.set_text_content_description_after_import(self.potentiated_text, self.get_data_description_string(filename, self.pot_data))
                if len(self.pot_data.shape) == 1: # turn a 1D array into a 2D matrix with one column
                    self.pot_data = self.pot_data.reshape(-1, 1)
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

    def export(self, *args):  # used to write an output file
        """
        Action for the export button widget.
        Exports the current spm t signal into a local CSV file.
        """
        if self.is_import_data_null():  # null data check
            return

        try:
            t = spm1d.stats.ttest2(self.pot_data.T, self.baseline_data.T, equal_var=False)
            filename = filedialog.asksaveasfilename(filetypes=[("CSV files", "*.csv")])
            if filename: np.savetxt(filename, t.z)  # null check on filename to avoid cancelled dialog problems
        except Exception as e:
            print("Error performing SPM analysis: " + str(e))
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

    # -----------------------------------------------------------------------------
    # START SPM SIGNAL PROCESSING FUNCTIONS
    # -----------------------------------------------------------------------------

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

    @ staticmethod
    def run_two_sample_test(base_data, pot_data, x_label="Independent Variable", y_label="Dependent Variable"):
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
        ti.plot_threshold_label(fontsize=10, color='black')
        ti.plot_p_values(size=10)  # offsets=[(0, 0.3)]
        ax.set_xlabel(x_label)
        plt.show()

    # -----------------------------------------------------------------------------
    # START GRAPH UPDATE FUNCTIONS
    # -----------------------------------------------------------------------------
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


if __name__ == "__main__":
    interface = MCModulationInterface()  # initial constructor call sets everything up
