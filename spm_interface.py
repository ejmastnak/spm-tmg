from pathlib import Path
import os
import traceback
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import numpy as np

import analysis
import data_processing
import plotting


class SPMInterface:

    def __init__(self, base_filename=None, pot_filename=None):
        """
        For now includes filename parameters as a hacky way to load data automatically on startup for testing
        :param base_filename:
        :param pot_filename:
        """
        self.BASE_POT = "BASE_POT"
        self.BASE_ATRO = "BASE_ATRO"
        self.mode = self.BASE_POT  # "BASE_POT" or "BASE_INJ" for comparing baseline to either potentiated or injured

        self.baseline_filename = ""  # name of file holding baseline data
        self.active_filename = ""  # name of file holding potentiated data
        self.baseline_data = np.zeros(shape=(0, 0))  # e.g. 1000 x 10
        self.active_data = np.zeros(shape=(0, 0))  # e.g. 1000 x 10
        self.new_data = True

        # START DATA PROCESSING PARAMETERS
        self.start_row = 1  # csv data file row at which to begin reading data (0-indexed)
        self.max_rows = 100  # number of rows to read after start_row is reached
        self.position_offset = 0.0
        self.normalize = False
        # END DATA PROCESSING PARAMETERS

        # START TKINTER WIDGETS
        self.root_window = tk.Tk()
        self.root_window.title("SPM 1D Analysis Interface")

        # create frames here
        self.rootframe = ttk.Frame(self.root_window, padding=(3, 12, 3, 12))
        self.textarea_frame = ttk.Frame(self.rootframe)  # panel to hold text areas
        self.controlframe = ttk.Frame(self.rootframe)  # panel to hold controls

        # Baseline data window
        self.baseline_label = ttk.Label(self.textarea_frame, text="Baseline Data")
        self.baseline_scroll = ttk.Scrollbar(self.textarea_frame)
        self.baseline_text_area = tk.Text(self.textarea_frame, height=5, width=52)
        self.baseline_scroll.config(command=self.baseline_text_area.yview)
        self.baseline_text_area.config(yscrollcommand=self.baseline_scroll.set)
        self.baseline_text_area.insert(tk.END, "No data imported")
        self.baseline_text_area.configure(state='disabled')

        # Active data window
        self.active_label = ttk.Label(self.textarea_frame, text="{} Data".format(self.get_mode_name()))
        self.active_scroll = ttk.Scrollbar(self.textarea_frame)
        self.active_text_area = tk.Text(self.textarea_frame, height=5, width=52)
        self.active_scroll.config(command=self.active_text_area.yview)
        self.active_text_area.config(yscrollcommand=self.active_scroll.set)
        self.active_text_area.insert(tk.END, "No data imported")
        self.active_text_area.configure(state='disabled')

        # SPM analysis results window
        self.spm_label = ttk.Label(self.textarea_frame, text="SPM Analysis Results")
        self.spm_scroll = ttk.Scrollbar(self.textarea_frame)
        self.spm_text_area = tk.Text(self.textarea_frame, height=7, width=52)
        self.spm_scroll.config(command=self.spm_text_area.yview)
        self.spm_text_area.config(yscrollcommand=self.spm_scroll.set)
        self.spm_text_area.insert(tk.END, "No analysis results")
        self.spm_text_area.configure(state='disabled')

        # create control widgets
        self.modebox_var = tk.StringVar()
        self.mode_combobox = ttk.Combobox(self.controlframe, state='readonly', textvariable=self.modebox_var)
        self.mode_combobox['values'] = ('Baseline-Potentiated Mode', "Baseline-Atrophied Mode")
        if self.mode == self.BASE_POT: self.mode_combobox.current(0)
        else: self.mode_combobox.current(1)
        self.mode_combobox.bind("<<ComboboxSelected>>", self.change_mode)

        self.import_active_button = ttk.Button(self.controlframe, text="Import {} Data".format(self.get_mode_name()), command=self.import_active_data)
        self.import_baseline_button = ttk.Button(self.controlframe, text="Import Baseline Data", command=self.import_baseline_data)
        self.compare_button = ttk.Button(self.controlframe, text="Compare", command=self.compare)
        self.export_curve_button = ttk.Button(self.controlframe, text="Export t Curve", command=self.export_tcurve)
        self.export_params_button = ttk.Button(self.controlframe, text="Export t Parameters", command=self.export_tparams)
        self.close_button = ttk.Button(self.controlframe, text="Exit", command=self.close)

        # gridding subframes
        self.rootframe.grid(column=0, row=0, sticky=(tk.N, tk.S, tk.E, tk.W))  # place the root frame

        self.textarea_frame.grid(column=0, row=0, sticky=(tk.N, tk.S, tk.E, tk.W))  # place textarea frame
        self.controlframe.grid(column=0, row=1, sticky=(tk.N, tk.S, tk.E, tk.W))  # place control frame

        # gridding text area frame
        self.baseline_label.grid(column=0, row=1)
        self.baseline_scroll.grid(column=1, row=2, sticky=tk.W)
        self.baseline_text_area.grid(column=0, row=2, sticky=tk.W)

        self.active_label.grid(column=0, row=3)
        self.active_scroll.grid(column=1, row=4, sticky=tk.W)
        self.active_text_area.grid(column=0, row=4, sticky=tk.W)

        self.spm_label.grid(column=0, row=5)
        self.spm_scroll.grid(column=1, row=6, sticky=tk.W)
        self.spm_text_area.grid(column=0, row=6, sticky=tk.W)

        # gridding control frame
        self.mode_combobox.grid(column=0, row=0, sticky=tk.W)
        self.import_baseline_button.grid(column=0, row=1, sticky=tk.W)
        self.import_active_button.grid(column=0, row=2, sticky=tk.W)
        self.compare_button.grid(column=0, row=3, sticky=tk.W)
        self.export_curve_button.grid(column=0, row=4, sticky=tk.W)
        self.export_params_button.grid(column=0, row=5, sticky=tk.W)
        self.close_button.grid(column=0, row=6, sticky=tk.W)

        # configure weights
        self.root_window.columnconfigure(0, weight=1)
        self.root_window.rowconfigure(0, weight=1)

        self.rootframe.columnconfigure(0, weight=1)  # control frame
        self.rootframe.rowconfigure(0, weight=1)

        # for loading data programtically (and not via gui) when testing
        if base_filename is not None and pot_filename is not None:
            self.set_baseline_data(base_filename)
            self.set_active_data(pot_filename)

        self.root_window.mainloop()
        # END TKINTER WIDGETS

    @ staticmethod
    def get_data_description(filename, data):
        """
        Used to get a string description of imported baseline/potentiated data
        to display in the GUI, to give the user a description of the file they've imported

        :param filename: full path to a file containing measurement data
        :param data: 2D numpy array holding the data measurement data; rows are data samples and columns are measurement sets
        """

        file_string = "Filename: " + Path(filename).name
        dim_string_1 = "Dimensions: {} rows by {} columns".format(data.shape[0], data.shape[1])
        dim_string_2 = "({} measurements of {} points per measurement)".format(data.shape[1], data.shape[0])
        path_string = "Location: " + os.path.dirname(filename)
        return file_string + "\n" + dim_string_1 + "\n" + dim_string_2 + "\n" + path_string

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
        text_widget.delete('1.0', tk.END)
        text_widget.insert(tk.END, text_content)
        text_widget.configure(state='disabled')

    def get_mode_name(self):
        """
        Returns either Potentiated or Atrophied based on the current working mode
        Used to get dynamically adaptive labels for various GUI elements
        """
        if self.mode == self.BASE_POT:
            return "Potentiated"
        elif self.mode == self.BASE_ATRO:
            return "Antrophied"
        else:  # should never happen
            print("Error: Unidentified mode: {}".format(self.mode))
            return "Potentiated"

    # -----------------------------------------------------------------------------
    # START GUI WIDGET ACTION FUNCTIONS
    # -----------------------------------------------------------------------------
    def change_mode(self, event):
        """
        Changes between "Potentiated" and "Atrophied" GUI modes based on the inputed
        :param event: the tkinter combobox event triggering the change mode action.
         Needed in the function declaration even though it is not explicitly used
        """
        if "Potentiated" in self.modebox_var.get():
            if self.mode == self.BASE_POT:  # if already in BASE-POT mode
                return  # exit and avoid unnecessary updates
            else:
                self.mode = self.BASE_POT
                self.update_mode_labels()
        else:  # "Atrophied" in self.modebox_var.get():
            if self.mode == self.BASE_ATRO:  # if already in BASE_INJ-POT mode
                return  # exit and avoid unnecessary updates
            else:
                self.mode = self.BASE_ATRO
                self.update_mode_labels()

    def update_mode_labels(self):
        """ Changes labels of various GUI components to reflect Potentiated or Atrophied mode """
        self.import_active_button['text'] = "Import {} Data".format(self.get_mode_name())
        self.active_label['text'] = "Import {} Data".format(self.get_mode_name())

    def import_baseline_data(self):
        """
        Implements the protocol for importing baseline measurement data.
        This method is set as the action for the import_baseline_button widget.
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

    def import_active_data(self):
        """
        Implements the protocol for importing "active" measurement data, which may be either
         potentiated or atrophied depending on the current GUI mode
        This method is set as the action for the import_active_button widget.
        """
        filename = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])  # get csv files
        if filename:  # only called if a file is chosen and avoids null filename on cancel click
            try:
                self.active_data = np.loadtxt(filename, delimiter=",", skiprows=self.start_row, max_rows=self.max_rows)  # load data
                self.process_active_data()  # process imported data
                self.set_imported_data_description(self.active_text_area, self.get_data_description(filename, self.active_data))
                self.active_filename = filename  # if import is successful, set potentiated filename

            except Exception as e:
                print("Error importing data: " + str(e))
                return

    def compare(self):
        """
        Action for the "Compare" button. Runs an SPM analysis between the imported baseline and active data.
        """
        if self.is_import_data_null():  # null data check
            return
        self.reshape_data()  # reshape input data as necessary
        self.plot_test_results()

    def plot_test_results(self):
        """
        Plots the baseline and active data's mean and standard deviation clouds on axis one
        Plots the SPM t-test between the baseline and active data on axis two
        """
        try:
            t, ti = analysis.get_spm_ti(self.baseline_data, self.active_data)  # try getting t an ti objects
        except Exception as e:
            print("Error performing SPM analysis: " + str(e))
            return

        self.set_imported_data_description(self.spm_text_area, analysis.get_spm_ti_string_description(ti, time_offset=self.get_time_offset()))
        plotting.plot_test_results(t, ti, self.baseline_data, self.active_data, time_offset=self.get_time_offset(),
                                   figure_output_path=Path(self.active_filename).name.replace(".csv", ".png"),
                                   mode_name=self.get_mode_name(), mode=self.mode)

    def export_tcurve(self, *args):  # used to write an output file
        """
        Action for the export t-curve button widget.
        Export the t-statistic as a 2D array to a local CSV file.
        First column is time, second is t-statistic
        """
        if self.is_import_data_null():  # null data check
            return

        try:
            t = analysis.get_spm_t(self.baseline_data, self.active_data)  # try getting t object
        except Exception as e:
            print("Error performing SPM analysis: " + str(e))
            return

        analysis.export_t_curve(self.baseline_data, self.active_data, self.get_time_offset())

    def export_tparams(self, *args):  # used to write an output file
        """
        Action for the export parameters button widget.
        Exports various ti object parameters to a local csv file in tabular format
        """
        if self.is_import_data_null():  # null data check
            return

        try:
            _, ti = analysis.get_spm_ti(self.baseline_data, self.active_data)  # try getting t an ti objects
        except Exception as e:
            print("Error performing SPM analysis: " + str(e))
            return

        analysis.export_ti_parameters(ti, time_offset=self.get_time_offset(),
                                      baseline_filename=self.baseline_filename, active_filename=self.active_filename,
                                      mode_name=self.get_mode_name())

    def close(self):
        """
        Action for the close button widget.
        Closes the current SPM window and exits the program
        """
        self.root_window.destroy()
        tk.sys.exit()
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

    def set_active_data(self, active_filename):
        """
        Action to programatically set active data
        Implements the protocol for importing active data.
        """
        if active_filename:  # only called if a file is chosen and avoids null filename on cancel click
            try:
                self.active_data = np.loadtxt(active_filename, delimiter=",", skiprows=self.start_row, max_rows=self.max_rows)  # load data
                self.process_active_data()  # process imported data
                self.set_imported_data_description(self.active_text_area, self.get_data_description(active_filename, self.active_data))
                self.active_filename = active_filename  # if import is successful, set active filename

            except Exception as e:
                print("Error importing active data: " + str(e))
                traceback.print_tb(e.__traceback__)
                return

    def is_import_data_null(self):
        """
        Used as a null check for imported data
        Returns true (data is null) if there are either no rows or
         no columns in either of baseline and active data arrays.

        e.g. base_data.shape = (0, 100) and active_data.shape = (10, 100) returns true (data is null)
        """
        base_shape = self.baseline_data.shape
        active_shape = self.active_data.shape

        base_rows = base_shape[0]
        base_cols = base_shape[1]
        active_rows = active_shape[0]
        active_cols = active_shape[1]

        if base_rows*base_cols*active_rows*active_cols == 0: # quick way to check if any of the values are zero
            return True
        else:
            return False

    def reshape_data(self):
        """ Reshapes imported measurement data, if necessary, into a format compatible with spm1d """

        base_shape = self.baseline_data.shape
        active_shape = self.active_data.shape

        base_rows = base_shape[0]
        base_cols = base_shape[1]
        active_rows = active_shape[0]
        active_cols = active_shape[1]

        if base_rows != active_rows and base_cols == active_cols:  # same number columns, different number of rows
            self.baseline_data, self.active_data = data_processing.match_rows(self.baseline_data, self.active_data, base_rows, active_rows)

        elif base_rows == active_rows and base_cols != active_cols:  # same number rows, different number columns
            self.baseline_data, self.active_data = data_processing.match_cols(self.baseline_data, self.active_data, base_rows, base_cols, active_rows, active_cols)

        elif base_shape != active_shape:  # different number of rows AND different number of columns
            self.baseline_data, self.active_data = data_processing.match_rows(self.baseline_data, self.active_data, base_rows, active_rows)
            self.baseline_data, self.active_data = data_processing.match_cols(self.baseline_data, self.active_data, base_rows, base_cols, active_rows, active_cols)

        if base_cols == 1 and active_cols == 1:
            self.baseline_data, self.active_data = data_processing.increase_cols(self.baseline_data, self.active_data, base_rows, base_cols, active_rows, active_cols)

    # -----------------------------------------------------------------------------
    # START SPM SIGNAL PROCESSING FUNCTIONS
    # -----------------------------------------------------------------------------
    # TODO move to data_processing
    def process_baseline_data(self):
        if self.baseline_data is None:  # null check
            return

        if self.normalize: self.baseline_data = self.baseline_data / self.baseline_data.max(axis=0)  # normalize
        self.baseline_data = self.baseline_data + self.position_offset  # add vertical offset

        # if data is a single column (1D array) reshape into a 2D array (matrix with one column)
        if len(self.baseline_data.shape) == 1:
            self.baseline_data = self.baseline_data.reshape(-1, 1)

        if self.active_data is not None:  # if active data has been set
            # fix potential false SPM significance region problems
            self.baseline_data, self.active_data = data_processing.fix_false_spm_significance(self.baseline_data, self.active_data, mode=self.mode)

    def process_active_data(self):
        if self.active_data is None:  # null check
            return

        if self.normalize: self.active_data = self.active_data / self.active_data.max(axis=0)  # normalize
        self.active_data = self.active_data + self.position_offset  # add vertical offset

        # if data is a single column (1D array) reshape into a 2D array (matrix with one column)
        if len(self.active_data.shape) == 1:
            self.active_data = self.active_data.reshape(-1, 1)

        if self.baseline_data is not None:  # if baseline data has been set
            # fix potential false SPM significance region problems
            self.baseline_data, self.active_data = data_processing.fix_false_spm_significance(self.baseline_data, self.active_data, mode=self.mode)

    def get_time_offset(self):
        """
        Corrects for time offset from skipping the first row of the data files, which contain
        zero displacement, to avoid singularities in the SPM t-statistic. Because of this skip
        all time is offset proportionally to the number of rows skipped.

        This function assumes the standard 1kHz TMG sample rate, so each row is one millisecond
        """
        return self.start_row
    # -----------------------------------------------------------------------------
    # END SPM SIGNAL PROCESSING FUNCTIONS
    # -----------------------------------------------------------------------------


def gui_launch():
    interface = SPMInterface()


def development_launch():
    # load data programatically for development use
    data_dir = "/Users/ejmastnak/Documents/Dropbox/projects-and-products/tmg-bmc/spm/spm-measurements/spm_1_9_2020/sd/"
    base_filename = data_dir + "sd_base.csv"
    pot_filename = data_dir + "sd_pot.csv"
    interface = SPMInterface(base_filename=base_filename, pot_filename=pot_filename)


if __name__ == "__main__":
    gui_launch()
    # development_launch()
    # practice()
