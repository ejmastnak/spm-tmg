import pandas as pd
import os

data_dir = "/Users/ejmastnak/Documents/Media/tmg-bmc-media/measurements/"


def try_next_output_dir(base_dir_name):
    try:
        os.mkdir(base_dir_name)
        return True
    except OSError as error:
        return False


def make_output_dir(base_dir_name):

    try:  # try directory named "converted"
        os.mkdir(base_dir_name + "/")
        return base_dir_name + "/"

    except OSError as error:  # directoy exists
        success = False
        counter = 0
        while not success:
            counter += 1
            if counter > 99:
                print("Aborting. Maximum output directory count exceeded.")
                return
            success = try_next_output_dir(base_dir_name + "_" + str(counter) + "/")

        return base_dir_name + "_" + str(counter) + "/"


def convert_directory(directory, conversion_mode, base_reps=1, pot_reps=1):
    """
    Input the path to a directory containing TMG-format excel files
    Converts all ".xlsx" files in the specified directory to csv files
    """
    output_dir = make_output_dir(directory + "converted")

    for filename in sorted(os.listdir(directory)):
        if ".xlsx" in filename and "$" not in filename:
            print(filename)
            if conversion_mode == EXCEL_TO_CSV:
                excel_to_csv(directory, filename, output_dir)
            elif conversion_mode == EXCEL_TO_CSV_BASE_POT:
                excel_to_csv_base_pot(directory, filename, output_dir, base_reps=base_reps, pot_reps=pot_reps)
            elif conversion_mode == EXCEL_TO_CSV_BASE_POT_SET_FILES:
                excel_to_csv_base_pot_set_files(directory, filename, output_dir, base_reps=base_reps, pot_reps=pot_reps)
            else:
                print("Aborting. Invalid conversion mode.")
                return


def excel_to_csv(parent_dir, filename, output_dir):
    """
    Converts a single TMG excel file to an equivalent csv file containing measurements in each column
    :param parent_dir: parent directory containing the xlsx file
    :param filename: the name of the xlsx file, including extension
    :param output_dir: output directory for the converted CSV file
    """
    file_path = parent_dir + filename

    # read excel and drop first column, which is empty in TMG format
    df = pd.read_excel(file_path, engine='openpyxl', header=None, skiprows=DATA_START_ROW).drop(columns=[0])
    df.to_csv(output_dir + filename.replace(".xlsx", ".csv"), header=False, index=False)


def excel_to_csv_base_pot(parent_dir, filename, output_dir, base_reps=1, pot_reps=1):
    """
    Converts a SPM-protocol TMG excel spreadsheet into a csv file,
        with baseline and potentiated measurements in separate files

    Input: path to an TMG-format Excel file, wihth columns in the format
        base_reps baseline measurements
        pot_reps  potentiated measurements
        base_reps baseline measurements
        pot_reps  potentiated measurements
        ... etc...
        base_reps baseline measurements
        pot_reps  potentiated measurements  # end with potentiated measurement

    Output: Two CSV files:
     Baseline file containing all baseline measurements in the order they appear in the Excel file
     Potentiated file containing all baseline measurements in the order they appear in the Excel file

    :param parent_dir: parent directory containing the xlsx file
    :param filename: the name of the xlsx file, including extension
    :param output_dir: output directory for the converted CSV file
    :param base_reps: number of measurements per baseline state
    :param pot_reps: number of measurements per potentiated state
    """
    path = parent_dir + filename

    # read excel and drop first column, which is empty in TMG format
    df = pd.read_excel(path, engine='openpyxl', header=None, skiprows=DATA_START_ROW).drop(columns=[0])

    n_cols = df.shape[1]
    reps_per_set = base_reps + pot_reps
    sets = n_cols/reps_per_set

    if not sets.is_integer():
        print("Error: Inputed repetition values lead to a non-integer number of sets.")
        print("Aborting")
        return
    else:
        sets = int(sets)

    base_cols = []  # construct column numbers of all baseline measurements in the entire Excel file
    pot_cols = []  # construct column numbers of all baseline measurements in the entire Excel file
    for s in range(sets):
        for r in range(base_reps):
            base_cols.append(s * reps_per_set + r + 1)
        for r in range(pot_reps):
            pot_cols.append(s * reps_per_set + r + 1 + base_reps)

    print("Baseline column numbers: ", end="")
    print(base_cols)
    print("Potentiated column numbers: ", end="")
    print(pot_cols)

    df.to_csv(output_dir + filename.replace(".xlsx", "-base.csv"), header=False, index=False, columns=base_cols)
    df.to_csv(output_dir + filename.replace(".xlsx", "-pot.csv"), header=False, index=False, columns=pot_cols)


def excel_to_csv_base_pot_set_files(parent_dir, filename, output_dir, base_reps=1, pot_reps=1):
    """
    Input: path to an TMG-format Excel file, wihth columns in the format
        base_reps baseline measurements
        pot_reps  potentiated measurements
        base_reps baseline measurements
        pot_reps  potentiated measurements
        ... etc...
        base_reps baseline measurements
        pot_reps  potentiated measurements  # end with potentiated measurement

    One group of base_reps + pot_reps measurements constitutes one set

    Output: Many CSV files (2x the number of sets in the inputed Excel file)
     One baseline CSV file for each measurement set
     One potentiate CSV file for each measurement set

    :param parent_dir: parent directory containing the xlsx file  e.g. "Users/user/tmg-data/measurements/"
    :param filename: the name of the xlsx file, including extension e.g. "EM1234.xlsx"
    :param output_dir: the parent output directory for the folders containing the converted CSV files
    :param base_reps: number of measurements per each baseline state
    :param pot_reps: number of measurements per each potentiated state
    """
    path = parent_dir + filename

    # read excel and drop first column, which is empty in TMG format
    df = pd.read_excel(path, engine='openpyxl', header=None, skiprows=DATA_START_ROW).drop(columns=[0])

    n_cols = df.shape[1]  # number of total measurements in Excel file
    reps_per_set = base_reps + pot_reps
    sets = n_cols/reps_per_set  # compute number of measurement sets in Excel file

    if not sets.is_integer():
        print("Error: Inputed repetition values lead to a non-integer number of sets.")
        print("Aborting")
        return
    else:
        sets = int(sets)

    output_dir = make_output_dir(output_dir + filename.replace(".xlsx", ""))  # create an output directory for each file
    for s in range(sets):
        print("Set Number: {}".format(s))
        base_cols = []  # construct column numbers of baseline measurements in the current set
        pot_cols = []  # construct column numbers of all baseline measurements in the current set
        for r in range(base_reps):
            base_cols.append(s * reps_per_set + r + 1)
        for r in range(pot_reps):
            pot_cols.append(s * reps_per_set + r + 1 + base_reps)

        print("Baseline column numbers: ", end="")
        print(base_cols)
        print("Potentiated column numbers: ", end="")
        print(pot_cols)
        print()

        df.to_csv(output_dir + filename.replace(".xlsx", "-base-{}.csv".format(s + 1)), header=False, index=False, columns=base_cols)
        df.to_csv(output_dir + filename.replace(".xlsx", "-pot-{}.csv".format(s + 1)), header=False, index=False, columns=pot_cols)


def conversion_wrapper():
    """ Wrapper method for running a conversion """
    # folder = "klanec-17-03-2021/"
    convert_dir = "/Users/ejmastnak/Documents/Media/tmg-bmc-media/measurements/potenciacija-19-03-2021/NF-squat/"
    base_reps = 8
    pot_reps = 8
    convert_directory(convert_dir, EXCEL_TO_CSV_BASE_POT, base_reps=base_reps, pot_reps=pot_reps)
    # convert_directory(convert_dir, EXCEL_TO_CSV_BASE_POT_SET_FILES, base_reps=base_reps, pot_reps=pot_reps)


DATA_START_ROW = 24
DATA_START_COL = "B"
# DATA_END_COL = "Q"
DATA_END_COL = "DY"

EXCEL_TO_CSV = 1  # take the measurements from a TMG excel file and put them verbatim in a CSV file. Do not separate baseline or potentiated
EXCEL_TO_CSV_BASE_POT = 2  # save all baseline measurements in a TMG excel file in one CSV file and all potentiated measurements in a separate CSV file.
EXCEL_TO_CSV_BASE_POT_SET_FILES = 3  # create separate CSV files for the baseline and potentiated measurements in each set of measurement in the TMG excel file


if __name__ == "__main__":
    # read_tmg_excel()
    conversion_wrapper()
    # test_cols()

# TODO: customizable data end column "DY"
