import pandas as pd

data_dir = "/Users/ejmastnak/Documents/Media/tmg-bmc-media/measurements/klanec-12-03-2021/"

DATA_START_ROW = 24
DATA_START_COL = "B"
DATA_END_COL = "Q"


def read_tmg_excel():
    """
    Read a TMG formatted excel spreadsheet of a TMG measurement session
    """
    file = "EM20210312111435.xlsx"
    path = data_dir + "EM/" + file
    df = pd.read_excel(path, engine='openpyxl', header=None, skiprows=DATA_START_ROW,
                       usecols="{}:{}".format(DATA_START_COL, DATA_END_COL))
    df.to_csv(data_dir + "EM/test.csv", header=False, index=False)


def excel_to_base_pot_csv():
    """
    Read a TMG formatted excel spreadsheet of a TMG measurement session
    Convert to base/pot csv
    """
    file = "EM20210312111435.xlsx"
    path = data_dir + "EM/" + file
    df = pd.read_excel(path, engine='openpyxl', header=None, skiprows=DATA_START_ROW,
                       usecols="{}:{}".format(DATA_START_COL, DATA_END_COL))
    df.to_csv(data_dir + "EM/test-base.csv", header=False, index=False, columns=df.columns[::2].tolist())
    df.to_csv(data_dir + "EM/test-pot.csv", header=False, index=False, columns=df.columns[1::2].tolist())


if __name__ == "__main__":
    # read_tmg_excel()
    excel_to_base_pot_csv()
