__all__ = ["csv_col_save"]

import csv

def csv_col_save(fileName, cols, colHeader):
    __save_to_csv_col(cols, fileName, colHeader)

def __save_to_csv_row(rows, fileName):
    """

    :param rows:
        list of Dictionaries where each item is a dictionary mapping the
        fidld name to the value.
    :param fileName:
        name of the csv file.
    :return:
        none
    """
    __just_save_it(rows, fileName)

def __save_to_csv_col(cols, fileName, colHeader):
    """

    :param cols:
        list of cols for the csv, all with the same length.
    :param fileName:
        the name of the csv file.
    :param colHeader:
        A list, containing all the column names, the same order as the list of cols.
    :return:
        none
    """
    CsvRowsDic = [{} for I in range(max(len(Col) for Col in cols))]
    for FieldName, Col in zip(colHeader, cols):
        for RowIdx, Value in enumerate(Col):
            CsvRowsDic[RowIdx][FieldName] = Value
    __just_save_it(CsvRowsDic, fileName)

def __just_save_it(csvDicList, fileName):
    with open(fileName, "w") as CSVDataFile:
        Writer = csv.DictWriter(CSVDataFile, fieldnames=csvDicList[0].keys())
        Writer.writeheader()
        Writer.writerows(csvDicList)
    pass