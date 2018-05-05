# extract.py
import json
import csv
import pandas
from zipfile import ZipFile


def extract_zip(input_zip):
    """
    Load a zip file in memory
    """
    input_zip = ZipFile(input_zip)
    return {name: input_zip.read(name) for name in input_zip.namelist()}


def zip_to_csv(data_zip_path, new_file):
    """
    Get a zip file containing json files and return a unique csv file based on json keys.
    data_zip_path : path to the zip file
    new_file : file name for csv file (example : "data.csv")
    """
    data = extract_zip(data_zip_path)  # load zip file in memory
    json_list = list(filter(lambda k: '.json' in k, list(data.keys())))  # get a list all json files in archive
    columns = list(json.loads(data[json_list[0]].decode('utf-8')).keys())  # get keys from the first json

    # write header to csv file
    with open(new_file, "a") as file:
        write = csv.writer(file)
        write.writerow(columns)

    # append data to the file
    for file in json_list:
        row_data = list(json.loads(data[file].decode('utf-8')).values())
        with open(new_file, "a") as file:
            write = csv.writer(file)
            write.writerow(row_data)


def zip_to_dataframe(zip_file):
    """
    Get a zip file containing json files and return a pandas dataframe based on json keys.
    zip_file : path to the zip file
    """
    data = extract_zip(zip_file)  # load zip file in memory
    json_list = sorted(list(filter(lambda k: '.json' in k, list(data.keys()))))  # get a list all json files in archive
    print(json_list)
    columns = list(json.loads(data[json_list[0]].decode('utf-8')).keys())  # get keys from the first json
    list_dataframe = []
    for item in json_list:  # append each json data into dataframe
        list_dataframe.append(list(json.loads(data[item].decode('utf-8')).values()))
    return pandas.DataFrame(list_dataframe, columns=columns)
