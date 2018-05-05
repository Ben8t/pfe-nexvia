# load_data.py

from dev.processing.extract import zip_to_dataframe

# Import and process data


def load_data(path="data/data.zip"):
    data = zip_to_dataframe(path)
    return data


def load_data_shuffle(path="data/data.zip", line_number=20000):
    data = zip_to_dataframe(path)
    data_shuffle_reduce = data.sample(line_number)
    return data_shuffle_reduce
