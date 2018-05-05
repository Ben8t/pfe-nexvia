# data_processing.py
import pandas
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder


def create_dummies(data, variable_list_to_dummize):
    """
    create dummies from feature list and data
    """
    dataframe_dummies = pandas.get_dummies(data[variable_list_to_dummize])
    return dataframe_dummies


def select_columns(data, columns_choice_list):
    return data[columns_choice_list]


def create_interval(data, columns_cut_size_list):
    """
    columns_cut_size_list : list of tuples (feature to cut, size of discretization)
    """
    new_columns_interval = []
    for column, size in columns_cut_size_list:
        new_columns_interval.append(pandas.cut(data[column], size, labels=False))
    data_interval = pandas.concat(new_columns_interval, axis=1)
    return data_interval


def cleaning_data(data):
    """
    clean dataset and keep all inputs and one output (prediction)
    data : panda dataframe
    """
    input_columns = [column for column in data.columns if 'input' in column]
    output_columns = ["output-cumulatedCostsBuy_homeAcquisitionCosts_1"]
    columns = input_columns + output_columns
    filtered_data = data[columns]
    filtered_data = filtered_data.dropna(axis=0, how='any')
    return filtered_data


def label_encoder(data, columns_choice_list):
    """
    transform label to int
    """
    label_encoder = LabelEncoder()
    result = data.copy()
    for column in columns_choice_list:
        result[column] = label_encoder.fit_transform(result[column])
    return result[columns_choice_list]


def polynomial_features(data, feature_list, dimension):
    """
    create non-linear features
    """
    poly = PolynomialFeatures(dimension, include_bias=False)
    data_poly_features = poly.fit_transform(data[feature_list])
    feature_names = poly.get_feature_names(input_features=feature_list)
    processed_data = pandas.DataFrame(data_poly_features, columns=feature_names)
    return processed_data


def split_dataset(dataframe, split_rate, output_variable):
    """
    Split dataset in 4 parts : x_train, y_train, x_test, y_test.
    Args:
        split_rate : explicit
        output_variable : variable to predict

    Return:
        x_train, y_train : train dataset to fit model.
        x_test : use for predictions
        y_test : use for error calculation
    """
    train_set, test_set = train_test_split(dataframe, test_size=split_rate)
    x_train = train_set.drop(output_variable, axis=1)
    y_train = train_set[output_variable]
    x_test = test_set.drop(output_variable, axis=1)
    y_test = test_set[output_variable]
    return x_train, y_train, x_test, y_test


def processing_full(data):
    """
    realize a dummies
    """
    # Select variables/parameters
    list_dummies = ['input-maritalStatus', 'input-homeType']
    list_select = list(data.drop(list_dummies, axis=1).columns)

    # Pipeline
    subdataframes = [create_dummies(data, list_dummies),
                     select_columns(data, list_select)]

    processed_data = pandas.concat(subdataframes, axis=1)
    return processed_data


def processing_less_feature(data):
    """
    reduce the number of features
    realize a dummies
    """
    list_dummies = ['input-maritalStatus', 'input-homeType']
    list_select = ['input-homePrice', 'output-cumulatedCostsBuy_homeAcquisitionCosts_1']

    # Pipeline
    subdataframes = [create_dummies(data, list_dummies),
                     select_columns(data, list_select)]

    processed_data = pandas.concat(subdataframes, axis=1)
    return processed_data


def processing_label_encoder(data):
    """
    string columns become int columns
    """
    list_change = ['input-maritalStatus', 'input-homeType']
    list_select = list(data.drop(list_change, axis=1).columns)

    # Pipeline
    subdataframes = [label_encoder(data, list_change),
                     select_columns(data, list_select)]

    processed_data = pandas.concat(subdataframes, axis=1)
    return processed_data


def processing_change_interval(data):
    """
    string columns become int columns
    transform 'input-previouslyUsedTaxCredit' in two interval
    """
    list_change = ['input-maritalStatus', 'input-homeType']
    list_split = [('input-previouslyUsedTaxCredit', 2)]
    list_select = list(data.drop(['input-previouslyUsedTaxCredit'] + list_change, axis=1).columns)

    # Pipeline
    subdataframes = [label_encoder(data, list_change),
                     create_interval(data, list_split),
                     select_columns(data, list_select)]

    processed_data = pandas.concat(subdataframes, axis=1)
    return processed_data


def processing_full_polynomial_feature(data):
    """
    realize dummies
    realize polynomial columns
    """
    # Select variables/parameters
    feature_dummies = ['input-maritalStatus', 'input-homeType']
    feature_select = ['output-cumulatedCostsBuy_homeAcquisitionCosts_1']
    feature_polynomial = list(data.drop(feature_dummies + feature_select, axis=1).columns)

    # Pipeline
    subdataframes = [create_dummies(data, feature_dummies),
                     select_columns(data, feature_select),
                     polynomial_features(data, feature_polynomial, 2)]

    processed_data = pandas.concat(subdataframes, axis=1)
    return processed_data
