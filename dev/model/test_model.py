"""
test_model.py
"""

import unittest
from dev.processing.data_processing import *
from dev.processing.extract import *
from dev.processing.load_data import load_data
from dev.model.model import *
import pandas
import numpy
import os


class ModelTest(unittest.TestCase):

    def test_DoubleHomeType_fit(self):
        data = load_data("data/data-min.zip")
        cleaned_data = cleaning_data(data)
        processed_data = processing_full(cleaned_data)
        x_train, y_train, x_test, y_test = split_dataset(processed_data, 0.3, "output-cumulatedCostsBuy_homeAcquisitionCosts_1")
        model = DoubleHomeType(model_regression)
        model.fit(x_train, y_train)
        self.assertIsInstance(model.model_new, LinearRegression)
        self.assertIsInstance(model.model_existing, LinearRegression)

    def test_DoubleHomeType_predict(self):
        data = load_data("data/data-min.zip")
        cleaned_data = cleaning_data(data)
        processed_data = processing_full(cleaned_data)
        x_train, y_train, x_test, y_test = split_dataset(processed_data, 0.3, "output-cumulatedCostsBuy_homeAcquisitionCosts_1")
        model = DoubleHomeType(model_regression)
        model.fit(x_train, y_train)
        prediction = model.predict(x_test)
        self.assertIsInstance(prediction, pandas.Series)

    def test_DiscretRegressor_fit(self):
        data = load_data("data/data-min.zip")
        cleaned_data = cleaning_data(data)
        processed_data = processing_label_encoder(cleaned_data)
        x_train, y_train, x_test, y_test = split_dataset(processed_data, 0.3, "output-cumulatedCostsBuy_homeAcquisitionCosts_1")
        model = DiscretRegressor(model_regression, ["input-homeType"])
        model.fit(x_train, y_train)
        self.assertIsInstance(model.list_model_valeur[0][0], LinearRegression)
        self.assertEqual(len(model.list_discret_variable), len(model.list_model_valeur[0][1]))

    def test_DiscretRegressor_predict(self):
        data = load_data("data/data-min.zip")
        cleaned_data = cleaning_data(data)
        processed_data = processing_label_encoder(cleaned_data)
        x_train, y_train, x_test, y_test = split_dataset(processed_data, 0.3, "output-cumulatedCostsBuy_homeAcquisitionCosts_1")
        model = DiscretRegressor(model_regression, ["input-homeType"])
        model.fit(x_train, y_train)
        prediction = model.predict(x_test)
        self.assertIsInstance(prediction, list)


unittest.main()
