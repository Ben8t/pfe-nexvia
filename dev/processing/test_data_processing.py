import unittest

from dev.processing.data_processing import *
from dev.processing.extract import *
import pandas


class DataProcessingTest(unittest.TestCase):

    def test_split_dataset(self):
        data = zip_to_dataframe("data/data-min.zip")
        x_train, y_train, x_test, y_test = split_dataset(data, 0.3, "output-cumulatedCostsBuy_homeAcquisitionCosts_1")
        self.assertEqual(x_train.shape[0], data.shape[0] * 0.7)
        self.assertEqual(x_test.shape[1], data.shape[1] - 1)
        self.assertEqual(type(y_test), pandas.Series)
        self.assertNotIn("output-cumulatedCostsBuy_homeAcquisitionCosts_1", list(x_train.columns))

    def test_cleaning_data(self):
        data = zip_to_dataframe("data/data-min.zip")
        cleaned_data = cleaning_data(data)
        columns = list(cleaned_data.columns)
        output_columns = "output-cumulatedCostsBuy_homeAcquisitionCosts_1"
        self.assertEqual(cleaned_data.isnull().values.any(), 0)
        self.assertIn(output_columns, columns)
        columns_input = list(filter(lambda k: 'input-' in k, columns))
        self.assertEqual(len(columns) - 1, len(columns_input))

    def test_data_processing(self):
        data = zip_to_dataframe("data/data-min.zip")
        cleaned_data = cleaning_data(data)
        data_processed1 = processing_less_feature(cleaned_data)
        columns1 = list(data_processed1.columns)
        self.assertEqual(len(columns1), 6)
        data_processed2 = processing_full(cleaned_data)
        columns2 = list(data_processed2.columns)
        self.assertEqual(len(columns2), 21)
        self.assertNotIn('input-maritalStatus', columns2)
        data_processed3 = processing_label_encoder(cleaned_data)
        columns3 = list(data_processed3.columns)
        self.assertEqual(len(columns3), 19)
        value = data_processed3['input-homeType'].value_counts().index[0]
        self.assertNotIsInstance(value, str)


unittest.main()
