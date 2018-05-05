"""
test_extract.py
"""

import unittest

from dev.processing.extract import *
import pandas
import os


class ExtractTest(unittest.TestCase):

    def test_extract_zip(self):
        data = extract_zip("data/data-min.zip")
        self.assertIsInstance(data, dict)

    def test_zip_to_csv(self):
        zip_to_csv("data/data-min.zip", "data/data_tmp.csv")
        is_exist = os.path.exists("data/data_tmp.csv")
        is_file = os.path.isfile("data/data_tmp.csv")
        self.assertTrue(is_exist)
        self.assertTrue(is_file)
        if is_exist:
            os.remove("data/data_tmp.csv")

    def test_zip_to_dataframe(self):
        data = zip_to_dataframe("data/data-min.zip")
        self.assertIsInstance(data, pandas.DataFrame)


unittest.main()
