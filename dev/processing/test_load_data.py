"""
test_load_data.py
"""

import unittest
from dev.processing.extract import zip_to_dataframe
from dev.processing.load_data import load_data_shuffle
import pandas


class LoadDataTest(unittest.TestCase):

    def test_load_data_shuffle(self):
        data = load_data_shuffle("data/data-min.zip", 20)
        self.assertEqual(data.shape[0], 20)


unittest.main()
