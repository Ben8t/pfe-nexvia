import unittest

from dev.processing.data_processing import *
from dev.processing.extract import *
import pandas
from flake8.api import legacy as flake8
import os


class StyleTest(unittest.TestCase):

    def test_style(self):
        test_file_list = []
        for root, dirs, files in os.walk("dev/"):
            for file in files:
                if file.endswith(".py") and file != "__init__.py" and file != "workshop.py":
                    test_file_list.append(os.path.join(root, file))
        for file in test_file_list:
            style_guide = flake8.get_style_guide(ignore=['E24', 'W503', 'E501', 'F401', 'F403', 'F405', 'F841'])
            report = style_guide.input_file(file)
            self.assertEqual(report.total_errors, 0)


unittest.main()
