"""Unit tests for the notebooks."""

import glob

from geomstats.test.test_case import TestCase

from .parametrizer import NotebooksParametrizer


class NotebooksTestData:
    def __init__(self):
        NOTEBOOKS_DIR = "notebooks"
        self.paths = sorted(glob.glob(f"{NOTEBOOKS_DIR}/**/*.ipynb", recursive=True))


class TestNotebooks(TestCase, metaclass=NotebooksParametrizer):
    testing_data = NotebooksTestData()
