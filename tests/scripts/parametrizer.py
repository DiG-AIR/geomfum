import json
import os

import pytest
from geomstats.test.parametrizers import _exec_notebook, _raise_missing_testing_data


class NotebooksParametrizer(type):
    def __new__(cls, name, bases, attrs):
        def _create_new_test(path, **kwargs):
            def new_test(self):
                return _exec_notebook(path=path)

            return new_test

        testing_data = locals()["attrs"].get("testing_data")
        _raise_missing_testing_data(testing_data)

        paths = testing_data.paths

        for path in paths:
            name = path.split(os.sep)[-1].split(".")[0]

            func_name = f"test_{name}"
            test_func = _create_new_test(path)

            with open(path, "r", encoding="utf8") as file:
                metadata = json.load(file).get("metadata")

            for require in metadata.get("requires", []):
                marker = getattr(pytest.mark, require)
                test_func = marker()(test_func)

            attrs[func_name] = test_func

        return super().__new__(cls, name, bases, attrs)
