"""Datasets for notebooks/docs."""

import io
import os

import numpy as np
import pandas as pd
import requests

from ._defaults import DATA_DIR
from ._utils import DownloadableFile


class NotebooksDataset:
    """Dataset to use within notebooks.

    Parameters
    ----------
    data_dir : str
        Directory where to store/access data.
    load_at_startup : bool
        Whether to (down)load files at startup.
    """

    def __init__(self, data_dir=None, load_at_startup=False):
        if data_dir is None:
            data_dir = os.environ.get("GEOMFUM_DATA_DIR", DATA_DIR)

        self.data_dir = data_dir

        pyfm_data_url = "https://raw.githubusercontent.com/RobinMagnet/pyFM/refs/heads/master/examples/data/"

        self.files = {
            "cat-00": DownloadableFile("cat-00.off", f"{pyfm_data_url}/cat-00.off"),
            "lion-00": DownloadableFile("lion-00.off", f"{pyfm_data_url}/lion-00.off"),
        }

        os.makedirs(data_dir, exist_ok=True)

        if load_at_startup:
            self.get_filenames()

    def get_filenames(self):
        """Get filenames after (down)loading.

        Uses cached files if already in the system.

        Returns
        -------
        file_paths : list[str]
            File names including directory.
        """
        return [
            file.get_filename(data_dir=self.data_dir) for file in self.files.values()
        ]

    def get_filename(self, index):
        """Get filename after (down)loading.

        Uses cached file if already in the system.

        Parameters
        ----------
        index : str
            File index in the dataset.

        Returns
        -------
        file_path : str
            File name including directory.
        """
        return self.files[index].get_filename(data_dir=self.data_dir)


class BrainstemDataset(NotebooksDataset):
    """SSA-brainstem dataset (auto-discovered from GitHub)."""

    GITHUB_API_URL = (
        "https://api.github.com/repos/"
        "Franca-exe/SSA-brainstem/contents/data/original"
    )

    def __init__(self, data_dir=None, load_at_startup=False):

        # Initialize base class (sets self.data_dir etc.)
        super().__init__(data_dir=data_dir, load_at_startup=False)

        # Override files dictionary
        self.files = self._build_file_dict()

        if load_at_startup:
            self.get_filenames()

    LABELS_URL = (
        "https://raw.githubusercontent.com/"
        "Franca-exe/SSA-brainstem/main/data/info_subjects.xlsx"
    )

    def _build_file_dict(self):
        """Build DownloadableFile dictionary dynamically."""
        response = requests.get(self.GITHUB_API_URL)
        response.raise_for_status()

        files = {}

        for item in response.json():
            if item["type"] == "file":
                filename = item["name"]
                files[filename] = DownloadableFile(
                    filename,
                    item["download_url"],
                )

        return files

    def get_labels(self, keys=None):
        """Download subject labels from ``info_subjects.xlsx``.

        Parameters
        ----------
        keys : list of str, optional
            Dataset keys (e.g. ``['1001.obj', ...]``).  When provided,
            returns a 1-D int array whose i-th entry is the label for
            ``keys[i]``.  When *None*, returns the raw ``DataFrame``.

        Returns
        -------
        labels : np.ndarray, shape (len(keys),), or pd.DataFrame
        """
        response = requests.get(self.LABELS_URL)
        response.raise_for_status()
        df = pd.read_excel(io.BytesIO(response.content))

        if keys is None:
            return df

        # Locate the subject-ID and label columns case-insensitively.
        cols_lower = {c.lower().strip(): c for c in df.columns}
        id_col = next(
            (cols_lower[c] for c in cols_lower if "subject" in c or c == "id"),
            None,
        )
        lbl_col = next(
            (cols_lower[c] for c in cols_lower if "label" in c or "pathology" in c),
            None,
        )
        if id_col is None or lbl_col is None:
            raise KeyError(
                f"Could not find subject-ID / label columns in {list(df.columns)}. "
                "Call get_labels() (no args) to inspect the raw DataFrame."
            )

        label_map = {
            str(sid).strip(): int(lbl)
            for sid, lbl in zip(df[id_col], df[lbl_col])
        }
        return np.array(
            [label_map[k.replace(".obj", "")] for k in keys], dtype=int
        )

