"""Datasets for notebooks/docs."""

import os

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
