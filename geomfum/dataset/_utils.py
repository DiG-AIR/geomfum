"""Dataset Utilities Module. This module contains utility classes for dataset management."""

import logging
import os
from urllib.request import urlretrieve


class DownloadableFile:
    """(Down)loadable file.

    Parameters
    ----------
    name : str
        File name (without directory).
    url : str
        Url for file download.
    """

    def __init__(self, name, url):
        self.name = name
        self.url = url

    def get_filename(self, data_dir):
        """Get filename after (down)loading.

        Uses cached file if already in the system.

        Parameters
        ----------
        data_dir : str
            Directory where to store/access data.

        Returns
        -------
        file_path : str
            File name including directory.
        """
        file_path = os.path.join(data_dir, self.name)

        if os.path.exists(file_path):
            logging.info(
                f"Data has already been downloaded... using cached file ('{file_path}')."
            )
        else:
            logging.info(f"Downloading '{file_path}' from {self.url} to '{data_dir}'.")
            urlretrieve(self.url, file_path)

        return file_path
