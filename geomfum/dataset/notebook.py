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

        faust_url = "https://raw.githubusercontent.com/JM-data/PyFuncMap/4bde4484c3e93bff925a6a82da29fa79d6862f4b/FAUST_shapes_off/"

        s4a_data_url = "https://raw.githubusercontent.com/gviga/S4A-Scalable-Spectral-Shape-Analysis/refs/heads/main/data/giorgio_/"
        self.files = {
            "cat-00": DownloadableFile("cat-00.off", f"{pyfm_data_url}/cat-00.off"),
            "lion-00": DownloadableFile("lion-00.off", f"{pyfm_data_url}/lion-00.off"),
            "faust-00": DownloadableFile("faust-00.off", f"{faust_url}/tr_reg_000.off"),
            "faust-04": DownloadableFile("faust-04.off", f"{faust_url}/tr_reg_004.off"),
            "brain_template": DownloadableFile(
                "brain_template.off", f"{s4a_data_url}/MNI_remesh.off"
            ),
            "brain_subject": DownloadableFile(
                "brain_subject.off", f"{s4a_data_url}/remeshed/000000_tumoredbrain.off"
            ),
            "template_labels": DownloadableFile(
                "template_labels.txt", f"{s4a_data_url}/remeshed_labels.txt"
            ),
            "template_ldmk": DownloadableFile(
                "template_ldmk.txt", f"{s4a_data_url}/remeshed_landmarks.txt"
            ),
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
