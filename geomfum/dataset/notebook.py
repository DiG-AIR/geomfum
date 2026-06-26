"""Datasets for notebooks/docs."""

import io
import os
import zipfile

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


class AfrotheriaBonyLabyrinthDataset(NotebooksDataset):
    """Mammalian bony labyrinth dataset (Grunstra et al., 2024).

    OSF:   https://osf.io/9mtwh/
    Paper: https://doi.org/10.1038/s41467-024-52180-1

    Contains PLY surface meshes of the bony labyrinth (inner ear) for 40
    mammal specimens — 20 Afrotheria and 20 outgroups — extracted from
    microCT scans.  Ecological metadata (habitat group, locomotion,
    body mass) is also available.
    """

    ZIP_URL            = "https://osf.io/download/57c9b/"
    SAMPLE_CSV_URL     = "https://osf.io/download/ugjfn/"
    CONTEXTUAL_CSV_URL = "https://osf.io/download/48cg9/"
    RAW_LMK_CSV_URL    = "https://osf.io/download/9bw3d/"
    SLID_LMK_CSV_URL   = "https://osf.io/download/ka9jm/"
    _ZIP_NAME = "Afrotheria_3D-surface.zip"

    def __init__(self, data_dir=None, load_at_startup=False):
        super().__init__(data_dir=data_dir, load_at_startup=False)
        self.files = {}
        self._ply_dir = None

        if load_at_startup:
            self._ensure_extracted()

    # ── Download / extraction ─────────────────────────────────────────────

    def _ensure_extracted(self):
        """Download and extract the surface zip if not already done."""
        for dirpath, _, filenames in os.walk(self.data_dir):
            if any(f.endswith(".ply") for f in filenames):
                self._ply_dir = dirpath
                self._build_file_dict()
                return

        zip_path = os.path.join(self.data_dir, self._ZIP_NAME)
        if not os.path.exists(zip_path):
            print("Downloading bony labyrinth meshes (~176 MB)…")
            resp = requests.get(self.ZIP_URL, stream=True)
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            downloaded = 0
            with open(zip_path, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=65536):
                    fh.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        print(
                            f"\r  {downloaded / 1e6:.1f} / {total / 1e6:.1f} MB",
                            end="",
                        )
            print()

        print("Extracting…")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(self.data_dir)

        for dirpath, _, filenames in os.walk(self.data_dir):
            if any(f.endswith(".ply") for f in filenames):
                self._ply_dir = dirpath
                break

        self._build_file_dict()

    def _build_file_dict(self):
        """Map stem → filename for every PLY in the extracted directory."""
        self.files = {
            os.path.splitext(fname)[0]: fname
            for fname in sorted(os.listdir(self._ply_dir))
            if fname.lower().endswith(".ply")
        }

    # ── File access ───────────────────────────────────────────────────────

    def get_filename(self, key):
        """Return full path to the PLY file for a given key."""
        return os.path.join(self._ply_dir, self.files[key])

    def get_filenames(self):
        """Return full paths for all PLY files."""
        return [self.get_filename(k) for k in self.files]

    # ── Metadata ──────────────────────────────────────────────────────────

    def _download_csv(self, url):
        resp = requests.get(url)
        resp.raise_for_status()
        return pd.read_csv(io.StringIO(resp.text))

    def get_sample_df(self):
        """Return specimen metadata (Order, Supraordinal taxon, sex, etc.)."""
        return self._download_csv(self.SAMPLE_CSV_URL)

    def get_contextual_df(self):
        """Return ecological metadata (HabitatGroup, locomotion, body mass)."""
        return self._download_csv(self.CONTEXTUAL_CSV_URL)

    def get_metadata(self, keys=None):
        """Merged specimen + ecological metadata, optionally aligned to ``keys``.

        Parameters
        ----------
        keys : list of str, optional
            PLY stem names (without extension).  When provided the returned
            DataFrame rows are ordered to match ``keys``.

        Returns
        -------
        df : pd.DataFrame
        """
        sample = self.get_sample_df()
        ctx = self.get_contextual_df()

        def _norm(s):
            return str(s).strip().replace(" ", "_").lower()

        sci_col = next(
            c for c in sample.columns
            if "scientific" in c.lower() or "linnean" in c.lower()
        )
        sample = sample.copy()
        sample["_jk"] = sample[sci_col].apply(_norm)

        ctx = ctx.copy()
        ctx["_jk"] = ctx["BinomialName"].apply(_norm)

        merged = sample.merge(ctx, on="_jk", how="left").drop(columns=["_jk"])

        if keys is None:
            return merged

        id_col = next(
            (c for c in sample.columns if "specimen" in c.lower() and "id" in c.lower()),
            None,
        )
        lookup = {}
        for _, row in merged.iterrows():
            if id_col and pd.notna(row.get(id_col)):
                lookup[str(row[id_col]).strip().lower()] = row
            norm_name = _norm(row[sci_col])
            lookup[norm_name] = row

        rows = []
        for k in keys:
            k_low = k.lower()
            if k_low in lookup:
                rows.append(lookup[k_low])
            else:
                match = next(
                    (v for kk, v in lookup.items() if kk in k_low or k_low in kk),
                    None,
                )
                rows.append(match if match is not None else pd.Series(dtype=object))

        return pd.DataFrame(rows).reset_index(drop=True)

    def get_labels(self, keys, column="HabitatGroup"):
        """Label array aligned to ``keys`` for a given metadata column.

        Parameters
        ----------
        keys : list of str
            PLY stem names.
        column : str
            Column from the merged metadata DataFrame.  Common choices:
            ``"Order"``, ``"HabitatGroup"``, ``"Supraordinal taxon"``.

        Returns
        -------
        labels : np.ndarray, shape (len(keys),)
        """
        return self.get_metadata(keys)[column].to_numpy()

    def get_landmarks(self, keys=None, slid=True):
        """Return 3D landmark coordinates for each specimen.

        Parameters
        ----------
        keys : list of str, optional
            PLY stem names.  When provided, rows are ordered to match ``keys``.
        slid : bool
            If True (default), use the slid semilandmarks (after sliding, before
            GPA).  If False, use the raw landmark file.

        Returns
        -------
        landmarks : np.ndarray, shape=(n_specimens, n_landmarks, 3)
            Landmark coordinates in original CT-scan units.
        """
        url = self.SLID_LMK_CSV_URL if slid else self.RAW_LMK_CSV_URL
        df  = self._download_csv(url)

        name_col   = df.columns[0]
        coord_cols = [c for c in df.columns if c != name_col]
        n_landmarks = len(coord_cols) // 3

        coords = df[coord_cols].to_numpy(dtype=float).reshape(len(df), n_landmarks, 3)

        def _norm(s):
            return str(s).strip().replace(" ", "_").lower()

        name_to_idx = {_norm(n): i for i, n in enumerate(df[name_col])}

        if keys is None:
            return coords

        result = np.full((len(keys), n_landmarks, 3), np.nan)
        for ki, k in enumerate(keys):
            k_norm = _norm(k)
            if k_norm in name_to_idx:
                result[ki] = coords[name_to_idx[k_norm]]
            else:
                match = next(
                    (v for kk, v in name_to_idx.items()
                     if kk in k_norm or k_norm in kk),
                    None,
                )
                if match is not None:
                    result[ki] = coords[match]

        return result
