"""Medical imaging dataset classes.

Requires optional dependencies: nibabel, scikit-image.
Install with: pip install nibabel scikit-image
"""

import os
import re

import numpy as np
from scipy import sparse

from geomfum.shape import TriangleMesh

# ACDC segmentation label mapping
LABEL_RV = 1  # right ventricle
LABEL_MYO = 2  # myocardium
LABEL_LV = 3  # left ventricle

STRUCTURE_LABELS = {
    "lv": LABEL_LV,
    "myo": LABEL_MYO,
    "rv": LABEL_RV,
}


def _laplacian_smooth(verts, faces, n_iter=3, lamb=0.5):
    """Umbrella Laplacian smoothing.

    Parameters
    ----------
    verts : array, shape=[n, 3]
    faces : array, shape=[m, 3]
    n_iter : int
    lamb : float  Relaxation factor in (0, 1).

    Returns
    -------
    verts : array, shape=[n, 3]
    """
    n = len(verts)
    tri = np.asarray(faces, dtype=np.int64)
    i = np.concatenate(
        [tri[:, 0], tri[:, 1], tri[:, 2], tri[:, 1], tri[:, 2], tri[:, 0]]
    )
    j = np.concatenate(
        [tri[:, 1], tri[:, 2], tri[:, 0], tri[:, 0], tri[:, 1], tri[:, 2]]
    )
    A = sparse.csr_matrix((np.ones(len(i)), (i, j)), shape=(n, n))
    deg = np.asarray(A.sum(axis=1)).flatten()
    deg[deg == 0] = 1.0
    L = sparse.diags(1.0 / deg) @ A

    verts = np.array(verts, dtype=float)
    for _ in range(n_iter):
        verts = (1.0 - lamb) * verts + lamb * (L @ verts)
    return verts


def nifti_seg_to_mesh(seg_path, label, voxel_spacing=True, smooth_iter=3):
    """Extract a triangle mesh from a NIfTI binary segmentation mask.

    Parameters
    ----------
    seg_path : str
        Path to the ``_gt.nii.gz`` segmentation file.
    label : int
        Integer label of the structure to extract.
    voxel_spacing : bool
        If True, multiply vertex positions by the NIfTI voxel size so
        coordinates are in mm.
    smooth_iter : int
        Number of Laplacian smoothing iterations.  Set to 0 to skip.

    Returns
    -------
    mesh : TriangleMesh
    """
    try:
        import nibabel as nib
    except ImportError as err:
        raise ImportError(
            "nibabel is required. Install with: pip install nibabel"
        ) from err
    try:
        from skimage.measure import marching_cubes
    except ImportError as err:
        raise ImportError(
            "scikit-image is required. Install with: pip install scikit-image"
        ) from err

    img = nib.load(seg_path)
    data = img.get_fdata()
    spacing = (
        tuple(float(s) for s in img.header.get_zooms()[:3])
        if voxel_spacing
        else (1.0, 1.0, 1.0)
    )

    mask = (data == label).astype(np.float32)
    if mask.sum() == 0:
        raise ValueError(
            f"Label {label} not found in '{seg_path}'. "
            f"Present labels: {np.unique(data).tolist()}"
        )

    verts, faces, _, _ = marching_cubes(mask, level=0.5, spacing=spacing)

    if smooth_iter > 0:
        verts = _laplacian_smooth(verts, faces, n_iter=smooth_iter)

    return TriangleMesh(vertices=verts, faces=faces)


def _parse_acdc_info(info_path):
    """Parse ACDC ``Info.cfg`` into a dict of strings."""
    info = {}
    with open(info_path) as fh:
        for line in fh:
            line = line.strip()
            if ":" in line:
                key, val = line.split(":", 1)
                info[key.strip()] = val.strip()
    return info


class AcdcPatient:
    """A single ACDC patient with lazy-loaded ED/ES meshes.

    Parameters
    ----------
    patient_dir : str
        Path to the patient directory (e.g. ``/data/acdc/training/patient001``).
    structure : str
        Cardiac structure to extract: ``'lv'``, ``'rv'``, or ``'myo'``.
    voxel_spacing : bool
        Apply voxel spacing so mesh coordinates are in mm.
    smooth_iter : int
        Laplacian smoothing iterations on the raw marching-cubes mesh.
    """

    def __init__(self, patient_dir, structure="lv", voxel_spacing=True, smooth_iter=3):
        self.patient_dir = patient_dir
        self.patient_id = os.path.basename(patient_dir)
        self.structure = structure
        self.voxel_spacing = voxel_spacing
        self.smooth_iter = smooth_iter

        info = _parse_acdc_info(os.path.join(patient_dir, "Info.cfg"))
        self.group = info.get("Group", "UNK")
        self.ed_frame = int(info.get("ED", 1))
        self.es_frame = int(info.get("ES", 1))
        self.height = float(info.get("Height", 0) or 0)
        self.weight = float(info.get("Weight", 0) or 0)

    @property
    def ed_seg_path(self):
        """Path to end-diastole segmentation file."""
        fname = f"{self.patient_id}_frame{self.ed_frame:02d}_gt.nii.gz"
        return os.path.join(self.patient_dir, fname)

    @property
    def es_seg_path(self):
        """Path to end-systole segmentation file."""
        fname = f"{self.patient_id}_frame{self.es_frame:02d}_gt.nii.gz"
        return os.path.join(self.patient_dir, fname)

    def load_ed(self):
        """Load and return the end-diastole TriangleMesh."""
        return nifti_seg_to_mesh(
            self.ed_seg_path,
            STRUCTURE_LABELS[self.structure],
            self.voxel_spacing,
            self.smooth_iter,
        )

    def load_es(self):
        """Load and return the end-systole TriangleMesh."""
        return nifti_seg_to_mesh(
            self.es_seg_path,
            STRUCTURE_LABELS[self.structure],
            self.voxel_spacing,
            self.smooth_iter,
        )

    def load_pair(self):
        """Return ``(mesh_ed, mesh_es)``."""
        return self.load_ed(), self.load_es()

    @property
    def metadata(self):
        """Metadata dict (no mesh loading)."""
        return {
            "patient_id": self.patient_id,
            "group": self.group,
            "ed_frame": self.ed_frame,
            "es_frame": self.es_frame,
            "height_cm": self.height,
            "weight_kg": self.weight,
        }

    def __repr__(self):
        """Return a string representation of the AcdcPatient."""
        return (
            f"AcdcPatient(id={self.patient_id!r}, group={self.group!r}, "
            f"structure={self.structure!r})"
        )


class AcdcDataset:
    """ACDC Automated Cardiac Diagnosis Challenge dataset.

    Parameters
    ----------
    root : str
        Path to the ACDC data directory.  Should contain ``patient001/``,
        ``patient002/``, … subdirectories (either directly or inside a
        ``training/`` or ``testing/`` subfolder — both layouts accepted).
    structure : str
        Cardiac structure: ``'lv'`` (left ventricle, default), ``'rv'``
        (right ventricle), ``'myo'`` (myocardium).
    groups : list[str] or None
        Filter by diagnostic group.  ACDC groups: ``'NOR'``, ``'DCM'``,
        ``'HCM'``, ``'MINF'``, ``'RVA'``.  ``None`` keeps all groups.
    voxel_spacing : bool
        Apply voxel spacing so mesh coordinates are in mm.  Default ``True``.
    smooth_iter : int
        Laplacian smoothing iterations on raw marching-cubes meshes.
        Default 3.  Set to 0 to skip smoothing.

    Notes
    -----
    **Download.** Register and download from
    https://www.creatis.insa-lyon.fr/Challenge/acdc/

    **Expected layout**::

        root/
        ├── patient001/
        │   ├── Info.cfg
        │   ├── patient001_frame01.nii.gz
        │   ├── patient001_frame01_gt.nii.gz   ← ED segmentation
        │   ├── patient001_frame12.nii.gz
        │   └── patient001_frame12_gt.nii.gz   ← ES segmentation
        ├── patient002/
        └── ...

    **Segmentation labels:** 0=background, 1=RV, 2=myocardium, 3=LV.

    Examples
    --------
    >>> dataset = AcdcDataset("/path/to/acdc/training", structure="lv")
    >>> print(dataset)
    AcdcDataset(root=..., n_patients=100, groups={'NOR': 20, ...})
    >>> mesh_ed, mesh_es, meta = dataset[0]
    >>> mesh_ed, mesh_es = dataset.patients[0].load_pair()
    """

    def __init__(
        self,
        root,
        structure="lv",
        groups=None,
        voxel_spacing=True,
        smooth_iter=3,
    ):
        self.root = root
        self.structure = structure
        self.voxel_spacing = voxel_spacing
        self.smooth_iter = smooth_iter

        # Accept root pointing directly to patient dirs or to a parent
        # containing training/ or testing/ subdirectories.
        if not any(re.match(r"patient\d+", d) for d in os.listdir(root)):
            for sub in ("training", "testing", "train", "test"):
                candidate = os.path.join(root, sub)
                if os.path.isdir(candidate) and any(
                    re.match(r"patient\d+", d) for d in os.listdir(candidate)
                ):
                    root = candidate
                    break

        patient_dirs = sorted(
            os.path.join(root, d)
            for d in os.listdir(root)
            if re.match(r"patient\d+", d) and os.path.isdir(os.path.join(root, d))
        )

        patients = []
        for d in patient_dirs:
            info_path = os.path.join(d, "Info.cfg")
            if os.path.exists(info_path):
                patients.append(
                    AcdcPatient(
                        d,
                        structure=structure,
                        voxel_spacing=voxel_spacing,
                        smooth_iter=smooth_iter,
                    )
                )

        if groups is not None:
            groups_set = set(groups)
            patients = [p for p in patients if p.group in groups_set]

        self.patients = patients

    # ------------------------------------------------------------------
    # Container protocol
    # ------------------------------------------------------------------

    def __len__(self):
        """Return the number of patients in the dataset."""
        return len(self.patients)

    def __getitem__(self, idx):
        """Return ``(mesh_ed, mesh_es, metadata)`` for patient at ``idx``."""
        patient = self.patients[idx]
        mesh_ed, mesh_es = patient.load_pair()
        return mesh_ed, mesh_es, patient.metadata

    def __iter__(self):
        """Iterate over patients, yielding ``(mesh_ed, mesh_es, metadata)``."""
        for patient in self.patients:
            mesh_ed, mesh_es = patient.load_pair()
            yield mesh_ed, mesh_es, patient.metadata

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def group_counts(self):
        """Dict mapping group name → patient count."""
        from collections import Counter

        return dict(Counter(p.group for p in self.patients))

    @property
    def metadata_list(self):
        """List of metadata dicts for all patients (no mesh loading)."""
        return [p.metadata for p in self.patients]

    def get_patient(self, patient_id):
        """Return ``AcdcPatient`` by ID string (e.g. ``'patient001'``)."""
        for p in self.patients:
            if p.patient_id == patient_id:
                return p
        raise KeyError(f"Patient '{patient_id}' not found in dataset.")

    def __repr__(self):
        """Return a string representation of the AcdcDataset."""
        return (
            f"AcdcDataset(root={self.root!r}, n_patients={len(self)}, "
            f"groups={self.group_counts}, structure={self.structure!r})"
        )
