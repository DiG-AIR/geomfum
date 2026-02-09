"""
Implementation of the Volumetric DiffusionNet feature extractor for tetrahedral meshes.

Extends DiffusionNet to handle 3D volumetric gradients instead of 2D
tangent-plane gradients on surfaces.

References
----------
.. DiffusionNet: Discretization Agnostic Learning on Surfaces
   Nicholas Sharp, Souhaib Attaiki, Keenan Crane, Maks Ovsjanikov
   https://arxiv.org/abs/2012.00888
"""

import gsops.backend as gs
import numpy as np
import scipy.sparse
import torch
import torch.nn as nn

from geomfum.descriptor.learned import BaseFeatureExtractor
from geomfum.wrap.diffusionnet import LearnedTimeDiffusion, MiniMLP


class VolumetricDiffusionnetFeatureExtractor(BaseFeatureExtractor, nn.Module):
    """Feature extractor using Volumetric DiffusionNet for tetrahedral meshes.

    Parameters
    ----------
    in_channels : int
        Number of input feature channels. Default is 3 (xyz).
    out_channels : int
        Number of output feature channels. Default is 128.
    hidden_channels : int
        Number of hidden channels in the network. Default is 128.
    n_block : int
        Number of DiffusionNet blocks. Default is 4.
    last_activation : nn.Module or None
        Activation function applied to the output. Default is None.
    mlp_hidden_channels : list or None
        Hidden layer sizes in the MLP blocks. Default is None.
    output_at : str
        Output type — one of ['vertices', 'global_mean']. Default is 'vertices'.
    dropout : bool
        Whether to apply dropout in MLP layers. Default is True.
    with_gradient_features : bool
        Whether to compute and include spatial gradient features. Default is True.
    with_gradient_rotations : bool
        Whether to use learned rotations in gradient features. Default is True.
    diffusion_method : str
        Diffusion method — one of ['spectral', 'implicit_dense']. Default is 'spectral'.
    k : int
        Number of eigenvectors/eigenvalues used for spectral diffusion. Default is 128.
    device : torch.device
        Device to run the model on. Default is CPU.
    descriptor : Descriptor or None
        Optional descriptor to compute input features. If None, uses vertex coordinates.
    """

    def __init__(
        self,
        in_channels=3,
        out_channels=128,
        hidden_channels=128,
        n_block=4,
        last_activation=None,
        mlp_hidden_channels=None,
        output_at="vertices",
        dropout=True,
        with_gradient_features=True,
        with_gradient_rotations=True,
        diffusion_method="spectral",
        k=128,
        device=torch.device("cpu"),
        descriptor=None,
    ):
        super(VolumetricDiffusionnetFeatureExtractor, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k = k
        self.device = device
        self.descriptor = descriptor

        self.model = (
            VolumetricDiffusionNet(
                in_channels=in_channels,
                out_channels=out_channels,
                hidden_channels=hidden_channels,
                n_block=n_block,
                last_activation=last_activation,
                mlp_hidden_channels=mlp_hidden_channels,
                output_at=output_at,
                dropout=dropout,
                with_gradient_features=with_gradient_features,
                with_gradient_rotations=with_gradient_rotations,
                diffusion_method=diffusion_method,
                k_eig=k,
            )
            .to(device)
            .float()
        )

    def forward(self, shape):
        """Forward pass through the Volumetric DiffusionNet model.

        Parameters
        ----------
        shape : TetrahedralMesh
            A tetrahedral mesh object.

        Returns
        -------
        torch.Tensor
            Extracted feature tensor of shape [1, V, out_channels].
        """
        v = gs.to_torch(shape.vertices).float().to(self.device)

        mass, L, evals, evecs, gradX, gradY, gradZ = self._get_operators(
            shape, k=self.k
        )

        v = v.unsqueeze(0).to(torch.float32)
        mass = mass.unsqueeze(0).to(torch.float32)
        L = L.unsqueeze(0).to(torch.float32)
        evals = evals.unsqueeze(0).to(torch.float32)
        evecs = evecs.unsqueeze(0).to(torch.float32)
        gradX = gradX.unsqueeze(0).to(torch.float32)
        gradY = gradY.unsqueeze(0).to(torch.float32)
        gradZ = gradZ.unsqueeze(0).to(torch.float32)

        if self.descriptor is None:
            input_feat = None
        else:
            input_feat = self.descriptor(shape)
            input_feat = gs.to_torch(input_feat).to(torch.float32).to(self.device)
            input_feat = input_feat.unsqueeze(0).transpose(2, 1)

            if input_feat.shape[-1] != self.in_channels:
                raise ValueError(
                    f"Input has {input_feat.shape[-1]} channels, "
                    f"expected {self.in_channels}."
                )

        return self.model(v, input_feat, mass, L, evals, evecs, gradX, gradY, gradZ)

    def _get_operators(self, mesh, k=200):
        """Compute spectral and gradient operators for a tetrahedral mesh.

        Parameters
        ----------
        mesh : TetrahedralMesh
            Input tetrahedral mesh.
        k : int
            Number of eigenvalues/eigenvectors to compute. Default 200.

        Returns
        -------
        massvec : torch.Tensor
            Diagonal elements of the mass matrix [n_vertices].
        L : torch.SparseTensor
            Sparse Laplacian matrix [n_vertices, n_vertices].
        evals : torch.Tensor
            Eigenvalues [spectrum_size].
        evecs : torch.Tensor
            Eigenvectors [n_vertices, spectrum_size].
        gradX : torch.SparseTensor
            X-component of vertex-to-vertex gradient [n_vertices, n_vertices].
        gradY : torch.SparseTensor
            Y-component of vertex-to-vertex gradient [n_vertices, n_vertices].
        gradZ : torch.SparseTensor
            Z-component of vertex-to-vertex gradient [n_vertices, n_vertices].
        """
        L, M = mesh.laplacian.find()
        evals, evecs = mesh.laplacian.find_spectrum(spectrum_size=k)

        massvec = torch.tensor(gs.sparse.to_scipy_csc(M).diagonal()).to(
            device=self.device, dtype=torch.float32
        )

        L = gs.sparse.to_torch_coo(gs.sparse.to_coo(L)).to(
            device=self.device, dtype=torch.float32
        )
        evals = gs.to_torch(evals).to(device=self.device, dtype=torch.float32)
        evecs = gs.to_torch(evecs).to(device=self.device, dtype=torch.float32)

        gradX, gradY, gradZ = self._build_vertex_gradient_matrices(mesh)

        return massvec, L, evals, evecs, gradX, gradY, gradZ

    def _build_vertex_gradient_matrices(self, mesh):
        """Build vertex-to-vertex gradient operators from the tetrahedral gradient.

        The tetrahedral gradient G is [3*n_tets, n_vertices] (per-tet, 3D).
        We split it into G_x, G_y, G_z each [n_tets, n_vertices], then compose
        with a tet-to-vertex averaging matrix A [n_vertices, n_tets] to get
        gradX, gradY, gradZ each [n_vertices, n_vertices].

        Parameters
        ----------
        mesh : TetrahedralMesh
            Input tetrahedral mesh.

        Returns
        -------
        gradX : torch.sparse.FloatTensor, shape=[n_vertices, n_vertices]
            X-component of vertex-to-vertex gradient.
        gradY : torch.sparse.FloatTensor, shape=[n_vertices, n_vertices]
            Y-component of vertex-to-vertex gradient.
        gradZ : torch.sparse.FloatTensor, shape=[n_vertices, n_vertices]
            Z-component of vertex-to-vertex gradient.
        """
        G_sparse = gs.sparse.to_scipy_csc(mesh.gradient.gradient_matrix)

        n_tets = mesh.n_tets
        n_verts = mesh.n_vertices

        # Split into x, y, z components, each [n_tets, n_vertices]
        G_x = G_sparse[0::3, :]
        G_y = G_sparse[1::3, :]
        G_z = G_sparse[2::3, :]

        # Build tet-to-vertex averaging matrix A [n_vertices, n_tets]
        # A[v, t] = vol(t) if vertex v is in tet t, then normalize rows
        tets_np = gs.to_numpy(gs.to_device(mesh.tets, "cpu"))
        vols_np = gs.to_numpy(gs.to_device(mesh.tet_volumes, "cpu"))

        rows = []
        cols = []
        vals = []
        for local_vi in range(4):
            vert_ids = tets_np[:, local_vi]
            tet_ids = np.arange(n_tets)
            rows.append(vert_ids)
            cols.append(tet_ids)
            vals.append(vols_np)

        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        vals = np.concatenate(vals)

        A = scipy.sparse.coo_matrix(
            (vals, (rows, cols)), shape=(n_verts, n_tets)
        ).tocsc()

        # Normalize rows so each row sums to 1
        row_sums = np.array(A.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1.0
        D_inv = scipy.sparse.diags(1.0 / row_sums)
        A = D_inv @ A

        # Compose: vertex-to-vertex gradient = averaging @ tet gradient
        gradX_vtx = A @ G_x
        gradY_vtx = A @ G_y
        gradZ_vtx = A @ G_z

        def _scipy_to_torch_sparse(sp_mat):
            sp_coo = sp_mat.tocoo()
            indices = torch.tensor(
                np.vstack([sp_coo.row, sp_coo.col]),
                dtype=torch.long,
            )
            values = torch.tensor(sp_coo.data, dtype=torch.float32)
            return torch.sparse_coo_tensor(indices, values, sp_coo.shape).to(
                self.device
            )

        return (
            _scipy_to_torch_sparse(gradX_vtx),
            _scipy_to_torch_sparse(gradY_vtx),
            _scipy_to_torch_sparse(gradZ_vtx),
        )


class VolumetricDiffusionNet(nn.Module):
    """Volumetric DiffusionNet: stack of VolumetricDiffusionNetBlocks.

    Adapts the DiffusionNet architecture to work with 3D volumetric
    gradient operators instead of 2D tangent-plane gradients.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    hidden_channels : int
        Number of hidden channels. Default 128.
    n_block : int
        Number of diffusion blocks. Default 4.
    last_activation : nn.Module or None
        Output activation. Default None.
    mlp_hidden_channels : list or None
        MLP hidden layer sizes. Default None.
    output_at : str
        Output location — one of ['vertices', 'global_mean']. Default 'vertices'.
    dropout : bool
        Whether to use dropout in MLP. Default True.
    with_gradient_features : bool
        Whether to use spatial gradient features. Default True.
    with_gradient_rotations : bool
        Whether to use learned rotations in gradient features. Default True.
    diffusion_method : str
        Diffusion method. Default 'spectral'.
    k_eig : int
        Number of eigenpairs. Default 128.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels=128,
        n_block=4,
        last_activation=None,
        mlp_hidden_channels=None,
        output_at="vertices",
        dropout=True,
        with_gradient_features=True,
        with_gradient_rotations=True,
        diffusion_method="spectral",
        k_eig=128,
    ):
        super(VolumetricDiffusionNet, self).__init__()

        assert diffusion_method in ["spectral", "implicit_dense"], (
            f"Invalid diffusion method: {diffusion_method}"
        )
        assert output_at in ["vertices", "global_mean"], (
            f"Invalid output_at: {output_at}. "
            "Volumetric meshes support 'vertices' and 'global_mean'."
        )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.n_block = n_block
        self.last_activation = last_activation
        self.output_at = output_at

        if not mlp_hidden_channels:
            mlp_hidden_channels = [hidden_channels, hidden_channels]

        self.first_linear = nn.Linear(in_channels, hidden_channels)
        self.last_linear = nn.Linear(hidden_channels, out_channels)

        blocks = []
        for _ in range(n_block):
            block = VolumetricDiffusionNetBlock(
                in_channels=hidden_channels,
                mlp_hidden_channels=mlp_hidden_channels,
                dropout=dropout,
                diffusion_method=diffusion_method,
                with_gradient_features=with_gradient_features,
                with_gradient_rotations=with_gradient_rotations,
            )
            blocks.append(block)
        self.blocks = nn.ModuleList(blocks)

    def forward(
        self,
        verts,
        feats=None,
        mass=None,
        L=None,
        evals=None,
        evecs=None,
        gradX=None,
        gradY=None,
        gradZ=None,
    ):
        """Forward pass of the Volumetric DiffusionNet.

        Parameters
        ----------
        verts : torch.Tensor
            Input vertices [B, V, 3].
        feats : torch.Tensor, optional
            Input features. Default None (uses vertices).
        mass : torch.Tensor
            Diagonal elements of mass matrix [B, V].
        L : torch.SparseTensor
            Sparse Laplacian matrix [B, V, V].
        evals : torch.Tensor
            Eigenvalues of Laplacian [B, K].
        evecs : torch.Tensor
            Eigenvectors of Laplacian [B, V, K].
        gradX : torch.SparseTensor
            X-component of vertex gradient [B, V, V].
        gradY : torch.SparseTensor
            Y-component of vertex gradient [B, V, V].
        gradZ : torch.SparseTensor
            Z-component of vertex gradient [B, V, V].

        Returns
        -------
        torch.Tensor
            Output features.
        """
        assert verts.dim() == 3, "Only support batch operation"

        x = feats if feats is not None else verts
        x = self.first_linear(x)

        for block in self.blocks:
            x = block(x, mass, L, evals, evecs, gradX, gradY, gradZ)

        x = self.last_linear(x)

        if self.output_at == "vertices":
            x_out = x
        else:  # global_mean
            x_out = torch.sum(x * mass.unsqueeze(-1), dim=-2) / torch.sum(
                mass, dim=-1, keepdim=True
            )

        if self.last_activation:
            x_out = self.last_activation(x_out)

        return x_out


class VolumetricDiffusionNetBlock(nn.Module):
    """Building block of Volumetric DiffusionNet.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    mlp_hidden_channels : list
        Hidden layer sizes for the MLP.
    dropout : bool
        Whether to use dropout. Default True.
    diffusion_method : str
        Diffusion method. Default 'spectral'.
    with_gradient_features : bool
        Whether to use spatial gradient features. Default True.
    with_gradient_rotations : bool
        Whether to use learned rotations in gradient features. Default True.
    """

    def __init__(
        self,
        in_channels,
        mlp_hidden_channels,
        dropout=True,
        diffusion_method="spectral",
        with_gradient_features=True,
        with_gradient_rotations=True,
    ):
        super(VolumetricDiffusionNetBlock, self).__init__()

        self.in_channels = in_channels
        self.with_gradient_features = with_gradient_features

        # Diffusion block (reused from surface DiffusionNet — purely spectral)
        self.diffusion = LearnedTimeDiffusion(in_channels, method=diffusion_method)

        # MLP input: original features + diffused features
        self.mlp_in_channels = 2 * in_channels

        # Spatial gradient block (3D for volumetric meshes)
        if self.with_gradient_features:
            self.gradient_features = VolumetricSpatialGradientFeatures(
                in_channels,
                with_gradient_rotations=with_gradient_rotations,
            )
            self.mlp_in_channels += in_channels

        self.mlp = MiniMLP(
            [self.mlp_in_channels] + mlp_hidden_channels + [in_channels],
            dropout=dropout,
        )

    def forward(self, feat_in, mass, L, evals, evecs, gradX, gradY, gradZ):
        """Forward pass of the volumetric diffusion block.

        Parameters
        ----------
        feat_in : torch.Tensor
            Input features [B, V, C].
        mass : torch.Tensor
            Mass vector [B, V].
        L : torch.SparseTensor
            Laplacian [B, V, V].
        evals : torch.Tensor
            Eigenvalues [B, K].
        evecs : torch.Tensor
            Eigenvectors [B, V, K].
        gradX : torch.SparseTensor
            X-component of vertex gradient [B, V, V].
        gradY : torch.SparseTensor
            Y-component of vertex gradient [B, V, V].
        gradZ : torch.SparseTensor
            Z-component of vertex gradient [B, V, V].

        Returns
        -------
        torch.Tensor
            Output features [B, V, C].
        """
        B = feat_in.shape[0]
        assert feat_in.shape[-1] == self.in_channels, (
            f"Expected feature channel: {self.in_channels}, "
            f"but got: {feat_in.shape[-1]}"
        )

        # Spectral diffusion (identical to surface version)
        feat_diffuse = self.diffusion(feat_in, L, mass, evals, evecs)

        if self.with_gradient_features:
            # Compute 3D gradient features
            feat_grads = []
            for b in range(B):
                gx = torch.mm(gradX[b, ...], feat_diffuse[b, ...])
                gy = torch.mm(gradY[b, ...], feat_diffuse[b, ...])
                gz = torch.mm(gradZ[b, ...], feat_diffuse[b, ...])
                feat_grads.append(torch.stack((gx, gy, gz), dim=-1))

            feat_grad = torch.stack(feat_grads, dim=0)  # [B, V, C, 3]
            feat_grad_features = self.gradient_features(feat_grad)

            feat_combined = torch.cat(
                (feat_in, feat_diffuse, feat_grad_features), dim=-1
            )
        else:
            feat_combined = torch.cat((feat_in, feat_diffuse), dim=-1)

        # MLP block
        feat_out = self.mlp(feat_combined)

        # Skip connection
        feat_out = feat_out + feat_in

        return feat_out


class VolumetricSpatialGradientFeatures(nn.Module):
    """Compute spatial gradient features for 3D volumetric meshes.

    Analogous to SpatialGradientFeatures but handles 3 gradient components
    (x, y, z) instead of 2 (real, imaginary of the complex tangent-plane gradient).

    Parameters
    ----------
    in_channels : int
        Number of input feature channels.
    with_gradient_rotations : bool
        Whether to use learned rotations in gradient space. Default True.
    """

    def __init__(self, in_channels, with_gradient_rotations=True):
        super(VolumetricSpatialGradientFeatures, self).__init__()

        self.in_channels = in_channels
        self.with_gradient_rotations = with_gradient_rotations

        if self.with_gradient_rotations:
            self.A_x = nn.Linear(in_channels, in_channels, bias=False)
            self.A_y = nn.Linear(in_channels, in_channels, bias=False)
            self.A_z = nn.Linear(in_channels, in_channels, bias=False)
        else:
            self.A = nn.Linear(in_channels, in_channels, bias=False)

    def forward(self, feat_in):
        """Compute spatial gradient features from 3D gradient vectors.

        Parameters
        ----------
        feat_in : torch.Tensor
            Input gradient features of shape [B, V, C, 3].

        Returns
        -------
        feat_out : torch.Tensor
            Output features of shape [B, V, C].
        """
        if self.with_gradient_rotations:
            feat_x = self.A_x(feat_in[..., 0])
            feat_y = self.A_y(feat_in[..., 1])
            feat_z = self.A_z(feat_in[..., 2])
        else:
            feat_x = self.A(feat_in[..., 0])
            feat_y = self.A(feat_in[..., 1])
            feat_z = self.A(feat_in[..., 2])

        feat_out = (
            feat_in[..., 0] * feat_x
            + feat_in[..., 1] * feat_y
            + feat_in[..., 2] * feat_z
        )

        return torch.tanh(feat_out)
