"""
Implementation of the DiffusionNet feature extractor for 3D shapes.

References
----------
..DiffusionNet: Discretization Agnostic Learning on Surfaces
Nicholas Sharp, Souhaib Attaiki, Keenan Crane, Maks Ovsjanikov
https://arxiv.org/abs/2012.00888
..https://github.com/dongliangcao/Self-Supervised-Multimodal-Shape-Matching by Dongliang Cao
..https://github.com/nmwsharp/diffusion-net

"""

import gsops.backend as gs
import torch
import torch.nn as nn

from geomfum.descriptor.learned import BaseFeatureExtractor
from geomfum.shape import TriangleMesh


class DiffusionnetFeatureExtractor(BaseFeatureExtractor, nn.Module):
    """Feature extractor that uses DiffusionNet for geometric deep learning on 3D mesh data.

    Parameters
    ----------
    in_channels : int
        Number of input feature channels (e.g., 3 for xyz). Default is 3.
    out_channels : int
        Number of output feature channels. Default is 128.
    hidden_channels : int
        Number of hidden channels in the network. Default is 128.
    n_block : int
        Number of DiffusionNet blocks. Default is 4.
    last_activation : nn.Module or None
        Activation function applied to the output. Default is None.
    mlp_hidden_channels : List[int] or None
        Hidden layer sizes in the MLP blocks. Default is None.
    output_at : str
        Output type — one of ['vertices', 'edges', 'faces', 'global_mean']. Default is 'vertices'.
    dropout : bool
        Whether to apply dropout in MLP layers. Default is True.
    with_gradient_features : bool
        Whether to compute and include spatial gradient features. Default is True.
    with_gradient_rotations : bool
        Whether to use gradient rotations in spatial features. Default is True.
    diffusion_method : str
        Diffusion method used — one of ['spectral', 'implicit_dense']. Default is 'spectral'.
    k : int
        Number of eigenvectors/eigenvalues used for spectral diffusion. Default is 128.
    cache_dir : str or None
        Path to cache directory for storing/loading spectral operators. Default is None.
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
        cache_dir=None,
        device=torch.device("cpu"),
        descriptor=None,
    ):
        super(DiffusionnetFeatureExtractor, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.n_block = n_block
        self.last_activation = last_activation
        self.mlp_hidden_channels = mlp_hidden_channels
        self.output_at = output_at
        self.dropout = dropout
        self.with_gradient_features = with_gradient_features
        self.with_gradient_rotations = with_gradient_rotations
        self.diffusion_method = diffusion_method
        self.k = k
        self.cache_dir = cache_dir
        self.model = (
            DiffusionNet(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                hidden_channels=self.hidden_channels,
                n_block=self.n_block,
                last_activation=self.last_activation,
                mlp_hidden_channels=self.mlp_hidden_channels,
                output_at=self.output_at,
                dropout=self.dropout,
                with_gradient_features=self.with_gradient_features,
                with_gradient_rotations=self.with_gradient_rotations,
                diffusion_method=self.diffusion_method,
                k_eig=self.k,
                cache_dir=self.cache_dir,
            )
            .to(device)
            .float()
        )
        self.descriptor = descriptor

        self.device = device

    def forward(self, shape):
        """Call pass through the DiffusionNet model.

        Parameters
        ----------
        shape : Shape
            A shape object.

        Returns
        -------
        torch.Tensor
            Extracted feature tensor of shape [1, V, out_channels].
        """
        # Support both Shape and dict
        v = gs.to_torch(shape.vertices).float().to(self.device)
        if isinstance(shape, TriangleMesh):
            f = gs.to_torch(shape.faces).int().to(self.device)
            f = f.unsqueeze(0).to(torch.float32)
        else:
            f = None
        # Compute spectral operators
        frames, mass, L, evals, evecs, gradX, gradY = self._get_operators(
            shape, k=self.k
        )
        v = v.unsqueeze(0).to(torch.float32)

        frames = frames.unsqueeze(0).to(torch.float32)
        mass = mass.unsqueeze(0).to(torch.float32)
        L = L.unsqueeze(0).to(torch.float32)
        evals = evals.unsqueeze(0).to(torch.float32)
        evecs = evecs.unsqueeze(0).to(torch.float32)
        gradX = gradX.unsqueeze(0).to(torch.float32)
        gradY = gradY.unsqueeze(0).to(torch.float32)

        if self.descriptor is None:
            input_feat = None
        else:
            input_feat = self.descriptor(shape)
            input_feat = gs.to_torch(input_feat).to(torch.float32).to(self.device)
            input_feat = input_feat.unsqueeze(0).transpose(2, 1)

            if input_feat.shape[-1] != self.in_channels:
                raise ValueError(
                    f"Input shape has {input_feat.shape[-1]} channels, "
                    f"but expected {self.in_channels} channels."
                )

        return self.model(v, f, input_feat, frames, mass, L, evals, evecs, gradX, gradY)

    def _get_operators(self, mesh, k=200):
        # TODO: add cache_dir
        """Compute the spectral operators for the input mesh.

        Parameters
        ----------
        mesh : TriangleMesh
            Input mesh.
        k : int
            Number of eigenvalues/eigenvectors to compute diffusion. Default 200.

        Returns
        -------
        frames : torch.Tensor
            Tangent frames for vertices.
        mass : torch.Tensor
            Diagonal elements in mass matrix [..., n_vertices].
        L : torch.SparseTensor
            Sparse Laplacian matrix [..., n_vertices, n_vertices].
        evals : torch.Tensor
            Eigenvalues of Laplacian Matrix [..., spectrum_dim].
        evecs : torch.Tensor
            Eigenvectors of Laplacian Matrix [..., n_vertices, spectrum_dim].
        gradX : torch.SparseTensor
            Real part of gradient matrix [..., n_vertices, n_vertices].
        gradY : torch.SparseTensor
            Imaginary part of gradient matrix [..., n_vertices, n_vertices].
        """
        assert k > 0, f"Number of eigenvalues/vectors should be positive, bug get {k}"

        frames = mesh.vertex_tangent_frames
        L, M = mesh.laplacian.find()
        evals, evecs = mesh.laplacian.find_spectrum(spectrum_size=k)
        grad = mesh.gradient.gradient_matrix
        grad_scipy = gs.sparse.to_scipy_csc(grad)
        frames = gs.to_torch(frames)
        massvec = torch.tensor(gs.sparse.to_scipy_csc(M).diagonal()).to(
            device=self.device, dtype=torch.float32
        )
        L = gs.sparse.to_torch_coo(gs.sparse.to_coo(L)).to(
            device=self.device, dtype=torch.float32
        )
        evals = gs.to_torch(evals).to(device=self.device, dtype=torch.float32)
        evecs = gs.to_torch(evecs).to(device=self.device, dtype=torch.float32)
        gradX = gs.sparse.to_torch_coo(
            gs.sparse.to_coo(gs.sparse.from_scipy_csc(grad_scipy.real))
        ).to(device=self.device, dtype=torch.float32)
        gradY = gs.sparse.to_torch_coo(
            gs.sparse.to_coo(gs.sparse.from_scipy_csc(grad_scipy.imag))
        ).to(device=self.device, dtype=torch.float32)
        return frames, massvec, L, evals, evecs, gradX, gradY


"""
Implementation from
https://github.com/dongliangcao/Self-Supervised-Multimodal-Shape-Matching by Dongliang Cao
"""


class DiffusionNet(nn.Module):
    """DiffusionNet: stacked of DiffusionBlocks.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    hidden_channels : int
        Number of hidden channels in diffusion block. Default 128.
    n_block : int
        Number of diffusion blocks. Default 4.
    last_activation : nn.Module or None
        Output layer. Default None.
    mlp_hidden_channels : List or None
        MLP hidden layers. Default None means [hidden_channels, hidden_channels].
    output_at : str
        Produce outputs at various mesh elements by averaging from vertices.
        One of ['vertices', 'edges', 'faces', 'global_mean']. Default 'vertices'.
    dropout : bool
        Whether use dropout in mlp. Default True.
    with_gradient_features : bool
        Whether use SpatialGradientFeatures in DiffusionBlock. Default True.
    with_gradient_rotations : bool
        Whether use gradient rotations in SpatialGradientFeatures. Default True.
    diffusion_method : str
        Diffusion method applied in diffusion layer.
        One of ['spectral', 'implicit_dense']. Default 'spectral'.
    k_eig : int
        Number of eigenvalues/eigenvectors to compute diffusion. Default 128.
    cache_dir : str or None
        Cache dir contains all pre-computed spectral operators. Default None.
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
        cache_dir=None,
    ):
        super(DiffusionNet, self).__init__()
        # sanity check
        assert diffusion_method in ["spectral", "implicit_dense"], (
            f"Invalid diffusion method: {diffusion_method}"
        )
        assert output_at in ["vertices", "edges", "faces", "global_mean"], (
            f"Invalid output_at: {output_at}"
        )

        # basic params
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.n_block = n_block
        self.cache_dir = cache_dir

        # output params
        self.last_activation = last_activation
        self.output_at = output_at

        # mlp options
        if not mlp_hidden_channels:
            mlp_hidden_channels = [hidden_channels, hidden_channels]
        self.mlp_hidden_channels = mlp_hidden_channels
        self.dropout = dropout

        # diffusion options
        self.diffusion_method = diffusion_method
        self.k_eig = k_eig

        # gradient feature options
        self.with_gradient_features = with_gradient_features
        self.with_gradient_rotations = with_gradient_rotations

        # setup networks

        # first and last linear layers
        self.first_linear = nn.Linear(in_channels, hidden_channels)
        self.last_linear = nn.Linear(hidden_channels, out_channels)

        # diffusion blocks
        blocks = []
        for i_block in range(self.n_block):
            block = DiffusionNetBlock(
                in_channels=hidden_channels,
                mlp_hidden_channels=mlp_hidden_channels,
                dropout=dropout,
                diffusion_method=diffusion_method,
                with_gradient_features=with_gradient_features,
                with_gradient_rotations=with_gradient_rotations,
            )
            blocks += [block]
        self.blocks = nn.ModuleList(blocks)

    def forward(
        self,
        verts,
        faces=None,
        feats=None,
        frames=None,
        mass=None,
        L=None,
        evals=None,
        evecs=None,
        gradX=None,
        gradY=None,
    ):
        """Compute the forward pass of the DiffusionNet.

        Parameters
        ----------
        verts : torch.Tensor
            Input vertices [B, V, 3].
        faces : torch.Tensor, optional
            Input faces [B, F, 3]. Default None.
        feats : torch.Tensor, optional
            Input features. Default None.
        frames : torch.Tensor
            Tangent frames for vertices.
        mass : torch.Tensor
            Diagonal elements in mass matrix.
        L : torch.SparseTensor
            Sparse Laplacian matrix.
        evals : torch.Tensor
            Eigenvalues of Laplacian Matrix.
        evecs : torch.Tensor
            Eigenvectors of Laplacian Matrix.
        gradX : torch.SparseTensor
            Real part of gradient matrix.
        gradY : torch.SparseTensor
            Imaginary part of gradient matrix.

        Returns
        -------
        torch.Tensor
            Output features.
        """
        assert verts.dim() == 3, "Only support batch operation"
        if faces is not None:
            assert faces.dim() == 3, "Only support batch operation"

        mass = mass
        L = L
        evals = evals
        evecs = evecs
        gradX = gradX
        gradY = gradY
        if feats is not None:
            x = feats
        else:
            x = verts

        x = self.first_linear(x)

        for block in self.blocks:
            x = block(x, mass, L, evals, evecs, gradX, gradY)

        x = self.last_linear(x)

        if self.output_at == "vertices":
            x_out = x
        elif self.output_at == "faces":
            x_gather = x.unsqueeze(-1).expand(-1, -1, -1, 3)
            faces_gather = faces.unsqueeze(2).expand(-1, -1, x.shape[-1], -1)
            x_out = torch.gather(x_gather, 1, faces_gather).mean(dim=-1)
        else:
            x_out = torch.sum(x * mass.unsqueeze(-1), dim=-1) / torch.sum(
                mass, dim=-1, keepdim=True
            )

        if self.last_activation:
            x_out = self.last_activation(x_out)

        return x_out


class DiffusionNetBlock(nn.Module):
    """Building Block of DiffusionNet.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    mlp_hidden_channels : List
        List of mlp hidden channels.
    dropout : bool
        Whether use dropout in MLP. Default True.
    diffusion_method : str
        Method for diffusion. Default "spectral".
    with_gradient_features : bool
        Whether use spatial gradient feature. Default True.
    with_gradient_rotations : bool
        Whether use spatial gradient rotation. Default True.
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
        super(DiffusionNetBlock, self).__init__()

        self.in_channels = in_channels
        self.mlp_hidden_channels = mlp_hidden_channels
        self.dropout = dropout
        self.with_gradient_features = with_gradient_features
        self.with_gradient_rotations = with_gradient_rotations

        # Diffusion block
        self.diffusion = LearnedTimeDiffusion(self.in_channels, method=diffusion_method)

        # concat of both diffused features and original features
        self.mlp_in_channels = 2 * self.in_channels

        # Spatial gradient block
        if self.with_gradient_features:
            self.gradient_features = SpatialGradientFeatures(
                self.in_channels, with_gradient_rotations=self.with_gradient_rotations
            )
            # concat of gradient features
            self.mlp_in_channels += self.in_channels

        # MLP block
        self.mlp = MiniMLP(
            [self.mlp_in_channels] + self.mlp_hidden_channels + [self.in_channels],
            dropout=self.dropout,
        )

    def forward(self, feat_in, mass, L, evals, evecs, gradX, gradY):
        """Compute the forward pass of the diffusion block.

        Parameters
        ----------
        feat_in : torch.Tensor
            Input feature vector [B, V, C].
        mass : torch.Tensor
            Diagonal elements of mass matrix [B, V].
        L : torch.SparseTensor
            Sparse Laplacian matrix [B, V, V].
        evals : torch.Tensor
            Eigenvalues of Laplacian Matrix [B, K].
        evecs : torch.Tensor
            Eigenvectors of Laplacian Matrix [B, V, K].
        gradX : torch.SparseTensor
            Real part of gradient matrix [B, V, V].
        gradY : torch.SparseTensor
            Imaginary part of gradient matrix [B, V, V].

        Returns
        -------
        torch.Tensor
            Output feature vector.
        """
        B = feat_in.shape[0]
        assert feat_in.shape[-1] == self.in_channels, (
            f"Expected feature channel: {self.in_channels}, but got: {feat_in.shape[-1]}"
        )

        # Diffusion block
        feat_diffuse = self.diffusion(feat_in, L, mass, evals, evecs)

        # Compute gradient features
        if self.with_gradient_features:
            # Compute gradient
            feat_grads = []
            for b in range(B):
                # gradient after diffusion
                feat_gradX = torch.mm(gradX[b, ...], feat_diffuse[b, ...])
                feat_gradY = torch.mm(gradY[b, ...], feat_diffuse[b, ...])

                feat_grads.append(torch.stack((feat_gradX, feat_gradY), dim=-1))
            feat_grad = torch.stack(feat_grads, dim=0)  # [B, V, C, 2]

            # Compute gradient features
            feat_grad_features = self.gradient_features(feat_grad)

            # Stack inputs to MLP
            feat_combined = torch.cat(
                (feat_in, feat_diffuse, feat_grad_features), dim=-1
            )
        else:
            # Stack inputs to MLP
            feat_combined = torch.cat((feat_in, feat_diffuse), dim=-1)

        # MLP block
        feat_out = self.mlp(feat_combined)

        # Skip connection
        feat_out = feat_out + feat_in

        return feat_out


class LearnedTimeDiffusion(nn.Module):
    """Applied diffusion with learned time per-channel.

    In the spectral domain this becomes f_out = e ^ (lambda_i * t) * f_in

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    method : str
        Method to perform time diffusion. Default 'spectral'.
    """

    def __init__(self, in_channels, method="spectral"):
        super(LearnedTimeDiffusion, self).__init__()
        assert method in ["spectral", "implicit_dense"], f"Invalid method: {method}"
        self.in_channels = in_channels
        self.diffusion_time = nn.Parameter(torch.Tensor(in_channels))
        self.method = method
        # init as zero
        nn.init.constant_(self.diffusion_time, 0.0)

    def forward(self, feat, L, mass, evals, evecs):
        """Forward pass of the diffusion layer.

        Parameters
        ----------
        feat : torch.Tensor
            Feature vector [B, V, C].
        L : torch.SparseTensor
            Sparse Laplacian matrix [B, V, V].
        mass : torch.Tensor
            Diagonal elements in mass matrix [B, V].
        evals : torch.Tensor
            Eigenvalues of Laplacian matrix [B, K].
        evecs : torch.Tensor
            Eigenvectors of Laplacian matrix [B, V, K].

        Returns
        -------
        feat_diffuse : torch.Tensor
            Diffused feature vector [B, V, C].
        """
        # project times to the positive half-space
        # (and away from 0 in the incredibly rare chance that they get stuck)
        with torch.no_grad():
            self.diffusion_time.data = torch.clamp(self.diffusion_time, min=1e-8)

        assert feat.shape[-1] == self.in_channels, (
            f"Expected feature channel: {self.in_channels}, but got: {feat.shape[-1]}"
        )

        if self.method == "spectral":
            # Transform to spectral
            feat_spec = torch.matmul(evecs.transpose(-2, -1), feat * mass.unsqueeze(-1))

            # Diffuse
            diffuse_coefs = torch.exp(
                -evals.unsqueeze(-1) * self.diffusion_time.unsqueeze(0)
            )
            feat_diffuse_spec = diffuse_coefs * feat_spec

            # Transform back to feature
            feat_diffuse = torch.matmul(evecs, feat_diffuse_spec)

        else:  # 'implicit_dense'
            # Form the dense matrix (M + tL) with dims (B, C, V, V)
            mat_dense = (
                L.to_dense().unsuqeeze(1).expand(-1, self.in_channels, -1, -1).clone()
            )
            mat_dense *= self.diffusion_time.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            mat_dense += torch.diag_embed(mass).unsqueeze(1)

            # Factor the system
            cholesky_factors = torch.linalg.cholesky(mat_dense)

            # Solve the system
            rhs = feat * mass.unsqueeze(-1)
            rhsT = rhs.transpose(1, 2).unsqueeze(-1)
            sols = torch.cholesky_solve(rhsT, cholesky_factors)
            feat_diffuse = sols.squeeze(-1).transpose(1, 2)

        return feat_diffuse


class SpatialGradientFeatures(nn.Module):
    """Compute dot-products between input vectors. Uses a learned complex-linear layer to keep dimension down.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    with_gradient_rotations : bool
        Whether with gradient rotations. Default True.
    """

    def __init__(self, in_channels, with_gradient_rotations=True):
        super(SpatialGradientFeatures, self).__init__()

        self.in_channels = in_channels
        self.with_gradient_rotations = with_gradient_rotations

        if self.with_gradient_rotations:
            self.A_re = nn.Linear(self.in_channels, self.in_channels, bias=False)
            self.A_im = nn.Linear(self.in_channels, self.in_channels, bias=False)
        else:
            self.A = nn.Linear(self.in_channels, self.in_channels, bias=False)

    def forward(self, feat_in):
        """Compute the spatial gradient features.

        Parameters
        ----------
        feat_in : torch.Tensor
            Input feature vector (B, V, C, 2).

        Returns
        -------
        feat_out : torch.Tensor
            Output feature vector (B, V, C)
        """
        feat_a = feat_in

        if self.with_gradient_rotations:
            feat_real_b = self.A_re(feat_in[..., 0]) - self.A_im(feat_in[..., 1])
            feat_img_b = self.A_re(feat_in[..., 0]) + self.A_im(feat_in[..., 1])
        else:
            feat_real_b = self.A(feat_in[..., 0])
            feat_img_b = self.A(feat_in[..., 1])

        feat_out = feat_a[..., 0] * feat_real_b + feat_a[..., 1] * feat_img_b

        return torch.tanh(feat_out)


class MiniMLP(nn.Sequential):
    """A simple MLP with configurable hidden layer sizes.

    Parameters
    ----------
    layer_sizes : List
        List of layer size.
    dropout : bool
        Whether use dropout. Default False.
    activation : nn.Module
        Activation function. Default ReLU.
    name : str
        Module name. Default 'miniMLP'
    """

    def __init__(self, layer_sizes, dropout=False, activation=nn.ReLU, name="miniMLP"):
        super(MiniMLP, self).__init__()

        for i in range(len(layer_sizes) - 1):
            is_last = i + 2 == len(layer_sizes)

            # Dropout Layer
            if dropout and i > 0:
                self.add_module(name + "_dropout_{:03d}".format(i), nn.Dropout(p=0.5))

            # Affine Layer
            self.add_module(
                name + "_linear_{:03d}".format(i),
                nn.Linear(layer_sizes[i], layer_sizes[i + 1]),
            )

            # Activation Layer
            if not is_last:
                self.add_module(name + "_activation_{:03d}".format(i), activation())
