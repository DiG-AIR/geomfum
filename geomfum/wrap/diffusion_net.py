'''
This is the wrapper of the diffusion net model from https://github.com/nmwsharp/diffusion-net
'''

from geomfum.descriptor._base import LearnedDescriptor
from geomfum.shape.mesh import TriangleMesh
import torch

class DiffusionNetDescriptor(torch.nn.Module,LearnedDescriptor):
    '''Descriptor representing the output of DiffusionNet.'''
    
    def __init__(self, in_channels=3, out_channels=128, hidden_channels=128, n_block=4, last_activation=None, 
                 mlp_hidden_channels=None, output_at='vertices', dropout=True, with_gradient_features=True, 
                 with_gradient_rotations=True, diffusion_method='spectral', k_eig=128, cache_dir=None, 
                 input_type='xyz', device=torch.device('cpu')):
        super(DiffusionNetDescriptor, self).__init__()
        
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
        self.k_eig = k_eig
        self.cache_dir = cache_dir
        self.input_type = input_type
        self.model = DiffusionNet(in_channels=self.in_channels, out_channels=self.out_channels, 
                                  hidden_channels=self.hidden_channels, n_block=self.n_block, 
                                  last_activation=self.last_activation, mlp_hidden_channels=self.mlp_hidden_channels, 
                                  output_at=self.output_at, dropout=self.dropout, 
                                  with_gradient_features=self.with_gradient_features, 
                                  with_gradient_rotations=self.with_gradient_rotations, 
                                  diffusion_method=self.diffusion_method, k_eig=self.k_eig, 
                                  cache_dir=self.cache_dir, input_type=self.input_type).to(device)

        self.n_features = self.out_channels
        self.device = device
        
    #this is both a nn.Module anda Learned Descriptor so it needs a forward and a __call__ method
    def forward(self, mesh):
        '''Forward pass through the DiffusionNet model, supports both TriangleMesh and dictionaries.'''
        
        if isinstance(mesh, dict):
            # If input is a dictionary containing tensors
            v = mesh['vertices'].to(torch.float32)
            f = mesh['faces'].to(torch.int32)
        elif isinstance(mesh, TriangleMesh):
            # If input is a TriangleMesh object, extract vertices and faces
            v = mesh.vertices.to(torch.float32)
            f = mesh.faces.to(torch.int32)
        else:
            raise TypeError("Input must be either a TriangleMesh or a dictionary containing 'vertices' and 'faces'")

        if v.dim() == 2 :
            v = v.unsqueeze(0)
            f = f.unsqueeze(0)

        self.features = self.model(v, f)
        return self.features

    def __call__(self, mesh):
        '''Forward pass through the DiffusionNet model, supports both TriangleMesh and dictionaries.'''
        
        if isinstance(mesh, dict):
            # If input is a dictionary containing tensors
            v = mesh['vertices'].to(torch.float32)
            f = mesh['faces'].to(torch.int32)
        elif isinstance(mesh, TriangleMesh):
            # If input is a TriangleMesh object, extract vertices and faces
            v = mesh.vertices.to(torch.float32)
            f = mesh.faces.to(torch.int32)
        else:
            raise TypeError("Input must be either a TriangleMesh or a dictionary containing 'vertices' and 'faces'")

        if v.dim() == 2 :
            v = v.unsqueeze(0)
            f = f.unsqueeze(0)

        self.features = self.model(v, f)
        return self.features

    def load_from_path(self, path):
        '''Load model parameters from the provided path'''
        self.model.load_state_dict(torch.load(path, map_location=self.device))

    def load(self, pre_model):
        '''Load model parameters from the provided pre-trained model.'''
        self.model.load_state_dict(pre_model)
    def save(self,path):
        torch.save(self.model.state_dict(), path)

    
    
"""
Implementation of DiffusionNet feature extractors from
https://github.com/dongliangcao/Self-Supervised-Multimodal-Shape-Matching by Dongliang Cao
"""

import torch.nn as nn

class LearnedTimeDiffusion(nn.Module):
    """
    Applied diffusion with learned time per-channel.

    In the spectral domain this becomes
        f_out = e ^ (lambda_i * t) * f_in
    """
    def __init__(self, in_channels, method='spectral'):
        """
        Args:
            in_channels (int): number of input channels.
            method (str, optional): method to perform time diffusion. Default 'spectral'.
        """
        super(LearnedTimeDiffusion, self).__init__()
        assert method in ['spectral', 'implicit_dense'], f'Invalid method: {method}'
        self.in_channels = in_channels
        self.diffusion_time = nn.Parameter(torch.Tensor(in_channels))
        self.method = method
        # init as zero
        nn.init.constant_(self.diffusion_time, 0.0)

    def forward(self, feat, L, mass, evals, evecs):
        """
        Args:
            feat (torch.Tensor): feature vector [B, V, C].
            L (torch.SparseTensor): sparse Laplacian matrix [B, V, V].
            mass (torch.Tensor): diagonal elements in mass matrix [B, V].
            evals (torch.Tensor): eigenvalues of Laplacian matrix [B, K].
            evecs (torch.Tensor): eigenvectors of Laplacian matrix [B, V, K].
        Returns:
            feat_diffuse (torch.Tensor): diffused feature vector [B, V, C].
        """
        # project times to the positive half-space
        # (and away from 0 in the incredibly rare chance that they get stuck)
        with torch.no_grad():
            self.diffusion_time.data = torch.clamp(self.diffusion_time, min=1e-8)

        assert feat.shape[-1] == self.in_channels, f'Expected feature channel: {self.in_channels}, but got: {feat.shape[-1]}'

        if self.method == 'spectral':
            # Transform to spectral
            feat_spec = to_basis(feat, evecs, mass)

            # Diffuse
            diffuse_coefs = torch.exp(-evals.unsqueeze(-1) * self.diffusion_time.unsqueeze(0))
            feat_diffuse_spec = diffuse_coefs * feat_spec

            # Transform back to feature
            feat_diffuse = from_basis(feat_diffuse_spec, evecs)

        else: # 'implicit_dense'
            V = feat.shape[-2]

            # Form the dense matrix (M + tL) with dims (B, C, V, V)
            mat_dense = L.to_dense().unsuqeeze(1).expand(-1, self.in_channels, -1, -1).clone()
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
    """
    Compute dot-products between input vectors.
    Uses a learned complex-linear layer to keep dimension down.
    """
    def __init__(self, in_channels, with_gradient_rotations=True):
        """
        Args:
            in_channels (int): number of input channels.
            with_gradient_rotations (bool, optional): whether with gradient rotations. Default True.
        """
        super(SpatialGradientFeatures, self).__init__()

        self.in_channels = in_channels
        self.with_gradient_rotations = with_gradient_rotations

        if self.with_gradient_rotations:
            self.A_re = nn.Linear(self.in_channels, self.in_channels, bias=False)
            self.A_im = nn.Linear(self.in_channels, self.in_channels, bias=False)
        else:
            self.A = nn.Linear(self.in_channels, self.in_channels, bias=False)

    def forward(self, feat_in):
        """
        Args:
            feat_in (torch.Tensor): input feature vector (B, V, C, 2).
        Returns:
            feat_out (torch.Tensor): output feature vector (B, V, C)
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
    """
    A simple MLP with configurable hidden layer sizes
    """
    def __init__(self, layer_sizes, dropout=False, activation=nn.ReLU, name='miniMLP'):
        """
        Args:
            layer_sizes (List): list of layer size.
            dropout (bool, optional): whether use dropout. Default False.
            activation (nn.Module, optional): activation function. Default ReLU.
            name (str, optional): module name. Default 'miniMLP'
        """
        super(MiniMLP, self).__init__()

        for i in range(len(layer_sizes) - 1):
            is_last = (i + 2 == len(layer_sizes))

            # Dropout Layer
            if dropout and i > 0:
                self.add_module(
                    name + '_dropout_{:03d}'.format(i),
                    nn.Dropout(p=0.5)
                )

            # Affine Layer
            self.add_module(
                name + '_linear_{:03d}'.format(i),
                nn.Linear(layer_sizes[i], layer_sizes[i+1])
            )

            # Activation Layer
            if not is_last:
                self.add_module(
                    name + '_activation_{:03d}'.format(i),
                    activation()
                )


class DiffusionNetBlock(nn.Module):
    """
    Building Block of DiffusionNet.
    """
    def __init__(self, in_channels, mlp_hidden_channels,
                 dropout=True,
                 diffusion_method='spectral',
                 with_gradient_features=True,
                 with_gradient_rotations=True):
        """
        Args:
            in_channels (int): number of input channels.
            mlp_hidden_channels (List): list of mlp hidden channels.
            dropout (bool, optional): whether use dropout in MLP. Default True.
            with_gradient_features (bool, optional): whether use spatial gradient feature. Default True.
            with_gradient_rotations (bool, optional): whether use spatial gradient rotation. Default True.
        """
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
            self.gradient_features = SpatialGradientFeatures(self.in_channels,
                                                             with_gradient_rotations=self.with_gradient_rotations)
            # concat of gradient features
            self.mlp_in_channels += self.in_channels

        # MLP block
        self.mlp = MiniMLP([self.mlp_in_channels] + self.mlp_hidden_channels + [self.in_channels], dropout=self.dropout)

    def forward(self, feat_in, mass, L, evals, evecs, gradX, gradY):
        """
        Args:
            feat_in (torch.Tensor): input feature vector [B, V, C].
            mass (torch.Tensor): diagonal elements of mass matrix [B, V].
            L (torch.SparseTensor): sparse Laplacian matrix [B, V, V].
            evals (torch.Tensor): eigenvalues of Laplacian Matrix [B, K].
            evecs (torch.Tensor): eigenvectors of Laplacian Matrix [B, V, K].
            gradX (torch.SparseTensor): real part of gradient matrix [B, V, V].
            gradY (torch.SparseTensor): imaginary part of gradient matrix [B, V, V].
        """

        B = feat_in.shape[0]
        assert feat_in.shape[-1] == self.in_channels, f'Expected feature channel: {self.in_channels}, but got: {feat_in.shape[-1]}'

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
            feat_grad = torch.stack(feat_grads, dim=0) # [B, V, C, 2]

            # Compute gradient features
            feat_grad_features = self.gradient_features(feat_grad)

            # Stack inputs to MLP
            feat_combined = torch.cat((feat_in, feat_diffuse, feat_grad_features), dim=-1)
        else:
            # Stack inputs to MLP
            feat_combined = torch.cat((feat_in, feat_diffuse), dim=-1)

        # MLP block
        feat_out = self.mlp(feat_combined)

        # Skip connection
        feat_out = feat_out + feat_in

        return feat_out


class DiffusionNet(nn.Module):
    """
    DiffusionNet: stacked of DiffusionBlock
    """
    def __init__(self, in_channels, out_channels,
                 hidden_channels=128,
                 n_block=4,
                 last_activation=None,
                 mlp_hidden_channels=None,
                 output_at='vertices',
                 dropout=True,
                 with_gradient_features=True,
                 with_gradient_rotations=True,
                 diffusion_method='spectral',
                 k_eig=128,
                 cache_dir=None,
                 input_type='xyz'
                 ):
        """
        Args:
            in_channels (int): number of input channels.
            out_channels (int): number of output channels.
            hidden_channels (int, optional): number of hidden channels in diffusion block. Default 128.
            n_block (int, optional): number of diffusion blocks. Default 4.
            last_activation (nn.Module, optional): output layer. Default None.
            mlp_hidden_channels (List, optional): mlp hidden layers. Default None means [hidden_channels, hidden_channels].
            output_at (str, optional): produce outputs at various mesh elements by averaging from vertices.
            One of ['vertices', 'edges', 'faces', 'global_mean']. Default 'vertices'.
            dropout (bool, optional): whether use dropout in mlp. Default True.
            with_gradient_features (bool, optional): whether use SpatialGradientFeatures in DiffusionBlock. Default True.
            with_gradient_rotations (bool, optional): whether use gradient rotations in SpatialGradientFeatures. Default True.
            diffusion_method (str, optional): diffusion method applied in diffusion layer.
            One of ['spectral', 'implicit_dense']. Default 'spectral'.
            k_eig (int, optional): number of eigenvalues/eigenvectors to compute diffusion. Default 128.
            cache_dir (str, optional): cache dir contains all pre-computed spectral operators. Default None.
            input_type (str, optional): input type. One of ['xyz', 'shot', 'hks'] Default 'xyz'.
        """
        super(DiffusionNet, self).__init__()
        # sanity check
        assert diffusion_method in ['spectral', 'implicit_dense'], f'Invalid diffusion method: {diffusion_method}'
        assert output_at in ['vertices', 'edges', 'faces', 'global_mean'], f'Invalid output_at: {output_at}'

        # basic params
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.n_block = n_block
        self.cache_dir = cache_dir
        self.input_type = input_type

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
                with_gradient_rotations=with_gradient_rotations
            )
            blocks += [block]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, verts, faces=None, feats=None):
        assert verts.dim() == 3, 'Only support batch operation'
        if faces is not None:
            assert faces.dim() == 3, 'Only support batch operation'

        # ensure reproducibility to first convert to cpu to find the precomputed spectral ops
        if faces is not None:
            _, mass, L, evals, evecs, gradX, gradY = get_all_operators(verts.cpu(), faces.cpu(), k=self.k_eig,
                                                                       cache_dir=self.cache_dir)
        else:
            _, mass, L, evals, evecs, gradX, gradY = get_all_operators(verts.cpu(), None, k=self.k_eig,
                                                                       cache_dir=self.cache_dir)
        mass = mass.to(device=verts.device)
        L = L.to(device=verts.device)
        evals = evals.to(device=verts.device)
        evecs = evecs.to(device=verts.device)
        gradX = gradX.to(device=verts.device)
        gradY = gradY.to(device=verts.device)

        # Compute hks when necessary
        if feats is not None:
            x = feats
        else:
            if self.input_type == 'hks':
                x = compute_hks_autoscale(evals, evecs)
            elif self.input_type == 'wks':
                x = compute_wks_autoscale(evals, evecs, mass)
            elif self.input_type == 'xyz':
                if self.training:
                    verts = data_augmentation(verts)
                x = verts

        # Apply the first linear layer
        x = self.first_linear(x)

        # Apply each of the diffusion block
        for block in self.blocks:
            x = block(x, mass, L, evals, evecs, gradX, gradY)

        # Apply the last linear layer
        x = self.last_linear(x)

        # remap output to faces/edges if requested
        if self.output_at == 'vertices':
            x_out = x
        elif self.output_at == 'faces':
            x_gather = x.unsqueeze(-1).expand(-1, -1, -1, 3)
            faces_gather = faces.unsqueeze(2).expand(-1, -1, x.shape[-1], -1)
            x_out = torch.gather(x_gather, 1, faces_gather).mean(dim=-1)
        else:  # global mean
            # Using a weighted mean according to the point mass/area is discretization-invariant.
            x_out = torch.sum(x * mass.unsqueeze(-1), dim=-1) / torch.sum(mass, dim=-1, keepdim=True)

        # Apply last non-linearity if specified
        if self.last_activation:
            x_out = self.last_activation(x_out)

        return x_out

"""
DiffusionNet Utils: Implementation by Dongliang Cao, this part can be fully removed and adapted to the geomfum library
"""
import os
import os.path as osp
import random
import hashlib
import numpy as np

import scipy
import scipy.spatial
import scipy.sparse.linalg as sla
import sklearn.neighbors as neighbors

import robust_laplacian
import potpourri3d as pp3d

import torch


# Hash a list of numpy arrays
def hash_arrays(arrs):
    running_hash = hashlib.sha1()
    for arr in arrs:
        if arr is not None:
            binarr = arr.view(np.uint8)
            running_hash.update(binarr)
    return running_hash.hexdigest()


def torch2np(tensor):
    assert isinstance(tensor, torch.Tensor)
    return tensor.detach().cpu().numpy()


def sparse_np_to_torch(A):
    Acoo = A.tocoo()
    values = Acoo.data
    indices = np.vstack((Acoo.row, Acoo.col))
    shape = Acoo.shape
    return torch.sparse_coo_tensor(torch.LongTensor(indices), torch.FloatTensor(values), torch.Size(shape)).coalesce()


def sparse_torch_to_np(A):
    assert len(A.shape) == 2

    indices = torch2np(A.indices())
    values = torch2np(A.values())

    mat = scipy.sparse.coo_matrix((values, indices), shape=A.shape).tocsc()
    return mat


def to_basis(feat, basis, massvec):
    """
    Transform feature into coefficients of orthonormal basis.
    Args:
        feat (torch.Tensor): feature vector [B, V, C]
        basis (torch.Tensor): functional basis [B, V, K]
        massvec (torch.Tensor): mass vector [B, V]
    Returns:
        coef (torch.Tensor): coefficient of basis [B, K, C]
    """
    basis_t = basis.transpose(-2, -1)
    coef = torch.matmul(basis_t, feat * massvec.unsqueeze(-1))
    return coef


def from_basis(coef, basis):
    """
    Transform coefficients of orthonormal basis into feature.
    Args:
        coef (torch.Tensor): coefficients [B, K, C]
        basis (torch.Tensor): functional basis [B, V, K]
    Returns:
        feat (torch.Tensor): feature vector [B, V, C]
    """
    feat = torch.matmul(basis, coef)
    return feat


def dot(a, b, keepdim=False):
    """
    Compute the dot product between vector a and vector b in last dimension

    Args:
        a (torch.Tensor): vector a [N, C].
        b (torch.Tensor): vector b [N, C].
        keepdim (bool, optional): keep dimension.
    Return:
        (torch.Tensor): dot product between a and b [N] or [N, 1].
    """
    assert a.shape == b.shape
    return torch.sum(a * b, dim=-1, keepdim=keepdim)


def cross(a, b):
    """
    Compute the cross product between vector a and vector b in last dimension

    Args:
        a (torch.Tensor): vector a [N, 3].
        b (torch.Tensor): vector b [N, 3].
    Return:
        (torch.Tensor): cross product between a and b [N, 3].
    """
    assert a.shape == b.shape and a.shape[-1] == 3
    return torch.cross(a, b, dim=-1)


def norm(x, keepdim=False):
    """
    Compute norm of an array of vectors.
    Given (N, C), return (N) or (N, 1) after norm along last dimension.
    """
    return torch.norm(x, dim=-1, keepdim=keepdim)


def square_norm(x, keepdim=False):
    """
    Compute square norm of an array of vectors.
    Given (N, C), return (N) after norm along last dimension.
    """
    return dot(x, x, keepdim=keepdim)


def normalize(x, eps=1e-12):
    """
    Normalize an array of vectors along last dimension.
    Given (N, C), return (N, C) after normalization.
    """
    assert x.dim() != 1
    return x / (norm(x, keepdim=True) + eps)


def face_coords(verts, faces):
    """
    Return face coordinates.
    Args:
        verts (torch.Tensor): vertices [V, 3]
        faces (torch.LongTensor): faces [F, 3]
    Return:
        coords (torch.Tensor): face coordinates [F, 3, 3]
    """
    coords = verts[faces]
    return coords


def project_to_tangent(vecs, normals):
    """
    Compute the tangent vectors of normals by vecs - proj(vecs, normals).
    Args:
        vecs (torch.Tensor): vecs [V, 3].
        normals (torch.Tensor): normal vectors assume to be unit [V, 3].
    """
    return vecs - dot(vecs, normals, keepdim=True) * normals


def face_area(verts, faces):
    """
    Compute face areas
    Args:
        verts (torch.Tensor): verts [V, 3]
        faces (torch.LongTensor): faces [F, 3]
    """
    coords = face_coords(verts, faces)
    vec_A = coords[:, 1, :] - coords[:, 0, :]
    vec_B = coords[:, 2, :] - coords[:, 0, :]

    # compute area by cross product
    normal = cross(vec_A, vec_B)
    return 0.5 * norm(normal)


def face_normal(verts, faces, is_normalize=True):
    """
    Compute face normal
    Args:
        verts (torch.Tensor): verts [V, 3]
        faces (torch.LongTensor): faces [F, 3]
        is_normalize (bool, optional): whether normalize face normal. Default True.
    """
    coords = face_coords(verts, faces)
    vec_A = coords[:, 1, :] - coords[:, 0, :]
    vec_B = coords[:, 2, :] - coords[:, 0, :]

    normal = cross(vec_A, vec_B)

    if is_normalize:
        normal = normalize(normal)

    return normal


def neighborhood_normal(pts):
    """
    Compute point cloud normal by performing PCA in neighborhood points.
    Args:
        pts (np.ndarray): points [V, N, 3], N: number of neighbors.
    """
    _, _, vh = np.linalg.svd(pts, full_matrices=False)
    normal = vh[:, 2, :]
    return normal / (np.linalg.norm(normal, axis=-1, keepdims=True) + 1e-12)


def mesh_vertex_normal(verts, faces):
    """
    Compute mesh vertex normal by adding neighboring faces' normals.
    Args:
        verts (np.ndarray): vertices [V, 3]
        faces (np.ndarray): faces [F, 3]
    Return:
        vertex_normals (np.ndarray): vertex normals [V, 3]
    """
    face_n = torch2np(face_normal(torch.tensor(verts), torch.tensor(faces)))

    vertex_normals = np.zeros_like(verts)
    for i in range(3):
        np.add.at(vertex_normals, faces[:, i], face_n)

    vertex_normals = vertex_normals / (np.linalg.norm(vertex_normals, axis=-1, keepdims=True) + 1e-12)

    return vertex_normals


def vertex_normal(verts, faces, n_neighbors=30):
    """
    Compute vertex normal supported by both point cloud and mesh

    Args:
        verts (torch.Tensor): vertices [V, 3].
        faces (torch.Tensor): faces [F, 3].
        n_neighbors (int, optional): number of neighbors to compute normal for point cloud. Default 30.
    """
    verts_np = torch2np(verts)

    if faces is None: # point cloud
        _, neigh_inds = find_knn(verts, verts, n_neighbors, omit_diagonal=True, method='cpu_kd')
        neigh_points = verts_np[torch2np(neigh_inds), :]
        neigh_points = neigh_points - verts_np[:, None, :]
        normals = neighborhood_normal(neigh_points)
    else:
        faces_np = torch2np(faces)
        normals = mesh_vertex_normal(verts_np, faces_np)

        # if any NaN, wiggle slightly and recompute
        bad_mask = np.isnan(normals).any(axis=1, keepdims=True)
        if bad_mask.any():
            bbox = np.amax(verts_np, axis=0) - np.amin(verts_np, axis=0)
            scale = np.linalg.norm(bbox) * 1e-4
            wiggle = (np.random.RandomState(seed=777).rand(*verts.shape) - 0.5) * scale
            wiggle_verts = verts_np + bad_mask * wiggle
            normals = mesh_vertex_normal(wiggle_verts, faces_np)

        # if still NaN assign random normals (probably unreferenced verts in mesh)
        bad_mask = np.isnan(normals).any(axis=1)
        if bad_mask.any():
            normals[bad_mask, :] = (np.random.RandomState(seed=777).rand(*verts.shape)-0.5)[bad_mask, :]
            normals = normals / (np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-12)

    normals = torch.from_numpy(normals).to(device=verts.device, dtype=verts.dtype)

    if torch.any(torch.isnan(normals)):
        raise ValueError('NaN normals')

    return normals


def find_knn(src_pts, target_pts, k, largest=False, omit_diagonal=False, method='brute'):
    """
    Finds the k nearest neighbors of source on target
    Args:
        src_pts (torch.Tensor): source points [Vs, 3]
        target_pts (torch.Tensor): target points [Vt, 3]
        k (int): number of neighbors
        largest (bool, optional): whether k largest neighbors. Default False.
        omit_diagonal (bool, optional): whether omit the point itself. Default False.
        method (str, optional): method, support 'brute', 'cpu_kd'. Default 'brute'
    Returns:
        dist (torch.Tensor): distances [Vs, k]
        indices (torch.Tensor): indices [Vs, k]
    """
    assert method in ['brute', 'cpu_kd'], f'Invalid method: {method}, only supports "brute" or "cpu_kd"'
    if omit_diagonal and src_pts.shape[0] != target_pts.shape[0]:
        raise ValueError('omit_diagonal can only be used when source and target are the same shape')

    # use 'cpu_kd' for large points
    if src_pts.shape[0] * target_pts.shape[0] > 1e8:
        method = 'cpu_kd'

    if method == 'brute':
        # Expand so both are VsxVtx3 tensor
        src_pts_expand = src_pts.unsqueeze(1).expand(-1, target_pts.shape[0], -1)
        target_pts_expand = target_pts.unsqueeze(0).expand(src_pts.shape[0], -1, -1)

        # Compute distance between target points and source points
        dist_mat = norm(src_pts_expand - target_pts_expand)

        if omit_diagonal:
            torch.diagonal(dist_mat)[:] = float('inf')

        dist, indices = torch.topk(dist_mat, k=k, largest=largest, sorted=True)
        return dist, indices
    else: # 'cpu_kd'
        assert largest == False, 'cannot do largest with cpu_kd'

        src_pts_np = torch2np(src_pts)
        target_pts_np = torch2np(target_pts)

        # Build the kd-tree
        kd_tree = neighbors.KDTree(target_pts_np)

        k_search = k + 1 if omit_diagonal else k
        _, indices = kd_tree.query(src_pts_np, k=k_search)

        if omit_diagonal:
            # Mask out self element
            mask = indices != np.arange(indices.shape[0])[:, None]

            # make sure we mask out exactly one element in each row, in rare case of many duplicate points
            mask[np.sum(mask, axis=1) == mask.shape[1], -1] = False

            indices = indices[mask].reshape((indices.shape[0], indices.shape[1]-1))

        indices = torch.tensor(indices, device=src_pts.device, dtype=torch.int64)
        dist = norm(src_pts.unsqueeze(1).expand(-1, k, -1) - target_pts[indices])

        return dist, indices


def build_targent_frames(verts, faces, vert_normals=None):
    """
    Build targent frames for each vertices with three orthogonal basis.
    Args:
        verts (torch.Tensor): vertices [V, 3].
        faces (torch.Tensor): faces [F, 3]
        vert_normals (torch.Tensor, optional): vertex normals [V, 3]. Default None
    Return:
        frames (torch.Tensor): frames [V, 3, 3]
    """
    V = verts.shape[0]
    device = verts.device
    dtype = verts.dtype

    # compute vertex normals when necessary
    if not vert_normals:
        vert_normals = vertex_normal(verts, faces)

    # find an orthogonal basis
    basis_cand1 = torch.tensor([1, 0, 0], device=device, dtype=dtype).expand(V, -1)
    basis_cand2 = torch.tensor([0, 1, 0], device=device, dtype=dtype).expand(V, -1)

    basisX = torch.where((torch.abs(dot(vert_normals, basis_cand1, keepdim=True)) < 0.9), basis_cand1, basis_cand2)
    basisX = project_to_tangent(basisX, vert_normals)
    basisX = normalize(basisX)
    basisY = cross(vert_normals, basisX)
    frames = torch.stack((basisX, basisY, vert_normals), dim=-2)

    if torch.any(torch.isnan(frames)):
        raise ValueError("NaN coordinate frame! Must be very degenerate")

    return frames


def build_grad_point_cloud(verts, frames, n_neighbors=30):
    """
    Build gradient matrix for point cloud
    Args:
        verts (torch.Tensor): vertices [V, 3].
        frames (torch.Tensor): frames [V, 3, 3].
        n_neighbors (int, optional): number of neighbors. Default 30.
    Returns:

    """
    verts_np = torch2np(verts)

    # find neighboring points
    _, neigh_inds = find_knn(verts, verts, n_neighbors, omit_diagonal=True, method='cpu_kd')

    # build edges
    edge_inds_from = np.repeat(np.arange(verts.shape[0]), n_neighbors)
    edges = np.stack((edge_inds_from, torch2np(neigh_inds).flatten()))
    edge_tangent_vecs = edge_tangent_vectors(verts, frames, edges)

    return build_grad(verts_np, edges, torch2np(edge_tangent_vecs))


def edge_tangent_vectors(verts, frames, edges):
    """
    Compute edge tangent vectors
    Args:
        verts (torch.Tensor): vertices [V, 3].
        frames (torch.Tensor): frames [V, 3, 3].
        edges (torch.Tensor): edges [2, E], where E = V * k, k: number of nearest neighbor.
    Returns:
        egde_tangent (torch.Tensor): edge tangent vectors [E, 2].
    """
    edge_vecs = verts[edges[1, :], :] - verts[edges[0, :], :]
    basisX = frames[edges[0, :], 0, :]
    basisY = frames[edges[0, :], 1, :]

    compX = dot(edge_vecs, basisX)
    compY = dot(edge_vecs, basisY)
    edge_tangent = torch.stack((compX, compY), dim=-1)

    return edge_tangent


def build_grad(verts, edges, edge_tangent_vectors):
    """
    Build a (V, V) complex sparse matrix grad operator.
    Given real inputs at vertices,
    produces a complex (vector value) at vertices giving the gradient.

    Args:
        verts (np.ndarray): vertices [V, 3]
        edges (np.ndarray): edges [2, E]
        edge_tangent_vectors (np.ndarray): edge tangent vectors [E, 2]
    """

    # Build outgoining neighbor lists
    V = verts.shape[0]
    vert_edge_outgoing = [[] for _ in range(V)]
    for e in range(edges.shape[1]):
        tail_ind = edges[0, e]
        tip_ind = edges[1, e]
        if tip_ind != tail_ind:
            vert_edge_outgoing[tail_ind].append(e)

    # Build local inversion matrix for each vertex
    row_inds = []
    col_inds = []
    data_vals = []
    eps_reg = 1e-5
    for iv in range(V):
        n_neigh = len(vert_edge_outgoing[iv])

        lhs_mat = np.zeros((n_neigh, 2))
        rhs_mat = np.zeros((n_neigh, n_neigh + 1))
        ind_lookup = [iv]
        for i_neigh in range(n_neigh):
            ie = vert_edge_outgoing[iv][i_neigh]
            jv = edges[1, ie]
            ind_lookup.append(jv)

            edge_vec = edge_tangent_vectors[ie][:]
            w_e = 1.

            lhs_mat[i_neigh][:] = w_e * edge_vec
            rhs_mat[i_neigh][0] = w_e * (-1)
            rhs_mat[i_neigh][i_neigh + 1] = w_e * 1

        lhs_T = lhs_mat.T
        lhs_inv = np.linalg.inv(lhs_T @ lhs_mat + eps_reg * np.identity(2)) @ lhs_T

        sol_mat = lhs_inv @ rhs_mat
        sol_coefs = (sol_mat[0, :] + 1j * sol_mat[1, :]).T

        for i_neigh in range(n_neigh + 1):
            i_glob = ind_lookup[i_neigh]

            row_inds.append(iv)
            col_inds.append(i_glob)
            data_vals.append(sol_coefs[i_neigh])

    # build the sparse matrix
    row_inds = np.array(row_inds)
    col_inds = np.array(col_inds)
    data_vals = np.array(data_vals)
    mat = scipy.sparse.coo_matrix(
        (data_vals, (row_inds, col_inds)),
        shape=(V, V)
    ).tocsc()

    return mat


def laplacian_decomposition(verts, faces, k=150):
    """
    Laplacian decomposition
    Args:
        verts (np.ndarray): vertices [V, 3].
        faces (np.ndarray): faces [F, 3]
        k (int, optional): number of eigenvalues/vectors to compute. Default 120.

    Returns:
        - evals: (k) list of eigenvalues of the Laplacian matrix.
        - evecs: (V, k) list of eigenvectors of the Laplacian.
        - evecs_trans: (k, V) list of pseudo inverse of eigenvectors of the Laplacian.
    """
    assert k >= 0, f'Number of eigenvalues/vectors should be non-negative, bug get {k}'
    is_cloud = (faces is None)
    eps = 1e-8

    # Build Laplacian matrix
    if is_cloud:
        L, M = robust_laplacian.point_cloud_laplacian(verts)
        massvec = M.diagonal()
    else:
        L = pp3d.cotan_laplacian(verts, faces, denom_eps=1e-10)
        massvec = pp3d.vertex_areas(verts, faces)
        massvec += eps * np.mean(massvec)

    if np.isnan(L.data).any():
        raise RuntimeError("NaN Laplace matrix")
    if np.isnan(massvec).any():
        raise RuntimeError("NaN mass matrix")

    # Compute the eigenbasis
    # Prepare matrices
    L_eigsh = (L + eps * scipy.sparse.identity(L.shape[0])).tocsc()
    massvec_eigsh = massvec
    Mmat = scipy.sparse.diags(massvec_eigsh)
    eigs_sigma = eps

    fail_cnt = 0
    while True:
        try:
            evals, evecs = sla.eigsh(L_eigsh, k=k, M=Mmat, sigma=eigs_sigma)
            # Clip off any eigenvalues that end up slightly negative due to numerical error
            evals = np.clip(evals, a_min=0., a_max=float('inf'))
            evals = evals.reshape(-1, 1)
            break
        except:
            if fail_cnt > 3:
                raise ValueError('Failed to compute eigen-decomposition')
            fail_cnt += 1
            print('Decomposition failed; adding eps')
            L_eigsh = L_eigsh + (eps * 10 ** fail_cnt) * scipy.sparse.identity(L.shape[0])

    evecs = np.array(evecs, ndmin=2)
    evecs_trans = evecs.T @ Mmat

    sqrt_area = np.sqrt(Mmat.diagonal().sum())
    return evals, evecs, evecs_trans, sqrt_area


def compute_operators(verts, faces, k=120, normals=None):
    """
    Build spectral operators for a mesh/point cloud.
    Constructs mass matrix, eigenvalues/vectors for Laplacian,
    and gradient matrix.

    Args:
         verts (torch.Tensor): vertices [V, 3].
         faces (torch.Tensor): faces [F, 3]
         k (int, optional): number of eigenvalues/vectors to compute. Default 120.
         normals (torch.Tensor, optional): vertex normals [V, 3]. Default None

    Returns:
        spectral_operators (dict):
            - frames: (V, 3, 3) X/Y/Z coordinate frame at each vertex.
            - massvec: (V) real diagonal of lumped mass matrix.
            - L: (V, V) Laplacian matrix.
            - evals: (k) list of eigenvalues of the Laplacian matrix.
            - evecs: (V, k) list of eigenvectors of the Laplacian.
            - gradX: (V, V) sparse matrix which gives X-component of gradient in the local basis.
            - gradY: (V, V) same as gradX but for Y-component of gradient.

    Note: PyTorch doesn't seem to like complex sparse matrices,
    so we store the "real" and "imaginary" (aka X and Y) gradient matrices separately,
    rather than as one complex sparse matrix.
    """
    assert k >= 0, f'Number of eigenvalues/vectors should be non-negative, bug get {k}'
    device = verts.device
    dtype = verts.dtype
    is_cloud = (faces is None)

    eps = 1e-8

    verts_np = torch2np(verts).astype(np.float64)
    faces_np = torch2np(faces) if faces is not None else None
    frames = build_targent_frames(verts, faces, vert_normals=normals)

    # Build Laplacian matrix
    if is_cloud:
        L, M = robust_laplacian.point_cloud_laplacian(verts_np)
        massvec_np = M.diagonal()
    else:
        L = pp3d.cotan_laplacian(verts_np, faces_np, denom_eps=1e-10)
        massvec_np = pp3d.vertex_areas(verts_np, faces_np)
        massvec_np += eps * np.mean(massvec_np)

    if np.isnan(L.data).any():
        raise RuntimeError("NaN Laplace matrix")
    if np.isnan(massvec_np).any():
        raise RuntimeError("NaN mass matrix")

    # Read off neighbors & rotations from the Laplacian
    L_coo = L.tocoo()
    inds_row = L_coo.row
    inds_col = L_coo.col

    # Compute the eigenbasis
    if k > 0:
        # Prepare matrices
        L_eigsh = (L + eps * scipy.sparse.identity(L.shape[0])).tocsc()
        massvec_eigsh = massvec_np
        Mmat = scipy.sparse.diags(massvec_eigsh)
        eigs_sigma = eps

        fail_cnt = 0
        while True:
            try:
                evals_np, evecs_np = sla.eigsh(L_eigsh, k=k, M=Mmat, sigma=eigs_sigma)
                # Clip off any eigenvalues that end up slightly negative due to numerical error
                evals_np = np.clip(evals_np, a_min=0., a_max=float('inf'))

                break
            except:
                if fail_cnt > 3:
                    raise ValueError('Failed to compute eigen-decomposition')
                fail_cnt += 1
                print('Decomposition failed; adding eps')
                L_eigsh = L_eigsh + (eps * 10 ** fail_cnt) * scipy.sparse.identity(L.shape[0])
    else: # k == 0
        evals_np = np.zeros((0))
        evecs_np = np.zeros((verts.shape[0], 0))

    # Build gradient matrices
    if is_cloud:
        grad_mat_np = build_grad_point_cloud(verts, frames)
    else:
        edges = torch.tensor(np.stack((inds_row, inds_col), axis=0), device=device, dtype=faces.dtype)
        edge_vecs = edge_tangent_vectors(verts, frames, edges)
        grad_mat_np = build_grad(verts_np, torch2np(edges), torch2np(edge_vecs))

    # split complex gradient into two real sparse matrices (PyTorch doesn't like complex sparse matrix)
    gradX_np = np.real(grad_mat_np)
    gradY_np = np.imag(grad_mat_np)

    # convert to torch
    massvec = torch.from_numpy(massvec_np).to(device=device, dtype=dtype)
    L = sparse_np_to_torch(L).to(device=device, dtype=dtype)
    evals = torch.from_numpy(evals_np).to(device=device, dtype=dtype)
    evecs = torch.from_numpy(evecs_np).to(device=device, dtype=dtype)
    gradX = sparse_np_to_torch(gradX_np).to(device=device, dtype=dtype)
    gradY = sparse_np_to_torch(gradY_np).to(device=device, dtype=dtype)

    return frames, massvec, L, evals, evecs, gradX, gradY


def get_operators(verts, faces, k=120, normals=None,
                  cache_dir=None, overwrite_cache=False):
    """
    See documentation for compute_operators().
    This essentailly just wraps a call to compute_operators, using a cache if possible.
    All arrays are always computed using double precision for stability,
    then truncated to single precision floats to store on disk,
    and finally returned as a tensor with dtype/device matching the `verts` input.
    """
    assert verts.dim() == 2, 'Please call get_all_operators() for a batch of vertices'
    device = verts.device
    dtype = verts.dtype
    verts_np = torch2np(verts)
    faces_np = torch2np(faces) if faces is not None else None

    if np.isnan(verts_np).any():
        raise ValueError('detect NaN vertices.')

    found = False
    if cache_dir:
        assert osp.isdir(cache_dir)
        hash_key_str = str(hash_arrays((verts_np, faces_np)))

        # Search through buckets with matching hashes.
        # When the loop exits,
        # this is the bucket index of the file we should write to.
        i_cache = 0
        while True:
            # From the name of the file to check
            search_path = osp.join(cache_dir, hash_key_str+'_'+str(i_cache)+'.npz')

            try:
                npzfile = np.load(search_path, allow_pickle=True)
                cache_verts = npzfile['verts']
                cache_faces = npzfile['faces']
                cache_k = npzfile['k_eig'].item()

                # If the cache doesn't match, keep searching
                if (not np.array_equal(verts, cache_verts)) or (not np.array_equal(faces, cache_faces)):
                    i_cache += 1
                    print('collision detected')
                    continue

                # Delete previous file and overwrite it
                if overwrite_cache or cache_k < k:
                    os.remove(search_path)
                    break

                def read_sp_mat(prefix):
                    data = npzfile[prefix + '_data']
                    indices = npzfile[prefix + '_indices']
                    indptr = npzfile[prefix + '_indptr']
                    shape = npzfile[prefix + '_shape']
                    mat = scipy.sparse.csc_matrix((data, indices, indptr), shape=shape)
                    return mat

                # this entry matches. return it.
                frames = npzfile['frames']
                mass = npzfile['mass']
                L = read_sp_mat('L')
                evals = npzfile['evals'][:k]
                evecs = npzfile['evecs'][:, :k]
                gradX = read_sp_mat('gradX')
                gradY = read_sp_mat('gradY')

                frames = torch.from_numpy(frames).to(device=device, dtype=dtype)
                mass = torch.from_numpy(mass).to(device=device, dtype=dtype)
                L = sparse_np_to_torch(L).to(device=device, dtype=dtype)
                evals = torch.from_numpy(evals).to(device=device, dtype=dtype)
                evecs = torch.from_numpy(evecs).to(device=device, dtype=dtype)
                gradX = sparse_np_to_torch(gradX).to(device=device, dtype=dtype)
                gradY = sparse_np_to_torch(gradY).to(device=device, dtype=dtype)

                found = True
                break
            except FileNotFoundError:
                # not found, create a new file
                break

    if not found:
        # recompute
        frames, mass, L, evals, evecs, gradX, gradY = compute_operators(verts, faces, k, normals)

        dtype_np = np.float32

        # save
        if cache_dir:
            frames_np = torch2np(frames).astype(dtype_np)
            mass_np = torch2np(mass).astype(dtype_np)
            evals_np = torch2np(evals).astype(dtype_np)
            evecs_np = torch2np(evecs).astype(dtype_np)
            L_np = sparse_torch_to_np(L).astype(dtype_np)
            gradX_np = sparse_torch_to_np(gradX).astype(dtype_np)
            gradY_np = sparse_torch_to_np(gradY).astype(dtype_np)

            np.savez(
                search_path,
                verts=verts_np,
                faces=faces_np,
                k_eig=k,
                frames=frames_np,
                mass=mass_np,
                evals=evals_np,
                evecs=evecs_np,
                L_data=L_np.data,
                L_indices=L_np.indices,
                L_indptr=L_np.indptr,
                L_shape=L_np.shape,
                gradX_data=gradX_np.data,
                gradX_indices=gradX_np.indices,
                gradX_indptr=gradX_np.indptr,
                gradX_shape=gradX_np.shape,
                gradY_data=gradY_np.data,
                gradY_indices=gradY_np.indices,
                gradY_indptr=gradY_np.indptr,
                gradY_shape=gradY_np.shape,
            )

    return frames, mass, L, evals, evecs, gradX, gradY


def get_all_operators(verts, faces, k=120,
                      normals=None,
                    cache_dir=None):
    """
    Get all operators from batch
    """
    assert verts.dim() == 3, 'please call get_operators() for a single vertices'

    B = verts.shape[0]

    frames = []
    mass = []
    L = []
    evals = []
    evecs = []
    gradX = []
    gradY = []

    for i in range(B):
        if faces is not None:
            if normals is not None:
                output = get_operators(verts[i], faces[i], k, normals[i], cache_dir)
            else:
                output = get_operators(verts[i], faces[i], k, None, cache_dir)
        else:
            if normals is not None:
                output = get_operators(verts[i], None, k, normals[i], cache_dir)
            else:
                output = get_operators(verts[i], None, k, None, cache_dir)
        frames += [output[0]]
        mass += [output[1]]
        L += [output[2]]
        evals += [output[3]]
        evecs += [output[4]]
        gradX += [output[5]]
        gradY += [output[6]]

    frames = torch.stack(frames)
    mass = torch.stack(mass)
    L = torch.stack(L)
    evals = torch.stack(evals)
    evecs = torch.stack(evecs)
    gradX = torch.stack(gradX)
    gradY = torch.stack(gradY)

    return frames, mass, L, evals, evecs, gradX, gradY


def compute_hks_autoscale(evals, evecs, count=16):
    """
    Compute heat kernel signature with auto-scale
    Args:
        evals (torch.Tensor): eigenvalues of Laplacian matrix [B, K]
        evecs (torch.Tensor): eigenvecetors of Laplacian matrix [B, V, K]
        count (int, optional): number of hks. Default 16.
    Returns:
        out (torch.Tensor): heat kernel signature [B, V, count]
    """
    scales = torch.logspace(-2.0, 0.0, steps=count, device=evals.device, dtype=evals.dtype)

    power_coefs = torch.exp(-evals.unsqueeze(1) * scales.unsqueeze(-1)).unsqueeze(1) # [B, 1, S, K]
    terms = power_coefs * (evecs * evecs).unsqueeze(2) # [B, V, S, K]

    out = torch.sum(terms, dim=-1) # [B, V, S]

    return out


def wks(evals, evecs, energy_list, sigma, scaled=False):
    assert sigma > 0, f"Sigma should be positive ! Given value : {sigma}"

    indices = (evals > 1e-5)
    evals = evals[indices]
    evecs = evecs[:, indices]

    coefs = torch.exp(-torch.square(energy_list[:, None] - torch.log(torch.abs(evals))[None, :]) / (2 * sigma ** 2))

    weighted_evecs = evecs[None, :, :] * coefs[:, None, :]
    wks = torch.einsum('tnk,nk->nt', weighted_evecs, evecs)

    if scaled:
        inv_scaling = coefs.sum(1)
        return (1 / inv_scaling)[None, :] * wks
    else:
        return wks


def auto_wks(evals, evecs, n_descr, scaled=True):
    abs_ev = torch.sort(evals.abs())[0]
    e_min, e_max = torch.log(abs_ev[1]), torch.log(abs_ev[-1])
    sigma = 7 * (e_max - e_min) / n_descr

    e_min += 2 * sigma
    e_max -= 2 * sigma

    energy_list = torch.linspace(float(e_min), float(e_max), n_descr, device=evals.device, dtype=evals.dtype)

    return wks(abs_ev, evecs, energy_list, sigma, scaled=scaled)


def compute_wks_autoscale(evals, evecs, mass, n_descr=128, subsample_step=1, n_eig=128):
    feats = []
    for b in range(evals.shape[0]):
        feat = auto_wks(evals[b, :n_eig], evecs[b, :, :n_eig], n_descr, scaled=True)
        feat = feat[:, torch.arange(0, feat.shape[1], subsample_step)]
        feat_norm = torch.einsum('np,np->p', feat, mass[b].unsqueeze(1) * feat)
        feat /= torch.sqrt(feat_norm)
        feats += [feat]
    feats = torch.stack(feats, dim=0)
    return feats


def euler_angles_to_rotation_matrix(theta):
    R_x = torch.tensor([[1, 0, 0], [0, torch.cos(theta[0]), -torch.sin(theta[0])], [0, torch.sin(theta[0]), torch.cos(theta[0])]])
    R_y = torch.tensor([[torch.cos(theta[1]), 0, torch.sin(theta[1])], [0, 1, 0], [-torch.sin(theta[1]), 0, torch.cos(theta[1])]])
    R_z = torch.tensor([[torch.cos(theta[2]), -torch.sin(theta[2]), 0], [torch.sin(theta[2]), torch.cos(theta[2]), 0], [0, 0, 1]])

    matrices = [R_x, R_y, R_z]

    R = torch.mm(matrices[2], torch.mm(matrices[1], matrices[0]))
    return R


def get_random_rotation(x, y, z):
    thetas = torch.zeros(3, dtype=torch.float)
    degree_angles = [x, y, z]
    for axis_ind, deg_angle in enumerate(degree_angles):
        rand_deg_angle = random.random() * 2 * deg_angle - deg_angle
        rand_radian_angle = float(rand_deg_angle * np.pi) / 180.0
        thetas[axis_ind] = rand_radian_angle

    return euler_angles_to_rotation_matrix(thetas)


def data_augmentation(verts, rot_x=0, rot_y=90.0, rot_z=0, std=0.01, noise_clip=0.05, scale_min=0.9, scale_max=1.1):
    # random rotation
    rotation_matrix = get_random_rotation(rot_x, rot_y, rot_z).repeat(verts.shape[0], 1, 1).to(verts.device)
    verts = torch.bmm(verts, rotation_matrix.transpose(1, 2))

    # random noise
    noise = std * torch.randn(verts.shape).to(verts.device)
    noise = noise.clamp(-noise_clip, noise_clip)
    verts += noise

    # random scaling
    scales = [scale_min, scale_max]
    scale = scales[0] + torch.rand((3,)) * (scales[1] - scales[0])
    verts = verts * scale.to(verts.device)

    return verts
