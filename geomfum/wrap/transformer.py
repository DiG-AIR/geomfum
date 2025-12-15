"""Transformer for Feature Extraction.

Copyright (c) 2025 Alessandro Riva

References
----------
[RRM2024] Riva, A., Raganato, A, Melzi, S., "Localized Gaussians as Self-Attention Weights for Point Clouds Correspondence",
            Smart Tools and Applications in Graphics, 2024, arXiv:2409.13291
https://github.com/ariva00/aided_transformer/
"""

import gsops.backend as gs
import torch
import torch.nn as nn

from geomfum.descriptor.learned import BaseFeatureExtractor


class TransformerFeatureExtractor(BaseFeatureExtractor, nn.Module):
    """Transformer Feature Extractor for point clouds.

    This feature extractor uses the Transformer architecture to extract
    features from shapes.

    Parameters
    ----------
        in_channels: int
            Input feature dimension (typically 3 for xyz coordinates)
        embed_dim: int
            Embedding dimension for the transformer
        num_heads: int
            Number of attention heads
        num_layers: int
            Number of transformer layers
        output_dim: int
            Output feature dimension
        dropout: float
            Dropout probability
        use_global_pool: bool
            Whether to use global max pooling for global features
    """

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        out_channels: int = 512,
        dropout: float = 0.1,
        use_global_pool: bool = True,
        k_neighbors: int = 16,
        device=None,
        descriptor=None,
    ):
        super(TransformerFeatureExtractor, self).__init__()

        self.device = torch.device(device) if device else torch.device("cpu")
        self.descriptor = descriptor

        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.out_channels = out_channels
        self.use_global_pool = use_global_pool
        self.k_neighbors = k_neighbors

        # Input projection
        self.input_projection = (
            nn.Sequential(
                nn.Linear(in_channels, embed_dim // 2),
                nn.ReLU(),
                nn.Linear(embed_dim // 2, embed_dim),
            )
            .to(self.device)
            .float()
        )

        # Transformer
        self.transformer = (
            Transformer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                dropout=dropout,
            )
            .to(self.device)
            .float()
        )

        self.output_projection = (
            nn.Sequential(nn.Linear(embed_dim, out_channels)).to(self.device).float()
        )

    def forward(self, shape, return_intermediate=False):
        """Forward pass of the feature extractor.

        Parameters
        ----------
            shape: Shape
                Input shape

        Returns
        -------
            features: torch.Tensor
                Extracted features tensor of shape (batch_size, num_points, output_dim)
        """
        if self.descriptor is None:
            input_feat = shape.vertices
        else:
            input_feat = self.descriptor(shape).T

        xyz = gs.to_torch(input_feat).to(self.device).float().unsqueeze(0)
        input_feat = gs.to_torch(input_feat).to(self.device).float()
        input_feat = input_feat.unsqueeze(0)
        # Project input features
        x = self.input_projection(input_feat)  # (B, N, embed_dim)

        # Apply transformer (self-attention: x attends to itself)
        transformer_output = self.transformer(x, x)  # (B, N, embed_dim)

        # Apply output projection
        point_features = self.output_projection(
            transformer_output
        )  # (B, N, output_dim)
        if return_intermediate:
            return {
                "point_features": point_features,
                "transformer_features": transformer_output,
            }
        return point_features


class MultiHeadAttention(torch.nn.Module):
    """Multi-Head Attention layer.

    Parameters
    ----------
        embed_dim: int
            Dimension of the used embedding
        num_heads: int
            Number of attention heads
        dropout: float
            Dropout rate
        bias: bool
            Set the use of leaned bias in the output linear layer of the attention heads
    """

    def __init__(
        self, embed_dim, num_heads: int, dropout: float = 0.0, bias: bool = True
    ):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = torch.nn.Dropout(dropout)
        self.linear_out = torch.nn.Linear(embed_dim, embed_dim, bias=bias)
        self.scale = self.head_dim**-0.5

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor = None,
        attn_prev: torch.Tensor = None,
    ):
        """Forward pass of the Multi-Head Attention layer.

        Parameters
        ----------
            query: torch.Tensor
                Query vectors of x
            key: torch.Tensor
                Key vectors of y
            value: torch.Tensor
                Value vectors of y
            attn_mask: torch.Tensor
                Mask tensor for the attention weights of shape (batch_size, num_heads, num_points_x, num_points_y) or (batch_size, num_points_x, num_points_y).
                If (batch_size, num_points_x, num_points_y) the mask will be broadcasted over the num_heads dimension
            attn_prev: torch.Tensor
                Attention weights of the previous layer to be used in the residual attention of shape (batch_size, num_heads, num_points_x, num_points_y)


        Returns
        -------
            output: torch.Tensor
                Attention output tensor of shape (batch_size, num_points_x, embed_dim)
            hiddens: dict
                Intermediate activations of the attention mechanism

        """
        query = query.reshape(
            query.size(0), self.num_heads, query.size(1), self.head_dim
        )
        key = key.reshape(key.size(0), self.num_heads, key.size(1), self.head_dim)
        value = value.reshape(
            value.size(0), self.num_heads, value.size(1), self.head_dim
        )

        attn = torch.matmul(query, key.transpose(2, 3))

        attn = attn * self.scale

        attn = self.dropout(attn)

        if attn_mask is not None:
            if attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(1)
            attn = attn.masked_fill(attn_mask == 0, -1e9)

        pre_softmax_attn = attn
        attn = attn.softmax(dim=-1)
        attn = attn + attn_prev if attn_prev is not None else attn
        output = torch.matmul(attn, value)
        output = output.transpose(1, 2).reshape(
            output.size(0), output.size(2), self.embed_dim
        )
        output = self.linear_out(output)

        hiddens = {
            "q": query,
            "k": key,
            "v": value,
            "attn": attn,
            "pre_softmax_attn": pre_softmax_attn,
        }
        return output, hiddens


class AttentionLayer(torch.nn.Module):
    """Attention Layer for the Transformer.

    Parameters
    ----------
        embed_dim: int
            Dimension of the used embedding
        num_heads: int
            Number of attention heads
        dropout: float
            Dropout rate
        attn_bias: bool
            Set the use of leaned bias in the output linear layer of the attention heads
        ff_mult: int
            Dimension factor of the feed forward section of the attention layer. The dimension expansion is computed as ff_mult * embed_dim
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout: float = 0.0,
        attn_bias: bool = True,
        ff_mult: int = 4,
    ):
        super(AttentionLayer, self).__init__()
        self.attn = MultiHeadAttention(
            embed_dim,
            num_heads,
            dropout,
            attn_bias,
        )
        self.norm1 = torch.nn.LayerNorm(embed_dim)
        self.norm2 = torch.nn.LayerNorm(embed_dim)
        self.to_q = torch.nn.Linear(embed_dim, embed_dim)
        self.to_k = torch.nn.Linear(embed_dim, embed_dim)
        self.to_v = torch.nn.Linear(embed_dim, embed_dim)
        self.feed_forward = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, embed_dim * ff_mult),
            torch.nn.GELU(),
            torch.nn.Linear(embed_dim * ff_mult, embed_dim),
        )

    def forward(self, x, y, attn_mask=None, x_mask=None, y_mask=None, attn_prev=None):
        """Forward pass of the Attention Layer.

        Parameters
        ----------
            x: torch.Tensor
                Tokens of x tensor of shape (batch_size, num_points_x, embed_dim)
            y: torch.Tensor
                Tokens of y tensor of shape (batch_size, num_points_y, embed_dim)
            attn_mask: torch.Tensor
                Mask tensor for the attention weights of shape (batch_size, num_heads, num_points_x, num_points_y) or (batch_size, num_points_x, num_points_y).
                If (batch_size, num_points_x, num_points_y) the mask will be broadcasted over the num_heads dimension
            x_mask: torch.Tensor
                Mask tensor of x of shape (batch_size, num_points_x)
            y_mask: torch.Tensor
                Mask tensor of y of shape (batch_size, num_points_y)
            attn_prev: torch.Tensor
                Attention weights of the previous layer to be used in the residual attention of shape (batch_size, num_heads, num_points_x, num_points_y)


        Returns
        -------
            output: torch.Tensor
                Attention output tensor of shape (batch_size, num_points_x, embed_dim)
            hiddens: dict
                Intermediate activations of the attention mechanism
        """
        if x_mask is not None:
            if y_mask is None:
                y_mask = x_mask
            input_mask = (
                (x_mask.float().unsqueeze(-1))
                .bmm(y_mask.float().unsqueeze(-1).transpose(-1, -2))
                .long()
                .unsqueeze(1)
            )
            if attn_mask is None:
                attn_mask = input_mask
            else:
                attn_mask = attn_mask & input_mask
        attn_output, hiddens = self.attn(
            self.to_q(x),
            self.to_k(y),
            self.to_v(y),
            attn_mask=attn_mask,
            attn_prev=attn_prev,
        )
        attn_output = self.norm1(x + attn_output)
        output = self.feed_forward(attn_output)
        output = self.norm2(attn_output + output)
        return output, hiddens


class Transformer(torch.nn.Module):
    """Transformer for feature extraction.

    Parameters
    ----------
        embed_dim: int
            Dimension of the used embedding
        num_heads: int
            Number of attention heads
        num_layers: int
            Number of attention layers
        dropout: float
            Dropout rate
        residual: bool
            Boolean control the use of the residual attention
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        num_layers,
        dropout=0.0,
        residual=False,
    ):
        super(Transformer, self).__init__()
        self.layers = torch.nn.ModuleList(
            [
                AttentionLayer(
                    embed_dim,
                    num_heads,
                    dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.residual = residual

    def forward(
        self, x, y, attn_mask=None, x_mask=None, y_mask=None, return_hiddens=False
    ):
        """Forward pass of the Transformer.

        Parameters
        ----------
            x: torch.Tensor
                Tokens of x tensor of shape (batch_size, num_points_x, embed_dim)
            y: torch.Tensor
                Tokens of y tensor of shape (batch_size, num_points_y, embed_dim)
            attn_mask: torch.Tensor
                Mask tensor for the attention weights of shape (batch_size, num_heads, num_points_x, num_points_y) or (batch_size, num_points_x, num_points_y).
                If (batch_size, num_points_x, num_points_y) the mask will be broadcasted over the num_heads dimension
            x_mask: torch.Tensor
                Mask tensor of x of shape (batch_size, num_points_x)
            y_mask: torch.Tensor
                Mask tensor of y of shape (batch_size, num_points_y)
            return_hiddens: bool
                Boolean to return intermediate activations of the attention mechanism. If True the output is a tuple (output, hiddens)


        Returns
        -------
            output: torch.Tensor
                Attention output tensor of shape (batch_size, num_points_x, embed_dim)
            hiddens: dict
                Intermediate activations of the attention mechanism
        """
        attn_hiddens = []
        for layer in self.layers:
            if self.residual and len(attn_hiddens) > 0:
                attn_prev = attn_hiddens[-1]["attn"]
            else:
                attn_prev = None
            output, hiddens = layer(
                x,
                y,
                attn_mask=attn_mask,
                x_mask=x_mask,
                y_mask=y_mask,
                attn_prev=attn_prev,
            )
            attn_hiddens.append(hiddens)
        if return_hiddens:
            return output, attn_hiddens
        else:
            return output
