"""Neural Adjoint Maps (NAMs) for functional maps."""

import torch.nn as nn


class NeuralAdjointMap(nn.Module):
    """
    Neural Adjoint Map (NAM) composed by a linear branch and a non-linear MLP branch.

    Parameters
    ----------
    input_dim : int
        The dimension of the input data.
    output_dim : int
        The dimension of the output data. If None, it defaults to input_dim.
    depth : int
        The number of layers in the MLP.
    width : int
        The width of each layer in the MLP.
    act : torch.nn.Module
        The activation function to be used in the MLP.

    References
    ----------
    .. "NAM: Neural Adjoint Maps for refining shape correspondences" by Giulio Viganò, Maks Ovsjanikov, Simone Melzi.
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        linear_module=None,
        non_linear_module=None,
        device="cpu",
    ):
        super().__init__()

        if output_dim is None:
            output_dim = input_dim

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.shape = (output_dim, input_dim)
        self.device = device

        # Linear Module
        self.linear_module = linear_module
        if self.linear_module is None:
            self.linear_module = nn.Linear(input_dim, output_dim, bias=False).to(
                self.device
            )

        # Non-linear MLP Module
        self.nonlinear_module = non_linear_module
        if self.nonlinear_module is None:
            self.nonlinear_module = MLP(
                input_dim=input_dim,
                output_dim=output_dim,
                depth=2,
                width=128,
                act=nn.LeakyReLU(),
            ).to(self.device)
        # Apply small scaling to MLP output for initialization
        self.mlp_scale = 0.01
        self._reset_parameters()

    def forward(self, x):
        """Forward pass through both the linear and non-linear modules."""
        x = x[:, : self.input_dim]

        fmap = self.linear_module(x)
        t = self.mlp_scale * self.nonlinear_module(x)
        x_out = fmap + t

        return x_out.squeeze()

    def _reset_parameters(self):
        """Initialize the model parameters using Xavier uniform distribution."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


class MLP(nn.Module):
    """
    A simple MLP (Multi-Layer Perceptron) module.

    Parameters
    ----------
    input_dim : int
        The dimension of the input data.
    output_dim : int
        The dimension of the output data.
    depth : int
        The number of layers in the MLP.
    width : int
        The width of each layer in the MLP.
    act : torch.nn.Module
        The activation function to be used in the MLP.
    """

    def __init__(
        self, input_dim, output_dim, depth=4, width=128, act=nn.LeakyReLU(), bias=True
    ):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for _ in range(depth):
            layers.append(nn.Linear(prev_dim, width, bias=bias))
            layers.append(act)  # Add activation after each layer
            prev_dim = width
        layers.append(nn.Linear(prev_dim, output_dim, bias=bias))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass through the MLP."""
        return self.mlp(x)
