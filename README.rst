
# GEOMFUM

![Alt text](https://github.com/user-attachments/assets/c96d2596-03bf-4361-a79d-3e83515d5c0b)

A Modular Python Package for Machine Learning with Functional Maps  
<https://dl.acm.org/doi/10.1145/2185520.2185526>

## Installation

```bash
pip install geomfum@git+https://github.com/luisfpereira/geomfum.git@main
```

Or the classic pipeline: `clone + pip install .`

> ⚠️ Installation issues may arise from dependencies relying on C++  
> (particularly [`robust_laplacian`](https://pypi.org/project/robust-laplacian/)).  
> Make sure all their requirements are installed.  
>
> For `pyRMT`, follow the instructions [here](https://github.com/filthynobleman/rematching/tree/python-binding).

## List of Implemented Papers

1. [Functional Maps: A Flexible Representation of Maps Between Shapes](http://www.lix.polytechnique.fr/~maks/papers/obsbg_fmaps.pdf)
2. [Rematching: Low-resolution representations for scalable shape correspondence](https://arxiv.org/abs/2305.09274)
3. [ZoomOut: Spectral Upsampling for Efficient Shape Correspondence](https://arxiv.org/abs/1904.07865)
4. [Fast Sinkhorn Filters: Using Matrix Scaling for Non-Rigid Shape Correspondence with Functional Maps](https://openaccess.thecvf.com/content/CVPR2021/html/Pai_Fast_Sinkhorn_Filters_Using_Matrix_Scaling_for_Non-Rigid_Shape_Correspondence_CVPR_2021_paper.html)
5. [Structured regularization of functional map computations](https://www.lix.polytechnique.fr/Labo/Ovsjanikov.Maks/papers/resolvent_SGP19_small.pdf)
6. [Bijective upsampling and learned embedding for point clouds correspondences](https://www.sciencedirect.com/science/article/pii/S0097849324001201)
7. [Deep Geometric Functional Maps: Robust Feature Learning for Shape Correspondence](https://arxiv.org/abs/2003.14286)
8. [Laplace-Beltrami Eigenfunctions Towards an Algorithm That "Understands" Geometry](https://brunolevy.github.io/papers/Laplacian_SMI_2006.pdf)
9. [The Heat Method for Distance Computation](https://www.cs.cmu.edu/~kmcrane/Projects/HeatMethod/)
10. [A Concise and Provably Informative Multi-Scale Signature Based on Heat Diffusion](http://www.lix.polytechnique.fr/~maks/papers/hks.pdf)
11. [The Wave Kernel Signature: A Quantum Mechanical Approach To Shape Analysis](http://imagine.enpc.fr/~aubrym/projects/wks/index.html)
12. [Informative Descriptor Preservation via Commutativity for Shape Matching](https://www.lix.polytechnique.fr/Labo/Ovsjanikov.Maks/papers/fundescEG17.pdf)
13. [DiffusionNet: Discretization Agnostic Learning on Surfaces](https://arxiv.org/abs/2012.00888)
14. [PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation](https://arxiv.org/abs/1612.00593)

How to use
----------

The `how-to notebooks <./notebooks/how_to>`_ are designed to safely let you dive in the package.
Why not starting from the `beginning <./notebooks/how_to/load_mesh_from_file.ipynb>`_ and simply follow the links that inspire you the most?

Have fun!
