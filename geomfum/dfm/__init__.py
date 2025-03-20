from geomfum._registry import (
    register_loss,
    register_model,
)


register_loss(
    "Frobenius", "SquaredFrobeniusLoss", requires="torch", as_default=True
)

register_loss(
    "Orthonormality", "OrthonormalityLoss", requires="torch", as_default=True
)

register_loss(
    "Bijectivity", "BijectivityLoss", requires="torch", as_default=True
)

register_loss(
    "Fmap_Supervision", "Fmap_Supervision", requires="torch", as_default=True
)

register_loss(
    "Laplacian_Commutativity", "LaplacianCommutativityLoss", requires="torch", as_default=True
)

register_loss(
    "Geodesic_Eval", "Geodesic_Evaluation", requires="torch", as_default=True
)

register_model(
    "VanillaFMNet", "FMNet", requires="torch", as_default=True
)


register_model(
    "ProperMapNet", "ProperMapNet", requires="torch", as_default=True
)


register_model(
    "CaoNet", "CaoNet", requires="torch", as_default=True
)


