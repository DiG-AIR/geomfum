"""
This is the wrapper of the PointNet model.
#TODO: Add references
#TODO: For the moment we assume to have the implementation of PointNet somewhere in the code
"""

from geomfum.feature_extractor.point_net.pointnet import PointNet  
from geomfum.descriptor._base import LearnedDescriptor
from geomfum.shape.mesh import TriangleMesh
import torch


class PointNetDescriptor(LearnedDescriptor):
    """Descriptor representing the output of PointNet."""

    def __init__(self, k=128,device=torch.device('cpu'), feature_transform=False):
        super(PointNetDescriptor, self).__init__()
        self.model = PointNet(k=k, feature_transform=feature_transform).to(device)
        self.n_features = k
        self.device = device

    def __call__(self, mesh):
        """Process the point cloud data using PointNet."""
        if isinstance(mesh, dict):
            # If input is a dictionary containing tensors
            v = mesh['vertices'].to(torch.float32) 
        elif isinstance(mesh, TriangleMesh):
        # If input is a TriangleMesh object, extract vertices and faces
            v = mesh.vertices[None].to(torch.float32) #Add batch dimension
        else:
            raise TypeError("Input must be either a TriangleMesh or a dictionary containing 'vertices' and 'faces'")

        with torch.no_grad():
            point_cloud = v.to(torch.float32)
            #ADDITIONAL CHECK ON THE DIMENSION
            if point_cloud.ndimension() == 2:
                point_cloud = point_cloud.unsqueeze(0)
            self.features = self.model(point_cloud.transpose(2,1))
        # for the moment the function outputs a numpy array of dimension DxN
        return self.features

    def load_from_path(self, path):
        #load model parameters from the provided path
        self.model.load_state_dict(torch.load(path,map_location=torch.device('cpu')))
    
    def load(self, premodel):
        #load model parameters from the provided path
        self.model.load_state_dict(premodel)
