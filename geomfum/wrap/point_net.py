"""
This is the wrapper of the PointNet model.
#TODO: Add references
#TODO: For the moment we assume to have the implementation of PointNet somewhere in the code
"""

from geomfum.feature_extractor.point_net.pointnet import PointNet  
from geomfum.descriptor import Descriptor
import torch

class PointNetDescriptor(Descriptor):
    """Descriptor representing the output of PointNet."""

    def __init__(self, k=128,device=torch.device('cpu'), feature_transform=False):
        super(PointNetDescriptor, self).__init__()
        self.model = PointNet(k=k, feature_transform=feature_transform).to(device)
        self.n_features = k
        self.device = device

    def __call__(self, mesh):
        """Process the point cloud data using PointNet."""
        with torch.no_grad():
            point_cloud = torch.tensor(mesh.vertices, dtype=torch.float32)
            if point_cloud.ndimension() == 2:
                point_cloud = point_cloud.unsqueeze(0)
            self.features = self.model(point_cloud.transpose(2,1))
        return self.features
    
    def load_from_path(self, path):
        #load model parameters from the provided path
        self.model.load_state_dict(torch.load(path,map_location=torch.device('cpu')))
    
    def load(self, premodel):
        #load model parameters from the provided path
        self.model.load_state_dict(premodel)
