"""
This file contains the implementation of different model that can be built using the geomfum library.
These are just some example that can be accoplished using the modules built. We report some of the main implementation from the literature.

"""
import torch
from geomfum.descriptor.learned import LearnedDescriptor
from geomfum.dfm.forward_functional_map import ForwardFunctionalMap
from geomfum.dfm.permutation import PermutationModule


class BaseModel(torch.nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
    
class FMNet(torch.nn.Module):
    def __init__(self, config,device='cuda'):
        """
        This is the simplest deep functional map model. It is composed by a descriptor and a forward map.
        Deep Geometric Functional Maps: Robust Feature Learning for Shape Correspondence, Nicolas Donati, Abhishek Sharma, Maks Ovsjanikov 2020
        """
        super(FMNet, self).__init__()
        
        self.config = config
        self.desc_model = LearnedDescriptor.from_registry(**config['descriptor']['params'],which=config['descriptor']['type'], device=device)
        self.fmap = ForwardFunctionalMap(**config['forward_map'])

    def forward(self, source, target):
        desc_a = self.desc_model(source)
        desc_b = self.desc_model(target)
        Cab,Cba= self.fmap(source, target, desc_a, desc_b)
        return {"Cab":Cab,"Cba":Cba}


    
class ProperMapNet(BaseModel):
    def __init__(self, config,device='cuda'):
        """
        This is deep functional map model returns a proper functional map.
        reference:
        Understanding and Improving Features Learned in Deep Functional Maps, Souhaib Attaiki, Maks Ovsjanikov, 2023
        """
        super(ProperMapNet, self).__init__()
        self.config = config
        self.desc_model = LearnedDescriptor.from_registry(**config['descriptor']['params'],which=config['descriptor']['type'], device=device)
        self.fmap = ForwardFunctionalMap(**config['forward_map'])

        self.perm = PermutationModule()

    def forward(self, source, target):
        desc_a = self.desc_model(source)
        desc_b = self.desc_model(target)
        Cab,Cba  = self.fmap(source, target, desc_a, desc_b)
        Pab = self.perm( source['basis'],target['basis']@Cab)
        C_p= torch.bmm(target['pinv'],torch.bmm(Pab,source['basis']))

        return {"Cab":Cab,"Cba":Cba,"Cab_sup": C_p}
    
        
class CaoNet(BaseModel):
    def __init__(self, config,device='cuda'):
        """
        This functional map model returns a functional map and a map obtained by the similarity of the descriptors.
        Reference:
        Unsupervised Learning of Robust Spectral Shape Matching , Dongliang Cao, Paul Roetzer, Florian Bernard 2023
        """
        super(CaoNet, self).__init__()
        self.config = config
        self.desc_model = LearnedDescriptor.from_registry(**config['descriptor']['params'],which=config['descriptor']['type'], device=device)
        self.fmap = ForwardFunctionalMap(**config['forward_map'])
        self.perm = PermutationModule()

    def forward(self, source, target):
        desc_a = self.desc_model(source)
        desc_b = self.desc_model(target)
        Cab,Cba  = self.fmap(source, target, desc_a, desc_b)
        Pba = self.perm(desc_a, desc_b)
        Pab = self.perm(desc_b, desc_a)

        Cba_p= torch.bmm(target['pinv'],torch.bmm(Pab,source['basis']))
        Cab_p= torch.bmm(source['pinv'],torch.bmm(Pba,target['basis']))

        return {"Cab":Cab,"Cba":Cba,"Cab_sup": Cab_p,"Cba_sup":Cba_p}