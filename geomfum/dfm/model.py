"""
This file contains the implementation of different model that can be built using the geomfum library.
These are just some example that can be accoplished using the modules built. We report some of the main implementation from the literature.

"""
import torch
from geomfum.descriptor.learned import LearnedDescriptor
from geomfum.dfm.forward_functional_map import ForwardFunctionalMap
from geomfum.dfm.permutation import PermutationModule



MODEL_REGISTRY = {}

def register_model(name):
    def decorator(cls):
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator

def get_model_class(name):
    return MODEL_REGISTRY.get(name)

class BaseModel(torch.nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
    
@register_model('VanillaFMNet')
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
        C = self.fmap(source, target, desc_a, desc_b)
        return {"Cxy":C}


    
@register_model('ProperMapNet')
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
        Cxy,Cyx  = self.fmap(source, target, desc_a, desc_b)
        P12 = self.perm( source['basis'],target['basis']@Cxy)
        C_p= torch.bmm(target['pinv'],torch.bmm(P12,source['basis']))

        return {"Cxy":Cxy,"Cyx":Cyx,"Cxy_sup": C_p}
    
        
@register_model('CaoNet')
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
        Cxy,Cyx  = self.fmap(source, target, desc_a, desc_b)
        Pxy = self.perm(desc_a, desc_b)
        C_p= torch.bmm(target['pinv'],torch.bmm(Pxy,source['basis']))
        return {"Cxy":Cxy,"Cyx":Cyx,"Cxy_sup": C_p}