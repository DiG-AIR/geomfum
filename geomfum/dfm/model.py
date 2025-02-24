
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
        super(ProperMapNet, self).__init__()
        self.config = config
        self.desc_model = LearnedDescriptor.from_registry(**config['descriptor']['params'],which=config['descriptor']['type'], device=device)
        self.fmap = ForwardFunctionalMap(**config['forward_map'])

        self.perm = PermutationModule()

    def forward(self, source, target):
        desc_a = self.desc_model(source)
        desc_b = self.desc_model(target)
        C = self.fmap(source, target, desc_a, desc_b)
        P12 = self.perm( source['basis'],target['basis']@C)
        C_p= torch.bmm(target['pinv'],torch.bmm(P12,source['basis']))

        return {"Cxy":C,"Cxy_sup": C_p}
        
@register_model('CaoNet')
class CaoNet(BaseModel):
    def __init__(self, config,device='cuda'):
        super(CaoNet, self).__init__()
        self.config = config
        self.desc_model = LearnedDescriptor.from_registry(**config['descriptor']['params'],which=config['descriptor']['type'], device=device)
        self.fmap = ForwardFunctionalMap(**config['forward_map'])
        self.perm = PermutationModule()

    def forward(self, source, target):
        desc_a = self.desc_model(source)
        desc_b = self.desc_model(target)
        C = self.fmap(source, target, desc_a, desc_b)
        P12 = self.perm(desc_a, desc_b)
        C_p= torch.bmm(target['pinv'],torch.bmm(P12,source['basis']))
        return {"Cxy":C,"Cxy_sup": C_p}