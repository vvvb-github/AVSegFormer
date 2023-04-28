from .resnet import B2_ResNet
from .pvt import pvt_v2_b5


def build_backbone(type, **kwargs):
    if type == 'res50':
        return B2_ResNet(**kwargs)
    elif type=='pvt_v2_b5':
        return pvt_v2_b5(**kwargs)
    
    
__all__=['build_backbone']
