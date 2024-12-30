from .AVSegFormer import AVSegFormer
from .UnifiedModel import UnifiedModel


def build_model(type, **kwargs):
    if type == 'AVSegFormer':
        return AVSegFormer(**kwargs)
    elif type == 'UnifiedModel':
        return UnifiedModel(**kwargs)
    else:
        raise ValueError
