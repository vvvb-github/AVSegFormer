from .AVSegFormer import AVSegFormer
from .TAVSegFormer import TAVSegFormer


def build_model(type, **kwargs):
    if type == 'AVSegFormer':
        return AVSegFormer(**kwargs)
    elif type=='TAVSegFormer':
        return TAVSegFormer(**kwargs)
    else:
        raise ValueError
