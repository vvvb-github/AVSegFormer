from .AVSegHead import AVSegHead
from .TAVSHead import TAVSHead


def build_head(type, **kwargs):
    if type == 'AVSegHead':
        return AVSegHead(**kwargs)
    elif type=='TAVSHead':
        return TAVSHead(**kwargs)
    else:
        raise ValueError


__all__ = ['build_head']
